import torch
import torch.profiler as profiler
from torch.profiler import ProfilerActivity
import numpy as np

import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ✅ FIX: Import corretto
try:
    # Quando eseguito come modulo del package
    from core.model.net import RSCDMamba
except ImportError:
    try:
        # Quando eseguito direttamente
        from model.net import RSCDMamba
    except ImportError:
        # Ultimo tentativo: import relativo
        from ...model.net import RSCDMamba

def profile_model(model, input_shape=(1, 3, 224, 224), num_iterations=10):
    """
    Profila il modello e mostra dove viene speso il tempo
    
    Args:
        model: Il modello da profilare
        input_shape: Shape dell'input (batch, channels, height, width)
        num_iterations: Numero di iterazioni da profilare
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()  # Importante: metti in training mode
    
    # Crea un input dummy
    x = torch.randn(input_shape, device=device)
    
    # Warmup
    print("Warmup...")
    for _ in range(3):
        output = model(x)
        loss = output.sum()
        loss.backward()
        model.zero_grad()
    
    print(f"\nProfilazione in corso con {num_iterations} iterazioni...")
    
    # Profiling con context manager
    with profiler.profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        with profiler.record_function("model_training"):
            for i in range(num_iterations):
                # Forward pass
                with profiler.record_function("forward"):
                    output = model(x)
                
                # Loss e backward
                with profiler.record_function("backward"):
                    loss = output.sum()
                    loss.backward()
                
                # Simula optimizer step (senza effettivamente aggiornare)
                with profiler.record_function("optimizer_step"):
                    model.zero_grad()
    
    return prof


def print_profile_results(prof, top_k=20):
    """
    Stampa i risultati del profiling in vari formati
    """
    print("\n" + "="*80)
    print("TOP OPERAZIONI PER TEMPO CPU")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="cpu_time_total", 
        row_limit=top_k
    ))
    
    if torch.cuda.is_available():
        print("\n" + "="*80)
        print("TOP OPERAZIONI PER TEMPO CUDA")
        print("="*80)
        print(prof.key_averages().table(
            sort_by="cuda_time_total", 
            row_limit=top_k
        ))
    
    print("\n" + "="*80)
    print("TOP OPERAZIONI PER USO MEMORIA")
    print("="*80)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage", 
        row_limit=top_k
    ))
    
    print("\n" + "="*80)
    print("STATISTICHE PER FUNZIONE CUSTOM")
    print("="*80)
    print(prof.key_averages(group_by_stack_n=5).table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=top_k
    ))


def export_chrome_trace(prof, filename="trace.json"):
    """
    Esporta il trace per visualizzazione in Chrome
    """
    prof.export_chrome_trace(filename)
    print(f"\n✓ Trace esportato in '{filename}'")
    print(f"  Aprilo in Chrome digitando: chrome://tracing")


def analyze_specific_layers(model, input_shape=(1, 3, 224, 224)):
    """
    Analizza layer specifici del modello
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()
    
    x = torch.randn(input_shape, device=device)
    
    timings = {}
    is_cuda = torch.cuda.is_available()
    
    # Profila ogni layer separatamente
    for name, module in model.named_children():
        if name in ['patch_embed', 'layers', 'head']:
            with profiler.profile(
                activities=[ProfilerActivity.CUDA] if is_cuda else [ProfilerActivity.CPU],
                record_shapes=False,
            ) as prof:
                with profiler.record_function(f"layer_{name}"):
                    if name == 'layers':
                        temp_x = model.patch_embed(x)
                        for i, layer in enumerate(module):
                            with profiler.record_function(f"layer_{i}"):
                                temp_x = layer(temp_x)
                    elif name == 'patch_embed':
                        _ = module(x)
                    elif name == 'head':
                        # Simula passaggio attraverso i layer precedenti
                        temp_x = model.patch_embed(x)
                        for layer in model.layers:
                            temp_x = layer(temp_x)
                        _ = module(temp_x)
            
            # ✅ FIX: Usa self_cuda_time o self_cpu_time a seconda del device
            if is_cuda:
                # Somma tutti i tempi CUDA self-time (tempo effettivo speso nel kernel)
                #avg_time = sum([e.self_cuda_time_total for e in prof.key_averages()]) / 1000  # ms
                avg_time = sum([e.self_device_time_total for e in prof.key_averages()]) / 1000  # ms
            else:
                avg_time = sum([e.self_cpu_time_total for e in prof.key_averages()]) / 1000  # ms
            
            timings[name] = avg_time
    
    print("\n" + "="*80)
    print("TIMING PER LAYER")
    print("="*80)
    for name, time in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:30s}: {time:10.2f} ms")
    
    return timings


def get_layer_statistics(prof):
    """
    Estrae statistiche dettagliate dalle operazioni profilate
    """
    stats = {}
    
    for event in prof.key_averages():
        # Raggruppa per tipo di operazione
        op_name = event.key
        
        if op_name not in stats:
            stats[op_name] = {
                'count': 0,
                'cpu_time': 0,
                'cuda_time': 0,
                'memory': 0
            }
        
        stats[op_name]['count'] += event.count
        stats[op_name]['cpu_time'] += event.cpu_time_total
        
        # ✅ FIX: Controlla se l'attributo esiste prima di accedere
        if hasattr(event, 'cuda_time_total') and torch.cuda.is_available():
            stats[op_name]['cuda_time'] += event.cuda_time_total
        
        if hasattr(event, 'self_cuda_memory_usage'):
            stats[op_name]['memory'] += event.self_cuda_memory_usage
    
    return stats


def print_layer_statistics(stats, top_k=20):
    """
    Stampa statistiche aggregate per tipo di operazione
    """
    print("\n" + "="*80)
    print("STATISTICHE AGGREGATE PER TIPO DI OPERAZIONE")
    print("="*80)
    
    # Ordina per tempo CUDA o CPU
    is_cuda = torch.cuda.is_available()
    sorted_stats = sorted(
        stats.items(),
        key=lambda x: x[1]['cuda_time'] if is_cuda else x[1]['cpu_time'],
        reverse=True
    )
    
    print(f"{'Operation':<50} {'Count':>8} {'Time (ms)':>12} {'Memory (MB)':>12}")
    print("-" * 80)
    
    for op_name, data in sorted_stats[:top_k]:
        time_ms = (data['cuda_time'] if is_cuda else data['cpu_time']) / 1000
        memory_mb = data['memory'] / (1024 * 1024)
        print(f"{op_name:<50} {data['count']:>8} {time_ms:>12.2f} {memory_mb:>12.2f}")


# ============================================================================
# ESEMPIO DI UTILIZZO
# ============================================================================

if __name__ == "__main__":
    print("Inizializzazione modello...")
    
    model = RSCDMamba(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=6,
        depths=[2, 4, 8, 4],
        dims=64
    )
    
    print("✓ Modello RSCDMamba caricato con successo!")
    
    # Opzione 1: Profiling completo
    print("\n" + "="*80)
    print("PROFILING COMPLETO")
    print("="*80)
    prof = profile_model(model, input_shape=(2, 3, 224, 224), num_iterations=5)
    print_profile_results(prof, top_k=30)
    export_chrome_trace(prof, "spatial_mamba_trace.json")
    
    # Estrai e stampa statistiche aggregate
    stats = get_layer_statistics(prof)
    print_layer_statistics(stats, top_k=30)
    
    # Opzione 2: Analisi per layer
    print("\n" + "="*80)
    print("ANALISI PER LAYER")
    print("="*80)
    timings = analyze_specific_layers(model, input_shape=(2, 3, 224, 224))
    
    print("\n" + "="*80)
    print("COSA CERCARE NEI RISULTATI:")
    print("="*80)
    print("1. Operazioni con tempo molto alto (ms)")
    print("2. Operazioni chiamate molte volte (guarda 'Count')")
    print("3. 'aten::pad' con mode='replicate' (potenzialmente lento)")
    print("4. 'aten::conv2d' chiamate multiple")
    print("5. 'Memory (MB)' per memory bottleneck")
    print("6. Apri 'spatial_mamba_trace.json' in chrome://tracing per visualizzazione")
    print("="*80)