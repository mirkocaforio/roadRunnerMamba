#!/bin/bash
# filepath: c:\Users\mirko\Downloads\Vim-main\Vim-main\vim\scripts\advanced_train.sh

# Valori di default
ENV_NAME="rscd_mamba"
DATA_PATH="../../../datahome/RSCD_dataset_1million_reorganized/"
BATCH_SIZE=64
MODEL="rscd_mamba_v1"
#OUTPUT_DIR=""
OUTPUT_DIR="../../../datahome/rscd_mamba_data/output/rscd_mamba_20260615_000356_1gpu/"
NUM_WORKERS=""
WEIGHT_DECAY=0.01
DROP_PATH=0.3
ENABLE_AMP=false
DRY_RUN=false
MULTI_GPU=false

# Funzione di help
show_help() {
    cat << EOF
RSCD_Mamba Training Script with GPU Auto-Detection

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENV_NAME          Conda environment name (default: rscd_mamba)
    -d, --data DATA_PATH        Path to dataset (default: ../../../RSCD_dataset_1million/)
    -b, --batch BATCH_SIZE      Batch size per GPU (default: 32)
    -m, --model MODEL_NAME      Model architecture (default: rscd_mamba...)
    -o, --output OUTPUT_DIR     Output directory (default: auto-generated)
    -w, --workers NUM_WORKERS   Number of data workers (default: auto-calculated)
    -r, --weight-decay RATE     Weight decay rate (default: 0.1)
    -p, --drop-path RATE        Drop path rate (default: 0.0)
    -a, --amp                   Enable Automatic Mixed Precision
    -g, --multi-gpu             Enable Multi-GPU training
    -n, --dry-run              Show configuration without training
    -h, --help                 Show this help message

EXAMPLES:
    $0                                          # Use all defaults
    $0 -e my_env -b 64                         # Custom env and batch size
    $0 -d /path/to/imagenet -a                 # Custom data path with AMP
    $0 -b 16 -w 4 --dry-run                   # Test configuration
    $0 --model vim_small_patch16_224 -o ./runs # Custom model and output

GPU Detection:
    - Automatically detects available GPUs
    - Scales batch size and workers accordingly
    - Shows GPU memory information

    Multi-GPU mode (--multi-gpu):
        - Uses all available GPUs
        - Distributed training with torchrun
        - Scales batch size and workers automatically
EOF
}

# Parse degli argomenti
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            ENV_NAME="$2"
            shift 2
            ;;
        -d|--data)
            DATA_PATH="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_SIZE="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        -r|--weight-decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        -p|--drop-path)
            DROP_PATH="$2"
            shift 2
            ;;
        -a|--amp)
            ENABLE_AMP=true
            shift
            ;;
        -g|--multi-gpu)
            MULTI_GPU=true
            shift
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Funzione per rilevare GPU e selezionare quella con più VRAM libera (o tutte)
detect_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found! CUDA not available?"
        exit 1
    fi
    
    echo "=== GPU Detection ==="
    
    # Mostra info su tutte le GPU
    nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader,nounits | \
    while IFS=, read -r index name total free; do
        total_gb=$((total / 1024))
        free_gb=$((free / 1024))
        echo "GPU $index: $name (${total_gb}GB totale, ${free_gb}GB libera)"
    done
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "ERROR: No GPUs found!"
        exit 1
    fi
    
    echo ""
    echo "Total GPUs available: $GPU_COUNT"
    
    if [ "$MULTI_GPU" = true ]; then
        # Modalità Multi-GPU: usa tutte le GPU
        GPU_LIST=$(seq -s, 0 $((GPU_COUNT-1)))
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        echo "Mode: Multi-GPU"
        echo "Using all GPUs: $GPU_LIST"
    else
        # Modalità Single-GPU: seleziona la GPU con maggior VRAM libera
        BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
                   sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
        
        FREE_MEM=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $BEST_GPU)
        FREE_MEM_GB=$((FREE_MEM / 1024))
        
        echo "Mode: Single-GPU (auto-selected)"
        echo "Selected GPU: $BEST_GPU (${FREE_MEM_GB}GB free)"
        
        # Imposta la GPU selezionata
        GPU_LIST="$BEST_GPU"
        export CUDA_VISIBLE_DEVICES="$BEST_GPU"
        
        # Per il training usiamo solo 1 GPU
        GPU_COUNT=1
    fi
}

# Calcola parametri automatici
calculate_params() {
    # Auto-calcola workers se non specificato.
    # Con augmentation pesanti su ~1M immagini servono molti worker per non far
    # "affamare" la GPU. Usa fino a 8 worker per GPU, limitati ai core CPU disponibili.
    if [ -z "$NUM_WORKERS" ]; then
        CPU_CORES=$(nproc 2>/dev/null || echo 8)
        NUM_WORKERS=$((8 * GPU_COUNT))
        # Lascia qualche core libero per il processo principale
        MAX_WORKERS=$((CPU_CORES - 2))
        if [ "$MAX_WORKERS" -lt 1 ]; then MAX_WORKERS=1; fi
        if [ "$NUM_WORKERS" -gt "$MAX_WORKERS" ]; then NUM_WORKERS=$MAX_WORKERS; fi
    fi
    
    # Auto-genera output dir se non specificata
    if [ -z "$OUTPUT_DIR" ]; then
        MODEL_SHORT=$(echo "$MODEL" | cut -d'_' -f1-2)  # vim_tiny
        OUTPUT_DIR="../../../datahome/rscd_mamba_data/output/${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)_${GPU_COUNT}gpu"
    fi
    
    # Calcola batch size totale
    TOTAL_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT))
    
    # Determina flag AMP (bfloat16). Il default del training è AMP disabilitato.
    AMP_FLAG=""
    if [ "$ENABLE_AMP" = true ]; then
        AMP_FLAG="--amp"
    fi
}

# Mostra configurazione
show_config() {
    echo "=== Training Configuration ==="
    echo "Environment:       $ENV_NAME"
    echo "Data path:         $DATA_PATH"
#    echo "Model:             $MODEL"
    echo "Batch size/GPU:    $BATCH_SIZE"
    echo "Total batch size:  $TOTAL_BATCH_SIZE"
    echo "GPUs:              $GPU_COUNT ($GPU_LIST)"
    echo "Workers:           $NUM_WORKERS"
#    echo "Weight decay:      $WEIGHT_DECAY"
    echo "Drop path:         $DROP_PATH"
    echo "Mixed precision:   $ENABLE_AMP"
    echo "Output dir:        $OUTPUT_DIR"
    echo "=========================="
}

# Validazioni
validate_inputs() {
    # Verifica ambiente conda
    if ! conda info --envs 2>/dev/null | grep -q "^${ENV_NAME} \|/${ENV_NAME}$"; then
        echo "ERROR: Conda environment '$ENV_NAME' not found."
        echo "Available environments:"
        conda env list
        exit 1
    fi
    
    # Verifica data path
    if [ ! -d "$DATA_PATH" ]; then
        echo "ERROR: Data path '$DATA_PATH' not found!"
        exit 1
    fi
    
    # Verifica batch size
    if ! [[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] || [ "$BATCH_SIZE" -lt 1 ]; then
        echo "ERROR: Batch size must be a positive integer!"
        exit 1
    fi
}

# Adjust data path for training script
adjust_data_path() {
    # Since we'll be in core/processing/, add ../ to the relative path
    # Only if the path is relative (doesn't start with /)
    if [[ "$DATA_PATH" != /* ]]; then
        DATA_PATH_ADJUSTED="../${DATA_PATH}"
        echo "=== Adjusting Data Path ==="
        echo "Original:  $DATA_PATH"
        echo "Adjusted:  $DATA_PATH_ADJUSTED"
    else
        DATA_PATH_ADJUSTED="$DATA_PATH"
        echo "=== Using Absolute Path ==="
        echo "Path:      $DATA_PATH_ADJUSTED"
    fi
}

# MAIN EXECUTION
echo "=== RSCD_Mamba Advanced Training Script ==="

# Rileva GPU
detect_gpus

# Calcola parametri
calculate_params

# Mostra configurazione
show_config

# Dry run - esce senza training
if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN: Configuration shown above. Exiting."
    exit 0
fi

# Vai nella directory core/processing
cd ../core/processing/ || { echo "ERROR: Failed to change directory!"; exit 1; }

# Validazioni
validate_inputs

# Adjust data path after validation but before changing directory
#adjust_data_path

# Attiva ambiente
#echo "=== Activating Conda Environment: $ENV_NAME ==="
#source "$(conda info --base)/etc/profile.d/conda.sh"
#conda activate "$ENV_NAME"

# Imposta PYTHONPATH per includere la root del progetto
#export PYTHONPATH="/home/jupiter_ntc/Desktop/rscd_mamba/rscd_mamba:$PYTHONPATH"
export PYTHONPATH="/home/mcaforio/rscd_mamba:$PYTHONPATH"

# Crea output directory
mkdir -p "$OUTPUT_DIR"

# Conferma prima di iniziare
#echo "=== Ready to Start Training ==="
#echo "Press Enter to continue or Ctrl+C to cancel..."
#read -r

echo "=== Starting Training ==="

# Launch training
#    --model "$MODEL" \
#    --weight-decay "$WEIGHT_DECAY" \

# Aggiunta di un livello a data path per core/processing


CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node="$GPU_COUNT" \
    training_run.py \
    --batch-size "$BATCH_SIZE" \
    --drop-path "$DROP_PATH" \
    --num-workers "$NUM_WORKERS" \
    --data-path "$DATA_PATH" \
    --output-dir "$OUTPUT_DIR" \
    $AMP_FLAG
#    --wandb stuff (disabled):
#    --wandb-enabled \
#    --wandb-project "rscd_mamba" \
#    --wandb-entity "mirko-caforio-university-of-salento" \
#    --wandb-name "${MODEL}_training" \
#    --wandb-tags "rscd_mamba,training,auto" \
#    --wandb-notes "None" \
#    --wandb-upload-checkpoints \
#    --wandb-save-code \
#    --wandb-log-parameters \
#    --wandb-log-freq 1 \
#    --stop-on-error \
#    --stop-on-batch-error \

echo "=== Training Completed ==="
echo "Results saved in: $OUTPUT_DIR"

# Mostra summary finale
echo "=== Training Summary ==="
echo "Total time: $(date)"
echo "Model: $MODEL"
echo "GPUs used: $GPU_COUNT"
echo "Final batch size: $TOTAL_BATCH_SIZE"
echo "Output: $OUTPUT_DIR"