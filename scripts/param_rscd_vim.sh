#!/bin/bash
# filepath: c:\Users\mirko\Downloads\Vim-main\Vim-main\vim\scripts\advanced_train.sh

# Valori di default
ENV_NAME="vim_test"
DATA_PATH="../../../RSCD dataset-1million/"
BATCH_SIZE=32
# TODO: change default model
MODEL="vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2"
OUTPUT_DIR=""
NUM_WORKERS=""
WEIGHT_DECAY=0.1
DROP_PATH=0.0
ENABLE_AMP=false
DRY_RUN=false

# Funzione di help
show_help() {
    cat << EOF
Vim Training Script with GPU Auto-Detection

Usage: $0 [OPTIONS]

OPTIONS:
    -e, --env ENV_NAME          Conda environment name (default: vim_test)
    -d, --data DATA_PATH        Path to dataset (default: ../../../RSCD dataset-1million/)
    -b, --batch BATCH_SIZE      Batch size per GPU (default: 32)
    -m, --model MODEL_NAME      Model architecture (default: vim_tiny_...)
    -o, --output OUTPUT_DIR     Output directory (default: auto-generated)
    -w, --workers NUM_WORKERS   Number of data workers (default: auto-calculated)
    -r, --weight-decay RATE     Weight decay rate (default: 0.1)
    -p, --drop-path RATE        Drop path rate (default: 0.0)
    -a, --amp                   Enable Automatic Mixed Precision
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

# Funzione per rilevare GPU
detect_gpus() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "ERROR: nvidia-smi not found! CUDA not available?"
        exit 1
    fi
    
    echo "=== GPU Detection ==="
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader,nounits | \
    while IFS=, read -r index name memory; do
        memory_gb=$((memory / 1024))
        echo "GPU $index: $name (${memory_gb}GB)"
    done
    
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ "$GPU_COUNT" -eq 0 ]; then
        echo "ERROR: No GPUs found!"
        exit 1
    fi
    
    GPU_LIST=$(seq -s, 0 $((GPU_COUNT-1)))
    echo "Total GPUs: $GPU_COUNT"
    echo "GPU list: $GPU_LIST"
}

# Calcola parametri automatici
calculate_params() {
    # Auto-calcola workers se non specificato
    if [ -z "$NUM_WORKERS" ]; then
        NUM_WORKERS=$((8 * GPU_COUNT))
    fi
    
    # Auto-genera output dir se non specificata
    if [ -z "$OUTPUT_DIR" ]; then
        MODEL_SHORT=$(echo "$MODEL" | cut -d'_' -f1-2)  # vim_tiny
        OUTPUT_DIR="./output/${MODEL_SHORT}_$(date +%Y%m%d_%H%M%S)_${GPU_COUNT}gpu"
    fi
    
    # Calcola batch size totale
    TOTAL_BATCH_SIZE=$((BATCH_SIZE * GPU_COUNT))
    
    # Determina flag AMP
    AMP_FLAG=""
    if [ "$ENABLE_AMP" = false ]; then
        AMP_FLAG="--no_amp"
    fi
}

# Mostra configurazione
show_config() {
    echo "=== Training Configuration ==="
    echo "Environment:       $ENV_NAME"
    echo "Data path:         $DATA_PATH"
    echo "Model:             $MODEL"
    echo "Batch size/GPU:    $BATCH_SIZE"
    echo "Total batch size:  $TOTAL_BATCH_SIZE"
    echo "GPUs:              $GPU_COUNT ($GPU_LIST)"
    echo "Workers:           $NUM_WORKERS"
    echo "Weight decay:      $WEIGHT_DECAY"
    echo "Drop path:         $DROP_PATH"
    echo "Mixed precision:   $ENABLE_AMP"
    echo "Output dir:        $OUTPUT_DIR"
    echo "=========================="
}

# Validazioni
validate_inputs() {
    # Verifica ambiente conda
    if ! conda info --envs | grep -q "^$ENV_NAME "; then
        echo "ERROR: Conda environment '$ENV_NAME' not found!"
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

# MAIN EXECUTION
echo "=== ViM Advanced Training Script ==="

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

# Vai nella directory vim
cd ../

# Validazioni
validate_inputs

# Attiva ambiente
echo "=== Activating Environment ==="
conda activate "$ENV_NAME"

# Crea output directory
mkdir -p "$OUTPUT_DIR"

# Conferma prima di iniziare
echo "=== Ready to Start Training ==="
echo "Press Enter to continue or Ctrl+C to cancel..."
read -r

echo "=== Starting Training ==="

# Launch training
CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
    --nproc_per_node="$GPU_COUNT" \
    main.py \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --drop-path "$DROP_PATH" \
    --weight-decay "$WEIGHT_DECAY" \
    --num_workers "$NUM_WORKERS" \
    --data-path "$DATA_PATH" \
    --output_dir "$OUTPUT_DIR" \
    $AMP_FLAG

echo "=== Training Completed ==="
echo "Results saved in: $OUTPUT_DIR"

# Mostra summary finale
echo "=== Training Summary ==="
echo "Total time: $(date)"
echo "Model: $MODEL"
echo "GPUs used: $GPU_COUNT"
echo "Final batch size: $TOTAL_BATCH_SIZE"
echo "Output: $OUTPUT_DIR"