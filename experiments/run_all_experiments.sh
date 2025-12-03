#!/bin/bash
# FastVLM Experiments Runner
# Runs all replication experiments for Tables 3, 4, 5 and Figure 5

set -e  # Exit on error

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/experiment_run_${TIMESTAMP}.log"

# Function to log to both terminal and file
log() {
    echo "$@" | tee -a "$LOG_FILE"
}

log "FastVLM Experiments Suite"
log "======================================"
log "Started: $(date)"
log "Log file: $LOG_FILE"
log ""

# Set default paths (modify these or pass as arguments)
MODEL_PATH="${1:-../checkpoints/llava-fastvithd_0.5b_stage3}"
DEVICE="${2:-cuda}"
NUM_SAMPLES="${3:-}"  # Leave empty for full dataset, or set a number for quick testing

log "Configuration:"
log "  Model Path: $MODEL_PATH"
log "  Device: $DEVICE"
if [ -z "$NUM_SAMPLES" ]; then
    log "  Samples: FULL DATASET"
    SAMPLES_ARG=""
else
    log "  Samples per Experiment: $NUM_SAMPLES"
    SAMPLES_ARG="--num-samples $NUM_SAMPLES"
fi
log ""

# Check if required model path exists
if [ ! -d "$MODEL_PATH" ]; then
    log "Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Create results directory
mkdir -p results
mkdir -p results/plots

log "Starting experiments..."
log ""

# ============================================================
# Experiment 1: Table 3 - Encoder Comparison (no dataset needed)
# ============================================================
log "============================================================"
log "[1/4] Running Table 3: Encoder Comparison Benchmark"
log "      (ViT-L/14, ConvNeXt-L, FastViT-HD latency comparison)"
log "============================================================"
python exp_table3.py 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "Table 3 completed successfully"
else
    log "Table 3 failed"
fi
log ""

# ============================================================
# Experiment 2: Table 4 - Visual Token Efficiency
# ============================================================
log "============================================================"
log "[2/4] Running Table 4: Visual Token Efficiency"
log "      (FastViT-HD & ConvNeXt-L at various resolutions)"
log "============================================================"
python exp_table4.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    $SAMPLES_ARG 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "Table 4 completed successfully"
else
    log "Table 4 failed"
fi
log ""

# ============================================================
# Experiment 3: Table 5 - FastViT-HD Visual Token Scaling
# ============================================================
log "============================================================"
log "[3/4] Running Table 5: FastViT-HD Visual Token Scaling"
log "      (256, 512, 768, 1024px resolutions on TextVQA)"
log "============================================================"
python exp_table5.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    $SAMPLES_ARG 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "Table 5 completed successfully"
else
    log "Table 5 failed"
fi
log ""

# ============================================================
# Experiment 4: Figure 5 - Vision vs LLM Prefilling Latency
# ============================================================
log "============================================================"
log "[4/4] Running Figure 5: Vision Latency vs LLM Prefilling"
log "      (Resolution scaling: 256, 512, 768, 1024, 1536px)"
log "============================================================"
python exp_figure5.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    $SAMPLES_ARG 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    log "Figure 5 completed successfully"
else
    log "Figure 5 failed"
fi
log ""

# ============================================================
# Summary
# ============================================================
log "============================================================"
log "Generating Summary..."
log "============================================================"

# Generate summary of all results
python << 'EOF' 2>&1 | tee -a "$LOG_FILE"
import os
from glob import glob

print("\nGenerated Output Files:")
print("-" * 50)

# List CSV files
csv_files = sorted(glob('results/*.csv'))
if csv_files:
    print("\nCSV Results:")
    for f in csv_files:
        size = os.path.getsize(f)
        print(f"   {f} ({size} bytes)")

# List PNG files
png_files = sorted(glob('results/plots/*.png'))
if png_files:
    print("\nPlots & Figures:")
    for f in png_files:
        size = os.path.getsize(f)
        print(f"   {f} ({size} bytes)")

# List table images in results root
table_imgs = sorted(glob('results/*.png'))
if table_imgs:
    print("\nTable Images:")
    for f in table_imgs:
        size = os.path.getsize(f)
        print(f"   {f} ({size} bytes)")

print("\n" + "=" * 50)
print("All experiments completed!")
print("=" * 50)
print("\nSee experiments/limitations.md for replication notes.")
EOF

log ""
log "Finished: $(date)"
log "FastVLM Experiment Suite Complete!"
log ""
log "Log saved to: $LOG_FILE"
log ""
log "Usage:"
log "  Quick Test (10 samples):  ./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage3 cuda 10"
log "  Full Dataset:             ./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage3 cuda"
