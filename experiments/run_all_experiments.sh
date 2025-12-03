#!/bin/bash
# FastVLM Experiments Runner
# Runs all replication experiments for Tables 3, 4, 5 and Figure 5

set -e  # Exit on error

echo "🚀 FastVLM Experiments Suite"
echo "======================================"

# Set default paths (modify these or pass as arguments)
MODEL_PATH="${1:-../checkpoints/llava-fastvithd_0.5b_stage3}"
DEVICE="${2:-cuda}"
NUM_SAMPLES="${3:-}"  # Leave empty for full dataset, or set a number for quick testing

echo "📂 Configuration:"
echo "  Model Path: $MODEL_PATH"
echo "  Device: $DEVICE"
if [ -z "$NUM_SAMPLES" ]; then
    echo "  Samples: FULL DATASET"
    SAMPLES_ARG=""
else
    echo "  Samples per Experiment: $NUM_SAMPLES"
    SAMPLES_ARG="--num-samples $NUM_SAMPLES"
fi
echo ""

# Check if required model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ Error: Model path not found: $MODEL_PATH"
    exit 1
fi

# Create results directory
mkdir -p results
mkdir -p results/plots

echo "🧪 Starting experiments..."
echo ""

# ============================================================
# Experiment 1: Table 3 - Encoder Comparison (no dataset needed)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Running Table 3: Encoder Comparison Benchmark"
echo "    (ViT-L/14, ConvNeXt-L, FastViT-HD latency comparison)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python exp_table3.py

if [ $? -eq 0 ]; then
    echo "✅ Table 3 completed"
else
    echo "❌ Table 3 failed"
fi
echo ""

# ============================================================
# Experiment 2: Table 4 - Visual Token Efficiency
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  Running Table 4: Visual Token Efficiency"
echo "    (FastViT-HD & ConvNeXt-L at various resolutions)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python exp_table4.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    $SAMPLES_ARG

if [ $? -eq 0 ]; then
    echo "✅ Table 4 completed"
else
    echo "❌ Table 4 failed"
fi
echo ""

# ============================================================
# Experiment 3: Table 5 - FastViT-HD Visual Token Scaling
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  Running Table 5: FastViT-HD Visual Token Scaling"
echo "    (256, 512, 768, 1024px resolutions on TextVQA)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python exp_table5.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    $SAMPLES_ARG

if [ $? -eq 0 ]; then
    echo "✅ Table 5 completed"
else
    echo "❌ Table 5 failed"
fi
echo ""

# ============================================================
# Experiment 4: Figure 5 - Vision vs LLM Prefilling Latency
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️⃣  Running Figure 5: Vision Latency vs LLM Prefilling"
echo "    (Resolution scaling: 256, 512, 768, 1024, 1536px)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python exp_figure5.py \
    --model-path "$MODEL_PATH" \
    --device "$DEVICE" \
    $SAMPLES_ARG

if [ $? -eq 0 ]; then
    echo "✅ Figure 5 completed"
else
    echo "❌ Figure 5 failed"
fi
echo ""

# ============================================================
# Summary
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 Combining Results..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Generate summary of all results
python << 'EOF'
import os
from glob import glob

print("\n📁 Generated Output Files:")
print("-" * 50)

# List CSV files
csv_files = sorted(glob('results/*.csv'))
if csv_files:
    print("\n📄 CSV Results:")
    for f in csv_files:
        size = os.path.getsize(f)
        print(f"   {f} ({size} bytes)")

# List PNG files
png_files = sorted(glob('results/plots/*.png'))
if png_files:
    print("\n🖼️  Plots & Figures:")
    for f in png_files:
        size = os.path.getsize(f)
        print(f"   {f} ({size} bytes)")

# List table images in results root
table_imgs = sorted(glob('results/*.png'))
if table_imgs:
    print("\n📊 Table Images:")
    for f in table_imgs:
        size = os.path.getsize(f)
        print(f"   {f} ({size} bytes)")

print("\n" + "=" * 50)
print("✅ All experiments completed!")
print("=" * 50)
print("\n📋 See experiments/limitations.md for replication notes.")
EOF

echo ""
echo "🎉 FastVLM Experiment Suite Complete!"
echo ""
echo "Usage:"
echo "  Quick Test (10 samples):  ./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage3 cuda 10"
echo "  Full Dataset:             ./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage3 cuda"
