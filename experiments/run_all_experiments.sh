#!/bin/bash
# FastVLM Experiments Runner (TextVQA Benchmark)
# Runs all ablation experiments sequentially and combines results

echo "🚀 FastVLM Experiments Suite (TextVQA)"
echo "======================================"

# Set default paths (modify these as needed)
STAGE2_PATH="${1:-../checkpoints/llava-fastvithd_0.5b_stage2}"
STAGE3_PATH="${2:-../checkpoints/llava-fastvithd_0.5b_stage3}"
DEVICE="${3:-cuda}"
NUM_SAMPLES="${4:-20}"

echo "📂 Configuration:"
echo "  Stage 2 Model: $STAGE2_PATH"
echo "  Stage 3 Model: $STAGE3_PATH"
echo "  Device: $DEVICE"
echo "  Samples per Exp: $NUM_SAMPLES"
echo ""

# Check if required paths exist
if [ ! -d "$STAGE2_PATH" ]; then
    echo "❌ Error: Stage 2 model path not found: $STAGE2_PATH"
    exit 1
fi

if [ ! -d "$STAGE3_PATH" ]; then
    echo "❌ Error: Stage 3 model path not found: $STAGE3_PATH"
    exit 1
fi

# Create results directory
mkdir -p results
mkdir -p results/plots

echo "🧪 Starting experiments..."
echo ""

# Experiment 1: Resolution Scaling
echo "1️⃣ Running Resolution Scaling Experiment..."
python exp_resolution_scaling.py \
    --model-path "$STAGE3_PATH" \
    --device "$DEVICE" \
    --num-samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✅ Resolution scaling completed"
else
    echo "❌ Resolution scaling failed"
fi
echo ""

# Experiment 2: Token Budget Ablation
echo "2️⃣ Running Token Budget Ablation Experiment..."
python exp_token_budget_ablation.py \
    --model-path "$STAGE3_PATH" \
    --device "$DEVICE" \
    --num-samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✅ Token budget ablation completed"
else
    echo "❌ Token budget ablation failed"
fi
echo ""

# Experiment 3: Stage Comparison
echo "3️⃣ Running Stage Comparison Experiment..."
python exp_stage_comparison.py \
    --stage2-path "$STAGE2_PATH" \
    --stage3-path "$STAGE3_PATH" \
    --device "$DEVICE" \
    --num-samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✅ Stage comparison completed"
else
    echo "❌ Stage comparison failed"
fi
echo ""

# Experiment 4: Prompt Length Effect
echo "4️⃣ Running Prompt Length Effect Experiment..."
python exp_prompt_length_effect.py \
    --model-path "$STAGE3_PATH" \
    --device "$DEVICE" \
    --num-samples "$NUM_SAMPLES"

if [ $? -eq 0 ]; then
    echo "✅ Prompt length effect completed"
else
    echo "❌ Prompt length effect failed"
fi
echo ""

echo "📊 Combining results..."

# Python script to combine CSV files
python << EOF
import pandas as pd
import os
from glob import glob

# Find all CSV files in results directory
csv_files = glob('results/*.csv')
print(f"Found CSV files: {csv_files}")

combined_data = {}

for csv_file in csv_files:
    if os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file)
            experiment_name = os.path.basename(csv_file).replace('.csv', '')
            df['experiment'] = experiment_name
            combined_data[experiment_name] = df
            print(f"✅ Loaded {csv_file}: {len(df)} rows")
        except Exception as e:
            print(f"❌ Error loading {csv_file}: {e}")

if combined_data:
    # Save individual experiment summaries
    summary_path = 'results/experiment_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("FastVLM Experiments Summary (TextVQA)\\n")
        f.write("=" * 50 + "\\n\\n")

        for exp_name, df in combined_data.items():
            f.write(f"{exp_name.upper()}:\\n")
            f.write(f"  Rows: {len(df)}\\n")
            f.write(f"  Columns: {list(df.columns)}\\n")

            # Add experiment-specific summary
            if 'latency' in df.columns:
                f.write(f"  Avg Latency: {df['latency'].mean():.1f}ms\\n")
            if 'ttft' in df.columns:
                f.write(f"  Avg TTFT: {df['ttft'].mean():.1f}ms\\n")
            if 'accuracy' in df.columns:
                f.write(f"  Avg Accuracy: {df['accuracy'].mean():.2%}\\n")
            if 'vram' in df.columns:
                f.write(f"  Avg VRAM: {df['vram'].mean():.2f}GB\\n")

            f.write("\\n")

    print(f"📋 Summary saved: {summary_path}")

else:
    print("❌ No CSV files found to combine")
EOF
