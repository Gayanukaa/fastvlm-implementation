#!/bin/bash
# FastVLM Experiments Runner
# Runs all ablation experiments sequentially and combines results

echo "🚀 FastVLM Experiments Suite"
echo "==========================="

# Set default paths (modify these as needed)
STAGE2_PATH="${1:-../checkpoints/llava-fastvithd_0.5b_stage2}"
STAGE3_PATH="${2:-../checkpoints/llava-fastvithd_0.5b_stage3}"
IMAGE_FOLDER="${3:-../images}"
DEVICE="${4:-cuda}"

echo "📂 Configuration:"
echo "  Stage 2 Model: $STAGE2_PATH"
echo "  Stage 3 Model: $STAGE3_PATH"
echo "  Images Folder: $IMAGE_FOLDER"
echo "  Device: $DEVICE"
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

if [ ! -d "$IMAGE_FOLDER" ]; then
    echo "❌ Error: Images folder not found: $IMAGE_FOLDER"
    exit 1
fi

# Create results directory
mkdir -p results
mkdir -p results/plots

# Change to experiments directory
cd experiments

echo "🧪 Starting experiments..."
echo ""

# Experiment 1: Resolution Scaling
echo "1️⃣ Running Resolution Scaling Experiment..."
python exp_resolution_scaling.py \
    --model-path "$STAGE3_PATH" \
    --image-folder "$IMAGE_FOLDER" \
    --device "$DEVICE" \
    --prompt "Describe this image in detail."

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
    --image-folder "$IMAGE_FOLDER" \
    --device "$DEVICE" \
    --prompt "Describe this image in detail."

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
    --image-folder "$IMAGE_FOLDER" \
    --device "$DEVICE"

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
    --image-folder "$IMAGE_FOLDER" \
    --device "$DEVICE"

if [ $? -eq 0 ]; then
    echo "✅ Prompt length effect completed"
else
    echo "❌ Prompt length effect failed"
fi
echo ""

# Go back to root directory
cd ..

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
        f.write("FastVLM Experiments Summary\\n")
        f.write("=" * 50 + "\\n\\n")

        for exp_name, df in combined_data.items():
            f.write(f"{exp_name.upper()}:\\n")
            f.write(f"  Rows: {len(df)}\\n")
            f.write(f"  Columns: {list(df.columns)}\\n")

            # Add experiment-specific summary
            if 'total_latency_ms' in df.columns:
                f.write(f"  Avg Latency: {df['total_latency_ms'].mean():.1f}ms\\n")
            if 'ttft_ms' in df.columns:
                f.write(f"  Avg TTFT: {df['ttft_ms'].mean():.1f}ms\\n")
            if 'vram_used_gb' in df.columns:
                f.write(f"  Avg VRAM: {df['vram_used_gb'].mean():.2f}GB\\n")

            f.write("\\n")

    print(f"📋 Summary saved: {summary_path}")

    # Try to combine all data (if columns are compatible)
    try:
        # This might not work if column schemas are very different
        all_data = pd.concat(combined_data.values(), ignore_index=True, sort=False)
        combined_path = 'results/combined.csv'
        all_data.to_csv(combined_path, index=False)
        print(f"📊 Combined data saved: {combined_path}")
    except Exception as e:
        print(f"⚠️  Could not combine all data: {e}")
        print("Individual CSV files are still available.")

else:
    print("❌ No CSV files found to combine")
EOF

# Display results summary
echo ""
echo "📈 Results Summary:"
echo "=================="

if [ -f "results/experiment_summary.txt" ]; then
    cat results/experiment_summary.txt
else
    echo "Summary file not found."
fi

echo ""
echo "📁 Generated Files:"
echo "  📊 CSV Results: results/*.csv"
echo "  📈 Plots: results/plots/*.png"
if [ -f "results/combined.csv" ]; then
    echo "  📋 Combined: results/combined.csv"
fi
echo "  📝 Summary: results/experiment_summary.txt"

echo ""
echo "🎉 All experiments completed!"
echo ""
echo "📖 Usage:"
echo "  ./run_all_experiments.sh [stage2_path] [stage3_path] [image_folder] [device]"
echo ""
echo "🚀 To view plots:"
echo "  xdg-open results/plots/"

# Try to open plots folder (if in GUI environment)
if command -v xdg-open &> /dev/null; then
    echo "🖼️  Opening plots folder..."
    xdg-open results/plots/ 2>/dev/null || true
fi