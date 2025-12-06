#!/bin/bash

echo "========================================="
echo "FastVLM Video Finetuning Setup Verification"
echo "========================================="

# Check dataset files
echo -e "\n[1] Checking dataset files..."
if [ -f "dataset/fastvlm_train.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/fastvlm_train.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/fastvlm_train.json found ($num_samples samples)"
else
    echo "✗ dataset/fastvlm_train.json NOT found"
    echo "  → Run: python utils/qved_from_fine_labels.py"
fi

if [ -f "dataset/fastvlm_val.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/fastvlm_val.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/fastvlm_val.json found ($num_samples samples)"
else
    echo "✗ dataset/fastvlm_val.json NOT found"
fi

if [ -f "dataset/fastvlm_test.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/fastvlm_test.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/fastvlm_test.json found ($num_samples samples)"
else
    echo "✗ dataset/fastvlm_test.json NOT found"
fi

if [ -f "dataset/manifest.json" ]; then
    echo "✓ dataset/manifest.json found"
else
    echo "✗ dataset/manifest.json NOT found"
fi

if [ -f "dataset/ground_truth.json" ]; then
    num_videos=$(python -c "import json; print(len(json.load(open('dataset/ground_truth.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/ground_truth.json found ($num_videos videos)"
else
    echo "✗ dataset/ground_truth.json NOT found"
fi

# Check video files
echo -e "\n[2] Checking video files..."

exercise_dirs=(
  "alternating_single_leg_glutes_bridge"
  "cat-cow_pose"
  "elbow_plank"
  "glute_hamstring_walkout"
  "glutes_bridge"
  "heel_lift"
  "high_plank"
  "lunges_leg_out_in_front"
  "opposite_arm_and_leg_lifts_on_knees"
  "pushups"
  "side_plank"
  "squats"
  "toe_touch"
  "tricep_stretch"
)

total_videos=0
for dir in "${exercise_dirs[@]}"; do
    if [ -d "dataset/$dir" ]; then
        num_videos=$(ls -1 dataset/$dir/*.mp4 2>/dev/null | wc -l)
        total_videos=$((total_videos + num_videos))
        echo "✓ dataset/$dir/ ($num_videos videos)"
    else
        echo "✗ dataset/$dir/ NOT found"
    fi
done
echo "  Total: $total_videos videos"

# Check if paths in dataset split files match actual files
echo -e "\n[3] Verifying video paths in dataset splits..."

# Check training set (FastVLM uses 'image' key for video paths)
if [ -f "dataset/fastvlm_train.json" ]; then
    result=$(python -c "
import json
import os
with open('dataset/fastvlm_train.json') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item['image'])))
    print(f'{total},{missing}')
" 2>/dev/null)

    total_count=$(echo $result | cut -d',' -f1)
    missing_count=$(echo $result | cut -d',' -f2)

    if [ "$missing_count" -eq 0 ]; then
        echo "✓ Training set: All $total_count video paths are valid"
    else
        echo "✗ Training set: $missing_count out of $total_count videos are missing"
    fi
fi

# Check validation set
if [ -f "dataset/fastvlm_val.json" ]; then
    result=$(python -c "
import json
import os
with open('dataset/fastvlm_val.json') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item['image'])))
    print(f'{total},{missing}')
" 2>/dev/null)

    total_count=$(echo $result | cut -d',' -f1)
    missing_count=$(echo $result | cut -d',' -f2)

    if [ "$missing_count" -eq 0 ]; then
        echo "✓ Validation set: All $total_count video paths are valid"
    else
        echo "✗ Validation set: $missing_count out of $total_count videos are missing"
    fi
fi

# Check test set
if [ -f "dataset/fastvlm_test.json" ]; then
    result=$(python -c "
import json
import os
with open('dataset/fastvlm_test.json') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item['image'])))
    print(f'{total},{missing}')
" 2>/dev/null)

    total_count=$(echo $result | cut -d',' -f1)
    missing_count=$(echo $result | cut -d',' -f2)

    if [ "$missing_count" -eq 0 ]; then
        echo "✓ Test set: All $total_count video paths are valid"
    else
        echo "✗ Test set: $missing_count out of $total_count videos are missing"
    fi
fi

# Check model checkpoint
echo -e "\n[4] Checking model checkpoint..."
if [ -d "checkpoints/llava-fastvithd_0.5b_stage3" ]; then
    if [ -f "checkpoints/llava-fastvithd_0.5b_stage3/model.safetensors" ]; then
        model_size=$(du -sh checkpoints/llava-fastvithd_0.5b_stage3/model.safetensors 2>/dev/null | cut -f1)
        echo "✓ checkpoints/llava-fastvithd_0.5b_stage3/ found (model: $model_size)"
    else
        echo "✗ Model weights not found in checkpoints/llava-fastvithd_0.5b_stage3/"
    fi

    if [ -f "checkpoints/llava-fastvithd_0.5b_stage3/config.json" ]; then
        echo "✓ config.json found"
    else
        echo "✗ config.json NOT found"
    fi

    if [ -f "checkpoints/llava-fastvithd_0.5b_stage3/tokenizer_config.json" ]; then
        echo "✓ tokenizer_config.json found"
    else
        echo "✗ tokenizer_config.json NOT found"
    fi
else
    echo "✗ checkpoints/llava-fastvithd_0.5b_stage3/ NOT found"
    echo "  → Download the pretrained model first"
fi

# Check required scripts and utilities
echo -e "\n[5] Checking required scripts..."
if [ -f "scripts/train_video_exercise.sh" ]; then
    echo "✓ scripts/train_video_exercise.sh found"
else
    echo "✗ scripts/train_video_exercise.sh NOT found"
fi

if [ -f "llava/train/train_mem.py" ]; then
    echo "✓ llava/train/train_mem.py found"
else
    echo "✗ llava/train/train_mem.py NOT found"
fi

if [ -f "utils/inference_single.py" ]; then
    echo "✓ utils/inference_single.py found"
else
    echo "✗ utils/inference_single.py NOT found"
fi

if [ -f "utils/plot_training_stats.py" ]; then
    echo "✓ utils/plot_training_stats.py found"
else
    echo "✗ utils/plot_training_stats.py NOT found"
fi

if [ -f "utils/qved_from_fine_labels.py" ]; then
    echo "✓ utils/qved_from_fine_labels.py found"
else
    echo "✗ utils/qved_from_fine_labels.py NOT found"
fi

# Check Python environment
echo -e "\n[6] Checking Python environment..."
if command -v conda &> /dev/null; then
    if conda env list | grep -q "fastvlm"; then
        echo "✓ Conda environment 'fastvlm' exists"
    else
        echo "⚠ Conda environment 'fastvlm' NOT found"
        echo "  Available environments:"
        conda env list | grep -v "^#" | head -5
    fi
fi

python_version=$(python --version 2>&1)
echo "  Python: $python_version"

# Check key dependencies
echo -e "\n[7] Checking key dependencies..."
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "✗ PyTorch NOT installed"
python -c "import transformers; print(f'✓ Transformers {transformers.__version__}')" 2>/dev/null || echo "✗ Transformers NOT installed"
python -c "import decord; print('✓ Decord (video loading)')" 2>/dev/null || echo "✗ Decord NOT installed (required for video)"
python -c "import PIL; print('✓ Pillow (image processing)')" 2>/dev/null || echo "✗ Pillow NOT installed"
python -c "import matplotlib; print('✓ Matplotlib (plotting)')" 2>/dev/null || echo "✗ Matplotlib NOT installed"

# Check GPU availability
echo -e "\n[8] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    echo "✓ $gpu_count GPU(s) detected"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader 2>/dev/null

    # Check CUDA availability in PyTorch
    python -c "
import torch
if torch.cuda.is_available():
    print(f'✓ CUDA available: {torch.cuda.get_device_name(0)}')
    print(f'  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('✗ CUDA not available in PyTorch')
" 2>/dev/null
else
    echo "✗ nvidia-smi not found - GPU may not be available"
fi

echo -e "\n========================================="
echo "Setup verification complete!"
echo "========================================="
