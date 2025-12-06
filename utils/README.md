# FastVLM Video Fine-tuning Utilities

This directory contains utility scripts for preparing datasets, running fine-tuning, performing inference, and generating evaluation reports.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Dataset Preparation](#dataset-preparation)
3. [Fine-tuning](#fine-tuning)
4. [Inference](#inference)
5. [Evaluation](#evaluation)
6. [Utility Scripts Reference](#utility-scripts-reference)

## Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install decord matplotlib wandb openpyxl scikit-learn sentence-transformers

apt-get install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super
```

## Dataset Preparation

### Step 1: Download Dataset

Use `load_dataset.py` to download videos from the HuggingFace dataset repository. This script:

- Downloads N random videos per exercise class (default: 10)
- Preserves the folder structure
- Downloads `fine_grained_labels.json` (complete ground truth)
- Creates `manifest.json` mapping downloaded videos to their exercise classes

```bash
python utils/load_dataset.py
```

To customize the number of videos per class, edit `load_dataset.py`:

```python
MAX_PER_CLASS = 10  # Change this value
RANDOM_SEED = 42    # For reproducibility
```

### Step 2: Filter Ground Truth Labels

Use `filter_ground_truth.py` to create a filtered ground truth file containing only the videos that were actually downloaded:

```bash
python utils/filter_ground_truth.py
```

This script:

- Reads `manifest.json` to identify downloaded videos
- Filters `fine_grained_labels.json` to include only matching entries
- Outputs `ground_truth.json` with filtered labels

### Step 3: Generate Training JSONs

Run the dataset conversion script to create train/val/test splits:

```bash
python utils/qved_from_fine_labels.py
```

This generates three files:

- `dataset/fastvlm_train.json` (60% of data)
- `dataset/fastvlm_val.json` (20% of data)
- `dataset/fastvlm_test.json` (20% of data)

**Output format** (FastVLM training format):

```json
[
  {
    "image": "squats/video_001.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nPlease evaluate the exercise form shown..."
      },
      {
        "from": "gpt",
        "value": "The person maintains good posture..."
      }
    ]
  }
]
```

Note: The `image` key is used even for videos. The training code automatically detects video files by extension.

## Fine-tuning

### Quick Start

Run the complete fine-tuning pipeline:

```bash
bash scripts/finetune_video.sh
```

This script:

1. Runs setup verification
2. Starts training with WandB logging
3. Generates training plots
4. Optionally uploads the model to HuggingFace

### Key Training Parameters

| Parameter                       | Description                 | Recommended Value    |
| ------------------------------- | --------------------------- | -------------------- |
| `--num_video_frames`            | Frames to extract per video | 4-8|
| `--per_device_train_batch_size` | Batch size per GPU          | 1     |
| `--gradient_accumulation_steps` | Effective batch multiplier  | 8                    |
| `--learning_rate`               | Learning rate               | 2e-5                 |
| `--num_train_epochs`            | Training epochs             | 3                    |
| `--evaluation_strategy`         | When to evaluate            | "steps" or "epoch"   |
| `--eval_steps`                  | Evaluation frequency        | 50                   |


## Inference

### Single Video/Image Inference

Run inference on a single file:

```bash
python utils/inference_single.py \
    --model-path models/exercise-video-finetuned \
    --input-file path/to/video.mp4
```

With custom prompt:

```bash
python utils/inference_single.py \
    --model-path models/exercise-video-finetuned \
    --input-file path/to/video.mp4 \
    --prompt "What exercise is being performed and how is the form?"
```

### Batch Test Inference

Run inference on the entire test set:

```bash
bash scripts/run_inference.sh --model_path models/exercise-video-finetuned
```

Or with a HuggingFace model:

```bash
bash scripts/run_inference.sh --hf_repo EdgeVLM-Labs/fastvlm-finetune-20251206
```

With sample limit for quick testing:

```bash
bash scripts/run_inference.sh \
    --model_path models/exercise-video-finetuned \
    --limit 10
```

## Evaluation

### Generate Test Report

After running inference, generate an evaluation report:

```bash
python utils/generate_test_report.py \
    --predictions models/exercise-video-finetuned/test_predictions.json \
    --output models/exercise-video-finetuned/test_evaluation_report.xlsx
```

Skip BERT similarity for faster evaluation:

```bash
python utils/generate_test_report.py \
    --predictions test_predictions.json \
    --output test_report.xlsx \
    --no-bert
```

The report includes:

- Per-sample predictions vs ground truth
- BERT semantic similarity scores
- METEOR scores
- Summary statistics with charts

### Generate Training Plots

Visualize training progress from log files:

```bash
python utils/plot_training_stats.py \
    --log_file models/training_log_20251206.log \
    --output_dir plots/fastvlm_exercise
```
