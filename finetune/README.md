# FastVLM Video Fine-tuning

This directory contains video fine-tuning scripts and usage demonstrations for FastVLM.

## Video Fine-tuning Example

**File:** `video_finetuning_example.py`

This script demonstrates how to set up and use FastVLM's video fine-tuning capabilities.

### Usage

```bash
python finetune/video_finetuning_example.py
```

This will create:
- `example_video_dataset.json` - Sample dataset in the correct format
- `train_video_example.sh` - Training script with recommended parameters

### What it demonstrates

1. **Dataset format** for mixed image/video training
2. **Training configuration** for video fine-tuning
3. **Parameter recommendations** for different use cases
4. **Memory optimization** strategies

### Quick Start

```bash
# 1. Run the example generator
cd finetune
python video_finetuning_example.py

# 2. Edit the generated training script with your paths
vim train_video_example.sh

# 3. Install decord for video processing
pip install decord

# 4. Run training
bash train_video_example.sh
```

## Key Features Demonstrated

- **Automatic video detection** - No special flags needed, just use `.mp4` (or other video) extensions
- **Frame sampling** - Control via `--num_video_frames` parameter
- **Token budget** - FastVLM's efficiency allows 8 frames in ~800 tokens vs 4608 for CLIP
- **Mixed datasets** - Seamlessly combine images and videos in the same training run

## Documentation

For comprehensive documentation, see:
- `finetune/VIDEO_FINETUNING.md` - Complete guide to video fine-tuning

## Dataset Format

```json
[
  {
    "image": "path/to/video.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nYour question here"
      },
      {
        "from": "gpt",
        "value": "Model response here"
      }
    ]
  }
]
```

The `<image>` token will automatically be expanded to N tokens (one per frame) during preprocessing.
