# Video Fine-tuning Support for FastVLM

This document describes the video fine-tuning capabilities added to FastVLM.

## Overview

FastVLM now supports video fine-tuning using a "frames as images" approach. This leverages FastVLM's compact vision encoder (FastViT-HD Stage 3) which produces significantly fewer tokens per image (~64-144 tokens) compared to standard CLIP-based models (576 tokens). This allows you to process multiple video frames within the same context window without running out of memory.

## Key Features

- **Automatic video detection**: The system automatically detects video files based on file extension
- **Uniform frame sampling**: Extracts N frames uniformly distributed across the video duration
- **Seamless integration**: Video frames are treated as multiple images in a single prompt
- **Configurable frame count**: Control the number of frames extracted per video via `num_video_frames` parameter

## Installation

The video processing functionality requires the `decord` library:

```bash
pip install decord
```

This dependency is automatically included in the updated `pyproject.toml`.

## Supported Video Formats

The following video formats are supported:
- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.flv`
- `.wmv`
- `.webm`
- `.m4v`

## Usage

### Training Data Format

Your training data JSON should reference video files just like image files:

```json
[
  {
    "image": "video_001.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nDescribe what happens in this video."
      },
      {
        "from": "gpt",
        "value": "In this video, a person is walking down a street..."
      }
    ]
  }
]
```

### Training Command

Use the standard training command with the optional `--num_video_frames` parameter:

```bash
python llava/train/train.py \
    --model_name_or_path path/to/base/model \
    --data_path path/to/training_data.json \
    --image_folder path/to/videos/ \
    --vision_tower path/to/vision/tower \
    --num_video_frames 8 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --bf16 True \
    --output_dir ./checkpoints/video-model \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 1 \
    --learning_rate 2e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True
```

### Configuration Parameters

- `--num_video_frames`: Number of frames to extract from each video (default: 8)
  - Recommended values: 4-16 frames depending on available memory
  - More frames = better temporal understanding but more memory usage

## How It Works

### 1. Video Loading (`llava/train/train.py`)

When the data loader encounters a video file:

1. Detects video format using `is_video_file()`
2. Loads N frames uniformly sampled across the video using `load_video_frames()`
3. Processes each frame through the image processor
4. Stacks frames into a tensor: `(T, C, H, W)` where T = num_frames

### 2. Token Expansion (`llava/train/train.py`)

The `preprocess_multimodal()` function expands image tokens for video:

```python
# For an 8-frame video:
# "<image>\nDescribe this video."
# becomes:
# "<image><image><image><image><image><image><image><image>\nDescribe this video."
```

This ensures the tokenizer expects the correct number of visual embeddings.

### 3. Vision Processing (`llava/model/llava_arch.py`)

The `prepare_inputs_labels_for_multimodal()` function handles video tensors:

1. Detects 5D tensors: `(B, T, C, H, W)`
2. Flattens temporal dimension: `(B*T, C, H, W)`
3. Passes all frames through the vision encoder
4. Each frame produces ~100 tokens (FastViT-HD Stage 3)
5. Vision features are mapped to corresponding image tokens in the text

## Memory Considerations

### Token Budget Comparison

**Standard LLaVA (CLIP ViT-L/14):**
- 8 frames × 576 tokens = 4,608 visual tokens
- Nearly exhausts a 4096-token context window

**FastVLM (FastViT-HD Stage 3):**
- 8 frames × ~100 tokens = ~800 visual tokens
- Leaves ~3,200 tokens for text reasoning

### Recommendations

- **4-8 frames**: Good for action recognition, short clips
- **8-16 frames**: Better for understanding longer sequences
- **16+ frames**: Only with very large context windows or shorter text

Adjust `num_video_frames` and `model_max_length` based on your hardware:

```bash
# Conservative (4 frames, 2048 tokens)
--num_video_frames 4 --model_max_length 2048

# Balanced (8 frames, 2048 tokens)
--num_video_frames 8 --model_max_length 2048

# Aggressive (16 frames, 4096 tokens)
--num_video_frames 16 --model_max_length 4096
```

## Implementation Details

### Files Modified

1. **`pyproject.toml`**: Added `decord` dependency
2. **`llava/mm_utils.py`**: 
   - Added `load_video_frames()` function
   - Added `is_video_file()` helper
   - Added numpy import
3. **`llava/train/train.py`**:
   - Added `num_video_frames` to `DataArguments`
   - Updated `LazySupervisedDataset.__getitem__()` to handle videos
   - Modified `preprocess_multimodal()` to expand tokens for videos
4. **`llava/model/llava_arch.py`**:
   - Updated `prepare_inputs_labels_for_multimodal()` to handle 5D tensors
   - Added temporal dimension flattening for video batches

### Code Architecture

```
Training Pipeline:
1. Dataset.__getitem__() → Load video & extract frames
2. preprocess_multimodal() → Expand image tokens (1 → N)
3. DataCollator → Batch videos/images
4. prepare_inputs_labels_for_multimodal() → Flatten 5D tensors
5. Vision Encoder → Process all frames
6. MM Projector → Map vision features to text embeddings
7. Language Model → Generate responses
```

## Limitations

1. **Uniform sampling**: Frames are sampled uniformly, which may miss important events
2. **No temporal modeling**: Each frame is processed independently (no attention across frames)
3. **Static frame count**: All videos use the same number of frames
4. **anyres support**: Video support with anyres image processing is experimental

## Future Enhancements

Potential improvements:
- Adaptive frame sampling based on video content
- Temporal attention mechanisms across frames
- Variable frame counts per video
- Support for video-specific data augmentation

## Troubleshooting

### ImportError: No module named 'decord'

Install decord:
```bash
pip install decord
```

### CUDA out of memory

Reduce `num_video_frames` or `per_device_train_batch_size`:
```bash
--num_video_frames 4 --per_device_train_batch_size 2
```

### Video file not recognized

Ensure your video file has a supported extension (case-insensitive):
- Supported: `.mp4`, `.avi`, `.mov`, `.mkv`, `.flv`, `.wmv`, `.webm`, `.m4v`

### Shape mismatch errors

Check that your dataset JSON correctly references video files in the `"image"` field.

## Examples

### Single Video Training

```json
{
  "image": "cooking_tutorial.mp4",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nWhat cooking technique is demonstrated?"
    },
    {
      "from": "gpt",
      "value": "The video demonstrates the technique of dicing onions..."
    }
  ]
}
```

### Mixed Image and Video Dataset

```json
[
  {
    "image": "photo.jpg",
    "conversations": [...]
  },
  {
    "image": "video.mp4",
    "conversations": [...]
  }
]
```

The system automatically handles both images and videos in the same dataset.

## Citation

If you use this video fine-tuning extension in your research, please cite:

```bibtex
@software{fastvlm_video,
  title={FastVLM Video Fine-tuning Extension},
  year={2025},
  url={https://github.com/Gayanukaa/fastvlm-implementation}
}
```
