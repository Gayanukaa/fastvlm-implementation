# Video Fine-tuning Adaptation for FastVLM

This document describes the modifications made to adapt FastVLM's training pipeline from image-only fine-tuning to support video fine-tuning.

## Table of Contents

1. [Background](#background)
2. [Original Limitations](#original-limitations)
3. [Adaptation Strategy](#adaptation-strategy)
4. [Implementation Details](#implementation-details)
5. [Tokenization Handling](#tokenization-handling)
6. [Validation Dataset Support](#validation-dataset-support)
7. [References](#references)

## Background

FastVLM is a vision-language model that uses FastViT-HD as its vision encoder, producing significantly fewer visual tokens per image (approximately 100 tokens) compared to standard CLIP-based models (576 tokens).

The original FastVLM training code was designed exclusively for image-text pair fine-tuning. This document describes the changes required to enable video fine-tuning.

## Original Limitations

The training code in `llava/train/train.py` and `llava/train/train_qwen.py` was designed specifically for image-text pairs, not videos. The following limitations existed:

### 1. Image-Only Dataset Loading

The `LazySupervisedDataset` class only handled single images:

```python
if 'image' in sources[0]:
    image_file = self.list_data_dict[i]['image']
    image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
```

This code assumed every media file was an image and used PIL's `Image.open()` directly, which cannot process video files.

### 2. Single Image Processing

The `preprocess_multimodal` function expected individual images. It processed a single `<image>` token in the conversation:

```python
if DEFAULT_IMAGE_TOKEN in sentence['value']:
    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
    sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
```

There was no mechanism to expand this token for multiple video frames.

### 3. No Video-Related Parameters

The `DataArguments` class only contained image-related fields:

```python
@dataclass
class DataArguments:
    data_path: Optional[List[str]] = field(default=None)
    image_folder: Optional[List[str]] = field(default=None)
    image_aspect_ratio: str = 'square'
    # No video parameters
```

## Adaptation Strategy

The adaptation follows the **"frames as images"** approach, also known as **sparse temporal sampling**. This strategy treats videos as a sequence of uniformly sampled frames, where each frame is processed independently through the vision encoder before being concatenated as visual tokens for the language model.

### Why This Approach

1. **Simplicity**: No architectural changes required to the vision encoder or language model.
2. **Compatibility**: Existing image-trained models can be directly fine-tuned on videos.

### How It Works

Think of it like a flipbook:

1. **Extract Pages**: Take N evenly-spaced snapshots from the video (for example, 4 frames from a 10-second video)
2. **Process Each Page**: Each frame goes through the same "image understanding" process
3. **Combine Understanding**: All the processed frames are placed together so the model sees them as one sequence
4. **Answer Questions**: The model reads all frames together to understand what happens in the video

In technical terms:

- A single `<image>` token in the prompt is expanded to N `<image>` tokens (one per frame)
- Each token corresponds to the visual features of one frame
- The language model processes all frame features together with the text

## Implementation Details

### 1. Video Loading Functions (`llava/mm_utils.py`)

Two new utility functions were added:

**`is_video_file(file_path)`**: Determines if a file is a video based on its extension.

```python
def is_video_file(file_path):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v'}
    _, ext = os.path.splitext(file_path.lower())
    return ext in video_extensions
```

**`load_video_frames(video_path, num_frames)`**: Extracts N frames uniformly distributed across the video duration using the `decord` library.

```python
def load_video_frames(video_path, num_frames=8):
    from decord import VideoReader, cpu
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frames = len(vr)

    # Sample N frames uniformly across the video
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    frames = vr.get_batch(indices).asnumpy()

    return [Image.fromarray(frame) for frame in frames]
```

The uniform sampling ensures temporal coverage regardless of video length. For a 30-second video with `num_frames=4`, frames would be extracted at approximately 0s, 10s, 20s, and 30s.

### 2. DataArguments Extension (`llava/train/train_qwen.py`)

A new parameter was added to control frame extraction:

```python
@dataclass
class DataArguments:
    # ... existing fields ...
    num_video_frames: int = field(
        default=8,
        metadata={"help": "Number of frames to extract from video files"}
    )
```

### 3. Dataset Modification (`LazySupervisedDataset.__getitem__`)

The dataset's `__getitem__` method was modified to detect and handle video files:

```python
if is_video_file(full_path):
    # Load video frames
    num_frames = self.data_args.num_video_frames
    frames = load_video_frames(full_path, num_frames=num_frames)

    # Process each frame through the image processor
    image_tensors = []
    for frame in frames:
        frame_tensor = processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
        image_tensors.append(frame_tensor)

    # Stack frames: (T, C, H, W)
    image = torch.stack(image_tensors, dim=0)
else:
    # Original image loading logic
    image = Image.open(full_path).convert('RGB')
```

The output tensor shape changes from `(C, H, W)` for images to `(T, C, H, W)` for videos, where T is the number of frames.

### 4. Token Expansion (`preprocess_multimodal`)

The preprocessing function was modified to expand the image token for videos:

```python
def preprocess_multimodal(sources, data_args, num_frames=1):
    for source in sources:
        for sentence in source:
            # For video: expand image token to num_frames tokens
            if num_frames > 1:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN,
                    DEFAULT_IMAGE_TOKEN * num_frames
                )
```

This ensures the tokenizer creates the correct number of placeholder positions for visual embeddings. For a 4-frame video, a single `<image>` becomes `<image><image><image><image>`.

### 5. Vision Encoder Handling (`llava/model/llava_arch.py`)

The `prepare_inputs_labels_for_multimodal` function was updated to handle 5D tensors:

```python
if images.dim() == 5:  # Video: (B, T, C, H, W)
    batch_size, num_frames = images.shape[:2]
    # Flatten temporal dimension for vision encoder
    images = images.view(batch_size * num_frames, *images.shape[2:])
    # Process all frames through vision encoder
    # Reshape back and concatenate features
```

## Tokenization Handling

During adaptation, a common issue arose where tokenization mismatches occurred between the expected and actual token counts. This happened due to differences in how separator tokens (like `<|im_end|>`) were handled.

### The Problem

The original code discarded any sample with a tokenization mismatch:

```python
if cur_len != total_len:
    target[:] = IGNORE_INDEX  # Discard entire sample
    print(f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}. (ignored)")
```

With video data, small mismatches of 1-2 tokens were common due to separator token handling, causing all samples to be discarded and resulting in zero loss during training.

### The Solution

The code was modified to tolerate small mismatches (up to 5 tokens) while still discarding samples with large mismatches that indicate real problems:

```python
diff = abs(cur_len - total_len)
if diff > 5:
    target[:] = IGNORE_INDEX
    print(f"WARNING: large tokenization mismatch (diff={diff}). Sample ignored.")
# Small mismatches (<=5 tokens) are silently accepted
```

Additionally, the separator token length is now dynamically calculated:

```python
sep_token_len = len(tokenizer(conv.sep, add_special_tokens=False).input_ids)
round_len += sep_token_len  # Proper accounting for separator tokens
```

## Validation Dataset Support

Support for evaluation during training was added through a new `validation_data_path` argument:

```python
@dataclass
class DataArguments:
    validation_data_path: Optional[str] = field(
        default=None,
        metadata={"help": "Optional path to the validation data."}
    )
```

The `make_supervised_data_module` function creates an evaluation dataset when this path is provided:

```python
def make_supervised_data_module(tokenizer, data_args):
    train_dataset = LazySupervisedDataset(...)

    eval_dataset = None
    if data_args.validation_data_path:
        val_data_path = [data_args.validation_data_path]
        eval_dataset = LazySupervisedDataset(
            tokenizer=tokenizer,
            data_path=val_data_path,
            data_args=data_args
        )

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator
    )
```

## Summary of Modified Files

| File                        | Changes                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `llava/mm_utils.py`         | Added `load_video_frames()` and `is_video_file()` functions                                                                          |
| `llava/train/train.py`      | Added `num_video_frames` and `validation_data_path` to DataArguments, modified dataset loading, fixed tokenization mismatch handling |
| `llava/train/train_qwen.py` | Same changes as train.py, plus separator token length calculation                                                                    |
| `llava/model/llava_arch.py` | Updated to handle 5D tensors for video frames                                                                                        |
| `llava/model/builder.py`    | Improved vision tower detection for model loading                                                                                    |
| `pyproject.toml`            | Added `decord` dependency                                                                                                            |

---

## References

The "frames as images" approach used in this adaptation is established in the video understanding literature:

> 1. **Lei, J., Li, L., Berg, T., & Bansal, M. (2021). "Less is More: ClipBERT for Video-and-Language Learning via Sparse Sampling."** _CVPR 2021._ This paper demonstrates that sparse sampling of video frames is sufficient for video-language tasks, achieving competitive performance while significantly reducing computational costs. The key insight is that uniform sampling captures sufficient temporal information for most video understanding tasks.

> 2. **Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., ... & Qiao, Y. (2023). "VideoChat: Chat-Centric Video Understanding."** _arXiv preprint arXiv:2305.06355._ This work extends image-language models to video by processing uniformly sampled frames through a frozen image encoder and concatenating the resulting features, similar to the approach implemented here.

### Code & Implementation Resources

For further reading and reference implementations of similar video adaptation techniques:

- **[Video-LLaVA](https://github.com/PKU-YuanGroup/Video-LLaVA)**: A direct adaptation of LLaVA for video that projects video features into the language feature space. Their `TRAIN_AND_VALIDATE.md` provides excellent guidance on data preparation and training pipelines.
- **[Video-ChatGPT](https://github.com/mbzuai-oryx/Video-ChatGPT)**: Implements a video adapter to connect visual encoders with LLMs using spatiotemporal representations.
- **[LLaVA-NeXT (Video Support)](https://github.com/LLaVA-VL/LLaVA-NeXT)**: The official LLaVA repository's implementation of video support, which treats video frames as a grid of images (AnyRes) or sequences.
- **[Decord Documentation](https://github.com/dmlc/decord)**: The library used in this project for efficient video loading and frame sampling.

## Usage

See `utils/README.md` for complete instructions on preparing datasets and running fine-tuning.

Basic training command:

```bash
bash scripts/finetune_video.sh
```
