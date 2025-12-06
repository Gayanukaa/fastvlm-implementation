#!/usr/bin/env python3
"""
Test Inference Script for FastVLM Exercise Dataset

This script runs inference on videos from the FastVLM test set using a finetuned model.
It loads videos from fastvlm_test.json and generates predictions.

Usage:
    python utils/test_inference.py --model_path models/exercise-video-finetuned
    python utils/test_inference.py --model_path models/exercise-video-finetuned --output test_predictions.json
"""

import sys
import os
import warnings
import logging
import argparse
import json

os.environ['PYTHONWARNINGS'] = 'ignore'

warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
    load_video_frames,
    is_video_file
)
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN
)


def load_model(model_path: str, device: str = "cuda", model_base: str = None):
    """Loads the FastVLM model and tokenizer.

    Args:
        model_path: Path to finetuned model directory
        device: Device to load model on
        model_base: Base model path (optional, for LoRA adapters)

    Returns:
        tuple: (model, tokenizer, image_processor)
    """
    print(f"Loading FastVLM model from: {model_path}")

    disable_torch_init()
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name, device=device
    )

    return model, tokenizer, image_processor


def run_inference(
    model,
    tokenizer,
    image_processor,
    video_path: str,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
    num_frames: int = 4,
    conv_mode: str = "qwen_2"
):
    """Runs inference on the given video file using FastVLM.

    Args:
        model: The loaded FastVLM model
        tokenizer: The tokenizer
        image_processor: The image processor
        video_path: Path to the video file
        prompt: The text prompt for the model
        device: Device to run inference on
        max_new_tokens: Maximum tokens to generate
        num_frames: Number of frames to extract from video
        conv_mode: Conversation template to use

    Returns:
        str: Model's generated response
    """
    is_video = is_video_file(video_path)

    if is_video:
        # Load video frames
        frames = load_video_frames(video_path, num_frames=num_frames)
        num_loaded_frames = len(frames)

        # Process each frame
        image_tensors = []
        for frame in frames:
            frame_tensor = process_images([frame], image_processor, model.config)[0]
            image_tensors.append(frame_tensor)

        # Stack frames: (T, C, H, W)
        image_tensor = torch.stack(image_tensors, dim=0)
        image_sizes = [frames[0].size] * num_loaded_frames

        # Construct prompt with multiple image tokens for video frames
        qs = prompt
        if model.config.mm_use_im_start_end:
            image_tokens = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) * num_loaded_frames
            qs = image_tokens + '\n' + qs
        else:
            image_tokens = DEFAULT_IMAGE_TOKEN * num_loaded_frames
            qs = image_tokens + '\n' + qs
    else:
        # Single image
        image = Image.open(video_path).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_sizes = [image.size]
        num_loaded_frames = 1

        # Construct prompt for single image
        qs = prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Build conversation
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(
        prompt_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Prepare image tensor for inference
    if is_video:
        # For video: (1, T, C, H, W) - add batch dimension
        images = image_tensor.unsqueeze(0).to(device)
    else:
        # For single image: (1, C, H, W)
        images = image_tensor.unsqueeze(0).to(device)

    # Convert to half precision for GPU
    if device in ["cuda", "mps"]:
        images = images.half()

    # Run inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=False,  # Use greedy decoding for reproducibility
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Run inference on FastVLM test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned FastVLM model directory")
    parser.add_argument("--model_base", type=str, default=None,
                        help="Base model path (for LoRA adapters)")
    parser.add_argument("--test_json", type=str, default="dataset/fastvlm_test.json",
                        help="Path to test set JSON")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Base path for video files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions (default: saves to model directory)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/mps/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--num_frames", type=int, default=4,
                        help="Number of frames to extract from each video")
    parser.add_argument("--conv_mode", type=str, default="qwen_2",
                        help="Conversation template to use")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")

    args = parser.parse_args()

    # Auto-detect device if needed
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # Set default output path to model directory if not provided
    if args.output is None:
        args.output = str(Path(args.model_path) / "test_predictions.json")
        print(f"Output will be saved to: {args.output}")

    # Load model
    print(f"📦 Loading model from: {args.model_path}")
    model, tokenizer, image_processor = load_model(
        args.model_path,
        device=args.device,
        model_base=args.model_base
    )

    # Load test data
    print(f"\n📋 Loading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Total test samples: {len(test_data)}")

    # Run inference
    results = []
    print("\n🎬 Running inference...")

    for item in tqdm(test_data, desc="Processing videos"):
        video_rel_path = item['video']
        video_path = str(Path(args.data_path) / video_rel_path)

        # Extract prompt and ground truth
        conversations = item['conversations']
        prompt = conversations[0]['value']
        ground_truth = conversations[1]['value']

        try:
            # Run inference
            prediction = run_inference(
                model, tokenizer, image_processor,
                video_path, prompt,
                args.device, args.max_new_tokens,
                args.num_frames, args.conv_mode
            )

            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "status": "success"
            })

        except Exception as e:
            print(f"\n✗ Error processing {video_rel_path}: {str(e)}")
            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": "",
                "status": "error",
                "error": str(e)
            })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("✅ Inference Complete!")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
