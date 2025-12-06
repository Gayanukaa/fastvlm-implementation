#
# Video inference script for FastVLM
# Based on predict.py but adapted for video input
#
import os
import argparse

import torch
from PIL import Image

from llava.utils import disable_torch_init
from llava.conversation import conv_templates
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path, load_video_frames, is_video_file
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def predict_video(args):
    # Remove generation config from model folder
    # to read generation parameters from args
    model_path = os.path.expanduser(args.model_path)
    generation_config = None
    if os.path.exists(os.path.join(model_path, 'generation_config.json')):
        generation_config = os.path.join(model_path, '.generation_config.json')
        os.rename(os.path.join(model_path, 'generation_config.json'),
                  generation_config)

    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load model
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name, device=device
    )

    # Check if input is video or image
    input_file = args.input_file
    is_video = is_video_file(input_file)

    if is_video:
        print(f"Processing video: {input_file}")
        print(f"Extracting {args.num_frames} frames...")

        # Load video frames
        frames = load_video_frames(input_file, num_frames=args.num_frames)
        num_frames = len(frames)
        print(f"Loaded {num_frames} frames, size: {frames[0].size}")

        # Process each frame
        image_tensors = []
        for frame in frames:
            frame_tensor = process_images([frame], image_processor, model.config)[0]
            image_tensors.append(frame_tensor)

        # Stack frames: (T, C, H, W)
        image_tensor = torch.stack(image_tensors, dim=0)
        image_sizes = [frames[0].size] * num_frames

        # Construct prompt with multiple image tokens for video frames
        qs = args.prompt
        if model.config.mm_use_im_start_end:
            image_tokens = (DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN) * num_frames
            qs = image_tokens + '\n' + qs
        else:
            image_tokens = DEFAULT_IMAGE_TOKEN * num_frames
            qs = image_tokens + '\n' + qs
    else:
        print(f"Processing image: {input_file}")

        # Load and preprocess single image
        image = Image.open(input_file).convert('RGB')
        image_tensor = process_images([image], image_processor, model.config)[0]
        image_sizes = [image.size]
        num_frames = 1

        # Construct prompt for single image
        qs = args.prompt
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    # Build conversation
    conv = conv_templates[args.conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Set the pad token id for generation
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Tokenize prompt
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Prepare image tensor for inference
    if is_video:
        # For video: (1, T, C, H, W) - add batch dimension
        images = image_tensor.unsqueeze(0).to(device)
        if device == "cuda":
            images = images.half()
        elif device == "mps":
            images = images.half()
    else:
        # For single image: (1, C, H, W)
        images = image_tensor.unsqueeze(0).to(device)
        if device == "cuda":
            images = images.half()
        elif device == "mps":
            images = images.half()

    # Run inference
    print("\nGenerating response...")
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=images,
            image_sizes=image_sizes,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True
        )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        print("\n" + "=" * 50)
        print("RESPONSE:")
        print("=" * 50)
        print(outputs)
        print("=" * 50)

    # Restore generation config
    if generation_config is not None:
        os.rename(generation_config, os.path.join(model_path, 'generation_config.json'))

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FastVLM Video/Image Inference")
    parser.add_argument("--model-path", type=str, default="./checkpoints/exercise-video-finetuned",
                        help="Path to the fine-tuned model")
    parser.add_argument("--model-base", type=str, default=None,
                        help="Base model path (if using LoRA)")
    parser.add_argument("--input-file", type=str, required=True,
                        help="Path to input video (.mp4, .avi, etc.) or image file")
    parser.add_argument("--prompt", type=str,
                        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
                        help="Prompt for the model")
    parser.add_argument("--conv-mode", type=str, default="qwen_2",
                        help="Conversation template mode")
    parser.add_argument("--num-frames", type=int, default=4,
                        help="Number of frames to extract from video (default: 4)")
    parser.add_argument("--temperature", type=float, default=0.2,
                        help="Sampling temperature (0 for greedy)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1,
                        help="Number of beams for beam search")
    parser.add_argument("--max-new-tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"],
                        help="Device to run inference on")
    args = parser.parse_args()

    predict_video(args)
