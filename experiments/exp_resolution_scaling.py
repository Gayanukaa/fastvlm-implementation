#!/usr/bin/env python3
"""
FastVLM Resolution Scaling Experiment
Tests different inddef measure_inference(model, tokenizer, image_processor, image: Image.Image,
                     prompt: str, device: str, max_tokens: int = 50) -> Dict: measure_inference(model, tokenizer, image_processor, image: Image.Image,
                     prompt: str, device: str, max_tokens: int = 50) -> Dict:t resolutions and measures latency, VRAM usage, and token generation.
"""

import argparse
import os
import sys
import time
import csv
import gc
from typing import List, Dict, Tuple
import torch
import torch.cuda
from PIL import Image
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add the parent directory to Python path to import llava modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

# Import plotting utilities
from utils_plot import save_plot, save_dual_axis_plot

def setup_model_and_tokenizer(model_path: str, device: str) -> Tuple:
    """Load model and tokenizer with optimizations for 8GB VRAM."""
    print(f"🔧 Loading model from {model_path}...")

    # Initialize torch
    disable_torch_init()

    # Load model using LLaVA's model builder
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,  # model_base
        model_name,
        device=device,
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        torch_dtype=torch.float16
    )

    model.eval()
    print(f"✅ Model loaded on {device}")
    print(f"📏 Context length: {context_len}")
    return model, tokenizer, image_processor

def preprocess_image(image_path: str, target_size: int) -> Image.Image:
    """Resize image to target resolution while maintaining aspect ratio."""
    image = Image.open(image_path).convert('RGB')

    # Calculate new size maintaining aspect ratio
    w, h = image.size
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)

    # Resize and pad to square if needed
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    # Pad to square
    if new_w != new_h:
        padded = Image.new('RGB', (target_size, target_size), (0, 0, 0))
        offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
        padded.paste(image, offset)
        image = padded

    return image

def measure_inference(model, tokenizer, image_processor, image: Image.Image,
                     prompt: str, device: str, max_tokens: int = 50) -> Dict:
    """Measure inference metrics with CUDA events for precise timing."""

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # Record initial VRAM
    vram_before = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0

    # Prepare input using LLaVA's approach
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates['qwen_2'].copy()  # Use qwen_2 template for qwen2 models
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()

    # Tokenize prompt
    input_ids = tokenizer_image_token(prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device=device)

    # Process image
    image_tensor = process_images([image], image_processor, model.config)[0]

    # CUDA events for precise timing
    if torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    # Generate response
    with torch.no_grad():
        if torch.cuda.is_available():
            start_event.record()

        start_time = time.perf_counter()

        # Generate tokens
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16),
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )

        if torch.cuda.is_available():
            first_token_event.record()

        first_token_time = time.perf_counter()

        # Decode output
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()

        end_time = time.perf_counter()

    # Calculate metrics
    total_latency_ms = (end_time - start_time) * 1000
    ttft_ms = (first_token_time - start_time) * 1000

    if torch.cuda.is_available():
        cuda_total_latency = start_event.elapsed_time(end_event)
        cuda_ttft = start_event.elapsed_time(first_token_event)
    else:
        cuda_total_latency = total_latency_ms
        cuda_ttft = ttft_ms

    # VRAM usage
    vram_peak = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
    vram_used = vram_peak - vram_before

    # Count tokens
    input_tokens = len(input_ids[0])
    output_tokens = len(outputs[0]) - input_tokens

    return {
        'total_latency_ms': cuda_total_latency,
        'ttft_ms': cuda_ttft,
        'vram_used_gb': vram_used,
        'vram_peak_gb': vram_peak,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'generated_text': generated_text[:200] + '...' if len(generated_text) > 200 else generated_text
    }

def run_resolution_experiment(model_path: str, image_folder: str, prompt: str,
                            device: str, batch_size: int = 1) -> None:
    """Run resolution scaling experiment."""

    print("🚀 Starting Resolution Scaling Experiment")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")

    # Load model
    model, tokenizer, image_processor = setup_model_and_tokenizer(model_path, device)

    # Test resolutions
    resolutions = [256, 512, 768, 1024]

    # Find test image
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    test_image_path = os.path.join(image_folder, image_files[0])
    print(f"📷 Using test image: {test_image_path}")

    # Results storage
    results = []

    # Run experiments
    for resolution in tqdm(resolutions, desc="Testing resolutions"):
        print(f"\n📐 Testing resolution: {resolution}x{resolution}")

        # Preprocess image
        image = preprocess_image(test_image_path, resolution)

        # Run inference multiple times for averaging
        run_results = []
        for run in range(3):  # Average over 3 runs
            try:
                metrics = measure_inference(
                    model, tokenizer, image_processor, image, prompt, device
                )
                run_results.append(metrics)
                print(f"  Run {run+1}: {metrics['total_latency_ms']:.1f}ms, "
                      f"VRAM: {metrics['vram_used_gb']:.2f}GB")

                # Clean up between runs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"  ❌ Run {run+1} failed: {e}")
                continue

        if run_results:
            # Average the results
            avg_result = {
                'resolution': resolution,
                'total_latency_ms': np.mean([r['total_latency_ms'] for r in run_results]),
                'ttft_ms': np.mean([r['ttft_ms'] for r in run_results]),
                'vram_used_gb': np.mean([r['vram_used_gb'] for r in run_results]),
                'vram_peak_gb': np.mean([r['vram_peak_gb'] for r in run_results]),
                'input_tokens': int(np.mean([r['input_tokens'] for r in run_results])),
                'output_tokens': int(np.mean([r['output_tokens'] for r in run_results])),
                'generated_text': run_results[0]['generated_text']  # Use first run's text
            }
            results.append(avg_result)

        print(f"  ✅ Average: {avg_result['total_latency_ms']:.1f}ms, "
              f"VRAM: {avg_result['vram_used_gb']:.2f}GB")

    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/resolution_scaling.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['resolution', 'total_latency_ms', 'ttft_ms', 'vram_used_gb',
                     'vram_peak_gb', 'input_tokens', 'output_tokens', 'generated_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Results saved to {csv_path}")

    # Generate plots
    if results:
        resolutions_list = [r['resolution'] for r in results]
        latencies = [r['total_latency_ms'] for r in results]
        vram_usage = [r['vram_used_gb'] for r in results]
        ttft_values = [r['ttft_ms'] for r in results]

        # Plot 1: Resolution vs Total Latency
        save_plot(
            resolutions_list, latencies,
            'Resolution (pixels)', 'Total Latency (ms)',
            'FastVLM: Resolution vs Total Latency',
            'resolution_vs_latency.png'
        )

        # Plot 2: Resolution vs VRAM Usage
        save_plot(
            resolutions_list, vram_usage,
            'Resolution (pixels)', 'VRAM Usage (GB)',
            'FastVLM: Resolution vs VRAM Usage',
            'resolution_vs_vram.png'
        )

        # Plot 3: Resolution vs TTFT
        save_plot(
            resolutions_list, ttft_values,
            'Resolution (pixels)', 'Time to First Token (ms)',
            'FastVLM: Resolution vs TTFT',
            'resolution_vs_ttft.png'
        )

        # Plot 4: Dual axis plot (Latency + VRAM)
        save_dual_axis_plot(
            resolutions_list, latencies, vram_usage,
            'Resolution (pixels)', 'Total Latency (ms)', 'VRAM Usage (GB)',
            'FastVLM: Resolution Scaling - Latency & VRAM',
            'resolution_scaling_dual.png'
        )

    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("="*50)
    for result in results:
        print(f"Resolution {result['resolution']:4d}: "
              f"{result['total_latency_ms']:6.1f}ms | "
              f"TTFT: {result['ttft_ms']:6.1f}ms | "
              f"VRAM: {result['vram_used_gb']:5.2f}GB")

    print(f"\n✅ Resolution scaling experiment completed!")
    print(f"📁 Results: {csv_path}")
    print(f"📈 Plots: results/plots/")

def main():
    parser = argparse.ArgumentParser(description='FastVLM Resolution Scaling Experiment')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--image-folder', default='../images', help='Folder containing test images')
    parser.add_argument('--prompt', default='Describe this image in detail.', help='Test prompt')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (keep 1 for VRAM)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder not found: {args.image_folder}")

    run_resolution_experiment(
        args.model_path, args.image_folder, args.prompt, args.device, args.batch_size
    )

if __name__ == "__main__":
    main()