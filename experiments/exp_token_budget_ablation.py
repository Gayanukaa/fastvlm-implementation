#!/usr/bin/env python3
"""
FastVLM Token Budget Ablation Experiment
Tests different token budgets by varying image resolution/cropping and measures TTFT and accuracy.
"""

import argparse
import os
import time
import csv
import gc
import math
from typing import List, Dict, Tuple
import torch
import torch.cuda
from PIL import Image
import numpy as np
from transformers import AutoTokenizer

# LLaVA imports for custom model loading
from llava.model.builder import load_pretrained_model
from llava.conversation import conv_templates, SeparatorStyle
from llava.mm_utils import tokenizer_image_token, process_images, IMAGE_TOKEN_INDEX
from llava.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from tqdm import tqdm
import warnings
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk
warnings.filterwarnings("ignore")

# Download NLTK data if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("📦 Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

# Import plotting utilities
from utils_plot import save_plot, save_grouped_bar

def setup_model_and_tokenizer(model_path: str, device: str) -> Tuple:
    """Load model and tokenizer using LLaVA's approach for FastVLM."""
    print(f"🔧 Loading model from {model_path}...")

    # Use LLaVA's load_pretrained_model for FastVLM compatibility
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name="llava_qwen",
        load_8bit=False,
        load_4bit=False,
        device_map="auto",
        device=device
    )

    model.eval()
    print(f"✅ Model loaded on {device}")
    print(f"📏 Context length: {context_len}")
    return model, tokenizer, image_processor

def calculate_token_budget_size(target_tokens: int) -> int:
    """
    Calculate image size based on target token budget.
    Rough approximation: tokens ≈ (height * width) / patch_size^2
    Assuming patch_size = 16 for ViT-like models
    """
    patch_size = 16
    total_pixels = target_tokens * (patch_size ** 2)
    image_size = int(math.sqrt(total_pixels))

    # Round to common sizes
    common_sizes = [224, 256, 336, 384, 448, 512, 576, 672, 768, 896, 1024]
    image_size = min(common_sizes, key=lambda x: abs(x - image_size))

    return image_size

def preprocess_image_for_tokens(image_path: str, target_tokens: int, method: str = 'resize') -> Tuple[Image.Image, int]:
    """
    Preprocess image to approximate target token count.

    Args:
        image_path: Path to input image
        target_tokens: Target number of visual tokens
        method: 'resize' or 'center_crop'

    Returns:
        Processed image and actual estimated tokens
    """
    image = Image.open(image_path).convert('RGB')
    target_size = calculate_token_budget_size(target_tokens)

    if method == 'resize':
        # Resize maintaining aspect ratio, then pad
        w, h = image.size
        if w > h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Pad to square
        if new_w != new_h:
            padded = Image.new('RGB', (target_size, target_size), (128, 128, 128))
            offset = ((target_size - new_w) // 2, (target_size - new_h) // 2)
            padded.paste(image, offset)
            image = padded

    elif method == 'center_crop':
        # Center crop to target size
        w, h = image.size

        # First resize so that smaller dimension equals target_size
        if w < h:
            new_w = target_size
            new_h = int(h * target_size / w)
        else:
            new_h = target_size
            new_w = int(w * target_size / h)

        image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        image = image.crop((left, top, left + target_size, top + target_size))

    # Estimate actual tokens (rough approximation)
    patch_size = 16
    actual_tokens = (target_size // patch_size) ** 2

    return image, actual_tokens

def calculate_bleu_score(reference: str, candidate: str) -> float:
    """Calculate BLEU score between two texts."""
    try:
        ref_tokens = word_tokenize(reference.lower())
        cand_tokens = word_tokenize(candidate.lower())

        if len(cand_tokens) == 0:
            return 0.0

        # Use BLEU-4 with smoothing
        score = sentence_bleu([ref_tokens], cand_tokens, weights=(0.25, 0.25, 0.25, 0.25))
        return score
    except:
        return 0.0

def measure_inference_with_ttft(model, tokenizer, image_processor, image: Image.Image,
                               prompt: str, device: str, max_tokens: int = 50) -> Dict:
    """Measure inference with focus on TTFT."""

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

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

        # Generate first token
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16),
            max_new_tokens=1,  # Just first token for TTFT
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        if torch.cuda.is_available():
            first_token_event.record()

        first_token_time = time.perf_counter()

        # Generate full response
        full_outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16),
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        if torch.cuda.is_available():
            end_event.record()
            torch.cuda.synchronize()

        end_time = time.perf_counter()

    # Calculate metrics
    ttft_ms = (first_token_time - start_time) * 1000
    total_latency_ms = (end_time - start_time) * 1000

    if torch.cuda.is_available():
        cuda_ttft = start_event.elapsed_time(first_token_event)
        cuda_total = start_event.elapsed_time(end_event)
    else:
        cuda_ttft = ttft_ms
        cuda_total = total_latency_ms

    # Decode output
    generated_text = tokenizer.decode(full_outputs[0], skip_special_tokens=True)

    # Count tokens
    input_tokens = len(input_ids[0])
    output_tokens = len(full_outputs[0]) - input_tokens

    return {
        'ttft_ms': cuda_ttft,
        'total_latency_ms': cuda_total,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'generated_text': generated_text
    }

def run_token_budget_experiment(model_path: str, image_folder: str, prompt: str,
                               device: str, batch_size: int = 1) -> None:
    """Run token budget ablation experiment."""

    print("🚀 Starting Token Budget Ablation Experiment")
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Prompt: {prompt}")

    # Load model
    model, tokenizer, image_processor = setup_model_and_tokenizer(model_path, device)

    # Test token budgets (approximate visual tokens)
    token_budgets = [16, 64, 144, 256]

    # Find test image
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        raise ValueError(f"No images found in {image_folder}")

    test_image_path = os.path.join(image_folder, image_files[0])
    print(f"📷 Using test image: {test_image_path}")

    # Generate reference output (highest budget for comparison)
    print("🎯 Generating reference output with highest budget...")
    ref_image, _ = preprocess_image_for_tokens(test_image_path, max(token_budgets))
    ref_result = measure_inference_with_ttft(model, tokenizer, image_processor, ref_image, prompt, device)
    reference_text = ref_result['generated_text']
    print(f"📝 Reference text: {reference_text[:100]}...")

    # Results storage
    results = []

    # Test different preprocessing methods
    methods = ['resize', 'center_crop']

    for method in methods:
        print(f"\n🔄 Testing method: {method}")

        for budget in tqdm(token_budgets, desc=f"Testing budgets ({method})"):
            print(f"\n🎯 Testing token budget: {budget} tokens ({method})")

            # Preprocess image
            image, actual_tokens = preprocess_image_for_tokens(test_image_path, budget, method)

            # Run inference multiple times for averaging
            run_results = []
            bleu_scores = []

            for run in range(3):  # Average over 3 runs
                try:
                    metrics = measure_inference_with_ttft(
                        model, tokenizer, image_processor, image, prompt, device
                    )
                    run_results.append(metrics)

                    # Calculate BLEU score vs reference
                    bleu = calculate_bleu_score(reference_text, metrics['generated_text'])
                    bleu_scores.append(bleu)

                    print(f"  Run {run+1}: TTFT {metrics['ttft_ms']:.1f}ms, "
                          f"BLEU: {bleu:.3f}")

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
                    'method': method,
                    'target_tokens': budget,
                    'actual_tokens': actual_tokens,
                    'ttft_ms': np.mean([r['ttft_ms'] for r in run_results]),
                    'total_latency_ms': np.mean([r['total_latency_ms'] for r in run_results]),
                    'input_tokens': int(np.mean([r['input_tokens'] for r in run_results])),
                    'output_tokens': int(np.mean([r['output_tokens'] for r in run_results])),
                    'bleu_score': np.mean(bleu_scores),
                    'generated_text': run_results[0]['generated_text'][:200] + '...'
                }
                results.append(avg_result)

                print(f"  ✅ Average: TTFT {avg_result['ttft_ms']:.1f}ms, "
                      f"BLEU: {avg_result['bleu_score']:.3f}")

    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/token_budget.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['method', 'target_tokens', 'actual_tokens', 'ttft_ms', 'total_latency_ms',
                     'input_tokens', 'output_tokens', 'bleu_score', 'generated_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Results saved to {csv_path}")

    # Generate plots
    if results:
        # Separate results by method
        resize_results = [r for r in results if r['method'] == 'resize']
        crop_results = [r for r in results if r['method'] == 'center_crop']

        if resize_results:
            budgets = [r['target_tokens'] for r in resize_results]
            ttft_values = [r['ttft_ms'] for r in resize_results]
            bleu_values = [r['bleu_score'] for r in resize_results]

            # Plot 1: Token Budget vs TTFT
            save_plot(
                budgets, ttft_values,
                'Token Budget', 'Time to First Token (ms)',
                'FastVLM: Token Budget vs TTFT (Resize)',
                'token_budget_vs_ttft_resize.png'
            )

            # Plot 2: Token Budget vs BLEU Score
            save_plot(
                budgets, bleu_values,
                'Token Budget', 'BLEU Score vs Reference',
                'FastVLM: Token Budget vs Accuracy (Resize)',
                'token_budget_vs_bleu_resize.png'
            )

        # Compare methods if both exist
        if resize_results and crop_results:
            budgets = [r['target_tokens'] for r in resize_results]

            data = {
                'Resize TTFT': [r['ttft_ms'] for r in resize_results],
                'Crop TTFT': [r['ttft_ms'] for r in crop_results]
            }

            save_grouped_bar(
                data, [str(b) for b in budgets],
                'Token Budget', 'Time to First Token (ms)',
                'FastVLM: TTFT Comparison - Resize vs Center Crop',
                'token_budget_methods_comparison.png'
            )

            # BLEU comparison
            bleu_data = {
                'Resize BLEU': [r['bleu_score'] for r in resize_results],
                'Crop BLEU': [r['bleu_score'] for r in crop_results]
            }

            save_grouped_bar(
                bleu_data, [str(b) for b in budgets],
                'Token Budget', 'BLEU Score',
                'FastVLM: Accuracy Comparison - Resize vs Center Crop',
                'token_budget_bleu_comparison.png'
            )

    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("="*70)
    print(f"{'Method':<12} {'Budget':<8} {'Actual':<8} {'TTFT(ms)':<10} {'BLEU':<8}")
    print("-"*70)
    for result in results:
        print(f"{result['method']:<12} {result['target_tokens']:<8} "
              f"{result['actual_tokens']:<8} {result['ttft_ms']:<10.1f} "
              f"{result['bleu_score']:<8.3f}")

    print(f"\n✅ Token budget ablation experiment completed!")
    print(f"📁 Results: {csv_path}")
    print(f"📈 Plots: results/plots/")

def main():
    parser = argparse.ArgumentParser(description='FastVLM Token Budget Ablation Experiment')
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

    run_token_budget_experiment(
        args.model_path, args.image_folder, args.prompt, args.device, args.batch_size
    )

if __name__ == "__main__":
    main()