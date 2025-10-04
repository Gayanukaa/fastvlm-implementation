#!/usr/bin/env python3
"""
FastVLM Prompt Length Effect Experiment
Tests how different prompt lengths affect inference latency and output quality.
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
from utils_plot import save_plot, save_grouped_bar, save_dual_axis_plot

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

def preprocess_image(image_path: str, target_size: int = 512) -> Image.Image:
    """Preprocess image to standard size."""
    image = Image.open(image_path).convert('RGB')

    # Resize maintaining aspect ratio
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

    return image

def get_test_prompts_by_length() -> List[Tuple[int, str, str]]:
    """Get test prompts of different lengths (target_words, category, prompt)."""
    return [
        # Short prompts (~5 words)
        (5, "short_caption", "Describe this image."),
        (5, "short_question", "What do you see?"),
        (5, "short_identify", "What is this?"),

        # Medium prompts (~20 words)
        (20, "medium_caption", "Please provide a detailed description of what you see in this image, including the main subjects and their activities."),
        (20, "medium_analysis", "Analyze this image and explain what is happening, including any notable details or interesting aspects you observe."),
        (20, "medium_context", "Describe this image in context, explaining not just what you see but also the setting and environment."),

        # Long prompts (~60 words)
        (60, "long_detailed", "Please provide a comprehensive and detailed analysis of this image, describing all visible elements including people, objects, colors, composition, lighting, and atmosphere. Explain what story or message this image might be conveying, and discuss any artistic or technical aspects that make it noteworthy or interesting."),
        (60, "long_professional", "As a professional image analyst, examine this photograph carefully and provide a thorough description that covers the technical aspects such as composition, lighting, and visual elements, as well as the contextual information including the setting, subjects, activities, and any cultural or social significance that can be inferred from the visual content."),
        (60, "long_creative", "Take on the role of an art critic and creative writer to analyze this image in great detail. Describe not only what is literally visible but also explore the emotional impact, artistic techniques, symbolic meanings, and broader implications. Discuss the mood, atmosphere, and any metaphorical or deeper meanings that emerge from careful examination of all visual elements.")
    ]

def count_words(text: str) -> int:
    """Count words in text."""
    return len(text.split())

def measure_inference(model, tokenizer, image_processor, image: Image.Image,
                     prompt: str, device: str, max_tokens: int = 150) -> Dict:
    """Measure inference metrics."""

    # Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Count prompt tokens
    prompt_tokens = len(tokenizer.encode(prompt))

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

        # Generate first token for TTFT
        first_output = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device=device, dtype=torch.float16),
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

        if torch.cuda.is_available():
            first_token_event.record()

        first_token_time = time.perf_counter()

        # Generate full response
        outputs = model.generate(
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
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Clean up prompt from output
    if prompt in generated_text:
        generated_text = generated_text.replace(prompt, "").strip()

    # Count tokens and words
    input_tokens = len(input_ids[0])
    output_tokens = len(outputs[0]) - input_tokens
    output_words = count_words(generated_text)

    return {
        'ttft_ms': cuda_ttft,
        'total_latency_ms': cuda_total,
        'prompt_tokens': prompt_tokens,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'output_words': output_words,
        'generated_text': generated_text
    }

def run_prompt_length_experiment(model_path: str, image_folder: str,
                               device: str, batch_size: int = 1) -> None:
    """Run prompt length effect experiment."""

    print("🚀 Starting Prompt Length Effect Experiment")
    print(f"Model: {model_path}")
    print(f"Device: {device}")

    # Load model
    model, tokenizer, image_processor = setup_model_and_tokenizer(model_path, device)

    # Find test images (use first 3 to save time)
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:3]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")

    print(f"📷 Using {len(image_files)} test images")

    # Get test prompts
    test_prompts = get_test_prompts_by_length()
    print(f"📝 Using {len(test_prompts)} prompts across 3 length categories")

    # Results storage
    results = []

    # Test on all image-prompt combinations
    for img_idx, image_file in enumerate(tqdm(image_files, desc="Testing images")):
        image_path = os.path.join(image_folder, image_file)
        image = preprocess_image(image_path)

        print(f"\n📷 Testing {image_file}")

        for target_words, category, prompt in test_prompts:
            actual_words = count_words(prompt)
            print(f"  📝 {category} ({actual_words} words): {prompt[:50]}...")

            # Run inference multiple times for averaging
            run_results = []

            for run in range(3):  # 3 runs per combination
                try:
                    metrics = measure_inference(
                        model, tokenizer, image_processor, image, prompt, device
                    )
                    run_results.append(metrics)

                    # Clean up between runs
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"    ❌ Run {run+1} failed: {e}")
                    continue

            if run_results:
                # Average the results
                avg_result = {
                    'image': image_file,
                    'target_words': target_words,
                    'actual_words': actual_words,
                    'category': category,
                    'prompt': prompt,
                    'ttft_ms': np.mean([r['ttft_ms'] for r in run_results]),
                    'total_latency_ms': np.mean([r['total_latency_ms'] for r in run_results]),
                    'prompt_tokens': int(np.mean([r['prompt_tokens'] for r in run_results])),
                    'input_tokens': int(np.mean([r['input_tokens'] for r in run_results])),
                    'output_tokens': int(np.mean([r['output_tokens'] for r in run_results])),
                    'output_words': int(np.mean([r['output_words'] for r in run_results])),
                    'generated_text': run_results[0]['generated_text'][:200] + '...'  # Truncate for CSV
                }
                results.append(avg_result)

                print(f"    ✅ Latency: {avg_result['total_latency_ms']:.1f}ms, "
                      f"Output: {avg_result['output_words']} words")

    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/prompt_length.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['image', 'target_words', 'actual_words', 'category', 'prompt',
                     'ttft_ms', 'total_latency_ms', 'prompt_tokens', 'input_tokens',
                     'output_tokens', 'output_words', 'generated_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Results saved to {csv_path}")

    # Generate plots
    if results:
        # Group results by prompt length category
        length_categories = ['short', 'medium', 'long']
        category_mapping = {5: 'short', 20: 'medium', 60: 'long'}

        # Calculate averages by length category
        avg_by_length = {}
        for length in length_categories:
            length_results = [r for r in results if category_mapping.get(r['target_words']) == length]
            if length_results:
                avg_by_length[length] = {
                    'actual_words': np.mean([r['actual_words'] for r in length_results]),
                    'total_latency_ms': np.mean([r['total_latency_ms'] for r in length_results]),
                    'ttft_ms': np.mean([r['ttft_ms'] for r in length_results]),
                    'output_words': np.mean([r['output_words'] for r in length_results]),
                    'prompt_tokens': np.mean([r['prompt_tokens'] for r in length_results])
                }

        if avg_by_length:
            lengths = list(avg_by_length.keys())
            word_counts = [avg_by_length[l]['actual_words'] for l in lengths]
            total_latencies = [avg_by_length[l]['total_latency_ms'] for l in lengths]
            ttft_values = [avg_by_length[l]['ttft_ms'] for l in lengths]
            output_words = [avg_by_length[l]['output_words'] for l in lengths]
            prompt_tokens = [avg_by_length[l]['prompt_tokens'] for l in lengths]

            # Plot 1: Prompt Length vs Total Latency
            save_plot(
                word_counts, total_latencies,
                'Prompt Length (words)', 'Total Latency (ms)',
                'FastVLM: Prompt Length vs Total Latency',
                'prompt_length_vs_latency.png'
            )

            # Plot 2: Prompt Length vs TTFT
            save_plot(
                word_counts, ttft_values,
                'Prompt Length (words)', 'Time to First Token (ms)',
                'FastVLM: Prompt Length vs TTFT',
                'prompt_length_vs_ttft.png'
            )

            # Plot 3: Prompt Length vs Output Length
            save_plot(
                word_counts, output_words,
                'Prompt Length (words)', 'Output Length (words)',
                'FastVLM: Prompt Length vs Output Length',
                'prompt_length_vs_output.png'
            )

            # Plot 4: Dual axis - Latency and Output Length
            save_dual_axis_plot(
                word_counts, total_latencies, output_words,
                'Prompt Length (words)', 'Total Latency (ms)', 'Output Length (words)',
                'FastVLM: Prompt Length Effect - Latency & Output',
                'prompt_length_dual_axis.png'
            )

            # Plot 5: Grouped bar chart comparing metrics
            metrics_data = {
                'Total Latency (ms)': total_latencies,
                'TTFT (ms)': ttft_values,
                'Output Words': output_words
            }

            save_grouped_bar(
                metrics_data, lengths,
                'Prompt Length Category', 'Value',
                'FastVLM: Prompt Length Effect - Multiple Metrics',
                'prompt_length_metrics_comparison.png'
            )

        # Additional analysis: Effect by specific prompt categories
        categories = list(set(r['category'] for r in results))
        if len(categories) > 1:
            cat_data = {}
            for cat in categories:
                cat_results = [r for r in results if r['category'] == cat]
                if cat_results:
                    cat_data[cat] = np.mean([r['total_latency_ms'] for r in cat_results])

            if cat_data:
                save_plot(
                    list(range(len(cat_data))), list(cat_data.values()),
                    'Prompt Category', 'Average Total Latency (ms)',
                    'FastVLM: Latency by Prompt Category',
                    'prompt_categories_latency.png'
                )

    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("="*100)
    print(f"{'Category':<15} {'Avg Words':<12} {'Avg Tokens':<12} {'Avg Latency(ms)':<15} "
          f"{'Avg TTFT(ms)':<12} {'Avg Output':<12}")
    print("-"*100)

    # Group by length category for summary
    for target_length in [5, 20, 60]:
        length_results = [r for r in results if r['target_words'] == target_length]
        if length_results:
            category_name = {5: 'Short', 20: 'Medium', 60: 'Long'}[target_length]
            avg_words = np.mean([r['actual_words'] for r in length_results])
            avg_tokens = np.mean([r['prompt_tokens'] for r in length_results])
            avg_latency = np.mean([r['total_latency_ms'] for r in length_results])
            avg_ttft = np.mean([r['ttft_ms'] for r in length_results])
            avg_output = np.mean([r['output_words'] for r in length_results])

            print(f"{category_name:<15} {avg_words:<12.1f} {avg_tokens:<12.1f} "
                  f"{avg_latency:<15.1f} {avg_ttft:<12.1f} {avg_output:<12.1f}")

    # Overall statistics
    print("-"*100)
    overall_avg_latency = np.mean([r['total_latency_ms'] for r in results])
    overall_avg_ttft = np.mean([r['ttft_ms'] for r in results])
    overall_avg_output = np.mean([r['output_words'] for r in results])

    print(f"{'OVERALL':<15} {'':<12} {'':<12} {overall_avg_latency:<15.1f} "
          f"{overall_avg_ttft:<12.1f} {overall_avg_output:<12.1f}")

    # Correlation analysis
    word_counts = [r['actual_words'] for r in results]
    latencies = [r['total_latency_ms'] for r in results]
    correlation = np.corrcoef(word_counts, latencies)[0, 1]

    print(f"\n📈 Correlation between prompt length and latency: {correlation:.3f}")

    print(f"\n✅ Prompt length effect experiment completed!")
    print(f"📁 Results: {csv_path}")
    print(f"📈 Plots: results/plots/")

def main():
    parser = argparse.ArgumentParser(description='FastVLM Prompt Length Effect Experiment')
    parser.add_argument('--model-path', required=True, help='Path to model checkpoint')
    parser.add_argument('--image-folder', default='../images/', help='Folder containing test images')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (keep 1 for VRAM)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path not found: {args.model_path}")
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder not found: {args.image_folder}")

    run_prompt_length_experiment(
        args.model_path, args.image_folder, args.device, args.batch_size
    )

if __name__ == "__main__":
    main()