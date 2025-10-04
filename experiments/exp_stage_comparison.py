#!/usr/bin/env python3
"""
FastVLM Stage Comparison Experiment
Compares Stage-2 vs Stage-3 models on various prompts and images.
"""

import argparse
import os
import time
import csv
import gc
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
from utils_plot import save_grouped_bar, save_plot

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

def measure_inference(model, tokenizer, image_processor, image: Image.Image,
                     prompt: str, device: str, max_tokens: int = 100) -> Dict:
    """Measure inference metrics."""

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

    # Count tokens
    input_tokens = len(input_ids[0])
    output_tokens = len(outputs[0]) - input_tokens

    return {
        'ttft_ms': cuda_ttft,
        'total_latency_ms': cuda_total,
        'input_tokens': input_tokens,
        'output_tokens': output_tokens,
        'generated_text': generated_text
    }

def get_test_prompts() -> List[Tuple[str, str]]:
    """Get test prompts for different tasks."""
    return [
        ("caption", "Describe this image in detail."),
        ("reasoning", "What is happening in this image and why might it be significant?"),
        ("ocr", "What text can you see in this image? Transcribe any visible text.")
    ]

def run_stage_comparison_experiment(stage2_path: str, stage3_path: str, image_folder: str,
                                  device: str, batch_size: int = 1) -> None:
    """Run stage comparison experiment."""

    print("🚀 Starting Stage Comparison Experiment")
    print(f"Stage 2: {stage2_path}")
    print(f"Stage 3: {stage3_path}")
    print(f"Device: {device}")

    # Find test images (limit to 5 as requested)
    image_files = [f for f in os.listdir(image_folder)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:5]

    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")

    print(f"📷 Using {len(image_files)} test images")

    # Get test prompts
    test_prompts = get_test_prompts()
    print(f"📝 Using {len(test_prompts)} prompts: {[p[0] for p in test_prompts]}")

    # Results storage
    results = []

    # Test both stages
    stages = [
        ("stage2", stage2_path),
        ("stage3", stage3_path)
    ]

    stage_outputs = {}  # Store outputs for BLEU comparison

    for stage_name, model_path in stages:
        print(f"\n🔄 Testing {stage_name}: {model_path}")

        # Load model
        model, tokenizer, image_processor = setup_model_and_tokenizer(model_path, device)
        stage_outputs[stage_name] = {}

        # Test on all image-prompt combinations
        for img_idx, image_file in enumerate(tqdm(image_files, desc=f"Testing {stage_name}")):
            image_path = os.path.join(image_folder, image_file)
            image = preprocess_image(image_path)

            stage_outputs[stage_name][image_file] = {}

            for prompt_type, prompt in test_prompts:
                print(f"  📷 {image_file} + 📝 {prompt_type}")

                # Run inference multiple times for averaging
                run_results = []

                for run in range(2):  # 2 runs per combination to save time
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
                        'stage': stage_name,
                        'image': image_file,
                        'prompt_type': prompt_type,
                        'prompt': prompt,
                        'ttft_ms': np.mean([r['ttft_ms'] for r in run_results]),
                        'total_latency_ms': np.mean([r['total_latency_ms'] for r in run_results]),
                        'input_tokens': int(np.mean([r['input_tokens'] for r in run_results])),
                        'output_tokens': int(np.mean([r['output_tokens'] for r in run_results])),
                        'generated_text': run_results[0]['generated_text']  # Use first run's text
                    }
                    results.append(avg_result)

                    # Store output for BLEU calculation
                    stage_outputs[stage_name][image_file][prompt_type] = avg_result['generated_text']

                    print(f"    ✅ TTFT: {avg_result['ttft_ms']:.1f}ms")

        # Clean up model before loading next stage
        del model, tokenizer, image_processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f"✅ {stage_name} testing completed")

    # Calculate BLEU scores between stages
    print("\n📊 Calculating BLEU scores between stages...")

    for result in results:
        if result['stage'] == 'stage2':
            # Find corresponding stage3 result
            stage3_text = stage_outputs['stage3'].get(result['image'], {}).get(result['prompt_type'], '')
            stage2_text = result['generated_text']

            if stage3_text and stage2_text:
                bleu_score = calculate_bleu_score(stage3_text, stage2_text)
                result['bleu_vs_stage3'] = bleu_score
            else:
                result['bleu_vs_stage3'] = 0.0
        else:
            result['bleu_vs_stage3'] = 1.0  # Stage3 vs itself

    # Save results to CSV
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/stage_compare.csv'

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['stage', 'image', 'prompt_type', 'prompt', 'ttft_ms', 'total_latency_ms',
                     'input_tokens', 'output_tokens', 'bleu_vs_stage3', 'generated_text']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Results saved to {csv_path}")

    # Generate plots
    if results:
        # Calculate averages by stage and prompt type
        stage2_results = [r for r in results if r['stage'] == 'stage2']
        stage3_results = [r for r in results if r['stage'] == 'stage3']

        # Average TTFT by stage
        stage2_ttft = np.mean([r['ttft_ms'] for r in stage2_results])
        stage3_ttft = np.mean([r['ttft_ms'] for r in stage3_results])

        # Average BLEU by stage
        stage2_bleu = np.mean([r['bleu_vs_stage3'] for r in stage2_results])
        stage3_bleu = np.mean([r['bleu_vs_stage3'] for r in stage3_results])

        # Plot 1: Average TTFT comparison
        ttft_data = {
            'Stage 2': [stage2_ttft],
            'Stage 3': [stage3_ttft]
        }

        save_grouped_bar(
            ttft_data, ['Average'],
            'Stage', 'Average TTFT (ms)',
            'FastVLM: Stage Comparison - Average TTFT',
            'stage_comparison_ttft.png'
        )

        # Plot 2: BLEU comparison
        bleu_data = {
            'Stage 2 vs Stage 3': [stage2_bleu],
            'Stage 3 vs Stage 3': [stage3_bleu]
        }

        save_grouped_bar(
            bleu_data, ['BLEU Score'],
            'Comparison', 'BLEU Score',
            'FastVLM: Stage Comparison - BLEU Similarity',
            'stage_comparison_bleu.png'
        )

        # Plot 3: TTFT by prompt type
        prompt_types = list(set(r['prompt_type'] for r in results))
        stage2_by_prompt = {}
        stage3_by_prompt = {}

        for pt in prompt_types:
            stage2_by_prompt[pt] = np.mean([r['ttft_ms'] for r in stage2_results if r['prompt_type'] == pt])
            stage3_by_prompt[pt] = np.mean([r['ttft_ms'] for r in stage3_results if r['prompt_type'] == pt])

        prompt_ttft_data = {
            'Stage 2': [stage2_by_prompt[pt] for pt in prompt_types],
            'Stage 3': [stage3_by_prompt[pt] for pt in prompt_types]
        }

        save_grouped_bar(
            prompt_ttft_data, prompt_types,
            'Prompt Type', 'Average TTFT (ms)',
            'FastVLM: TTFT by Prompt Type and Stage',
            'stage_comparison_by_prompt.png'
        )

    # Print summary
    print("\n📊 EXPERIMENT SUMMARY")
    print("="*80)
    print(f"{'Stage':<8} {'Prompt':<12} {'Avg TTFT(ms)':<15} {'Avg BLEU':<12} {'Images':<8}")
    print("-"*80)

    for stage_name in ['stage2', 'stage3']:
        stage_results = [r for r in results if r['stage'] == stage_name]

        for prompt_type in ['caption', 'reasoning', 'ocr']:
            prompt_results = [r for r in stage_results if r['prompt_type'] == prompt_type]

            if prompt_results:
                avg_ttft = np.mean([r['ttft_ms'] for r in prompt_results])
                avg_bleu = np.mean([r['bleu_vs_stage3'] for r in prompt_results])
                num_images = len(set(r['image'] for r in prompt_results))

                print(f"{stage_name:<8} {prompt_type:<12} {avg_ttft:<15.1f} "
                      f"{avg_bleu:<12.3f} {num_images:<8}")

    # Overall averages
    print("-"*80)
    stage2_avg_ttft = np.mean([r['ttft_ms'] for r in results if r['stage'] == 'stage2'])
    stage3_avg_ttft = np.mean([r['ttft_ms'] for r in results if r['stage'] == 'stage3'])
    stage2_avg_bleu = np.mean([r['bleu_vs_stage3'] for r in results if r['stage'] == 'stage2'])

    print(f"{'stage2':<8} {'OVERALL':<12} {stage2_avg_ttft:<15.1f} {stage2_avg_bleu:<12.3f} {len(image_files):<8}")
    print(f"{'stage3':<8} {'OVERALL':<12} {stage3_avg_ttft:<15.1f} {'1.000':<12} {len(image_files):<8}")

    print(f"\n✅ Stage comparison experiment completed!")
    print(f"📁 Results: {csv_path}")
    print(f"📈 Plots: results/plots/")
    print(f"🎯 Stage 2 is {stage2_avg_ttft/stage3_avg_ttft:.2f}x {'faster' if stage2_avg_ttft < stage3_avg_ttft else 'slower'} than Stage 3")
    print(f"📝 Stage 2 achieves {stage2_avg_bleu:.3f} BLEU similarity to Stage 3")

def main():
    parser = argparse.ArgumentParser(description='FastVLM Stage Comparison Experiment')
    parser.add_argument('--stage2-path', required=True, help='Path to Stage 2 model checkpoint')
    parser.add_argument('--stage3-path', required=True, help='Path to Stage 3 model checkpoint')
    parser.add_argument('--image-folder', default='../images', help='Folder containing test images')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size (keep 1 for VRAM)')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.stage2_path):
        raise ValueError(f"Stage 2 model path not found: {args.stage2_path}")
    if not os.path.exists(args.stage3_path):
        raise ValueError(f"Stage 3 model path not found: {args.stage3_path}")
    if not os.path.exists(args.image_folder):
        raise ValueError(f"Image folder not found: {args.image_folder}")

    run_stage_comparison_experiment(
        args.stage2_path, args.stage3_path, args.image_folder, args.device, args.batch_size
    )

if __name__ == "__main__":
    main()