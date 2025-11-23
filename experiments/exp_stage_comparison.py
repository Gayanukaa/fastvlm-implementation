#!/usr/bin/env python3
"""
FastVLM Stage Comparison Experiment (TextVQA Benchmark)
Compares Stage-2 vs Stage-3 models on TextVQA validation set.
Measures OCR Accuracy (Exact Match) and Inference Latency.
"""

import argparse
import os
import sys
import time
import csv
import gc
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import warnings

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates

# Import utilities
from utils_plot import save_grouped_bar
from utils_dataset import get_benchmark_dataset

warnings.filterwarnings("ignore")

def setup_model(model_path, device):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=device, device_map="auto", torch_dtype=torch.float16
    )
    model.eval()
    return model, tokenizer, image_processor

def calculate_exact_match(prediction: str, ground_truths: list) -> float:
    """Checks if prediction matches any of the ground truth answers exactly (case-insensitive)."""
    if not ground_truths:
        return 0.0
    pred = prediction.lower().strip().replace('.', '') # Simple normalization
    for gt in ground_truths:
        if pred == gt.lower().strip():
            return 1.0
    return 0.0

def measure_inference(model, tokenizer, image_processor, image, prompt, device):
    # Prepare input
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

    conv = conv_templates['qwen_2'].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Timing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
    
    start_time = time.perf_counter()
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device, dtype=torch.float16),
            max_new_tokens=20, # Short answers for VQA
            do_sample=False,
            use_cache=True
        )
    
    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event)
    else:
        latency = (time.perf_counter() - start_time) * 1000

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return latency, generated_text

def run_experiment(args):
    print(f"🚀 Starting Stage Comparison (TextVQA)")
    print(f"Stage 2: {args.stage2_path}")
    print(f"Stage 3: {args.stage3_path}")

    # Load Dataset
    dataset = get_benchmark_dataset("textvqa", split="validation", max_samples=args.num_samples)
    
    results = []
    
    # Define stages to test
    stages = [
        ("Stage 2", args.stage2_path),
        ("Stage 3", args.stage3_path)
    ]

    for stage_name, model_path in stages:
        print(f"\n🔄 Testing {stage_name}...")
        
        # Load model (one at a time to save VRAM)
        model, tokenizer, image_processor = setup_model(model_path, args.device)
        
        stage_latencies = []
        stage_accuracies = []
        
        for i, sample in enumerate(tqdm(dataset)):
            image = sample['image']
            prompt = sample['question'] if 'question' in sample else "What is in this image?"
            ground_truths = sample.get('answers', [])
            
            try:
                latency, pred_text = measure_inference(model, tokenizer, image_processor, image, prompt, args.device)
                
                # Calculate Accuracy
                acc = calculate_exact_match(pred_text, ground_truths)
                
                stage_latencies.append(latency)
                stage_accuracies.append(acc)
                
                results.append({
                    'stage': stage_name,
                    'id': sample['id'],
                    'latency': latency,
                    'accuracy': acc,
                    'prediction': pred_text,
                    'ground_truths': str(ground_truths)
                })
                
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue
        
        avg_lat = np.mean(stage_latencies) if stage_latencies else 0
        avg_acc = np.mean(stage_accuracies) if stage_accuracies else 0
        print(f"✅ {stage_name} Results: Avg Latency={avg_lat:.2f}ms, Avg Accuracy={avg_acc:.2%}")
        
        # Cleanup
        del model, tokenizer, image_processor
        gc.collect()
        torch.cuda.empty_cache()

    # Save Results
    os.makedirs('results', exist_ok=True)
    csv_path = 'results/stage_comparison_textvqa.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = ['stage', 'id', 'latency', 'accuracy', 'prediction', 'ground_truths']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved results to {csv_path}")

    # Plotting
    if results:
        # Prepare data for plotting
        stage2_res = [r for r in results if r['stage'] == 'Stage 2']
        stage3_res = [r for r in results if r['stage'] == 'Stage 3']
        
        metrics = {
            'Latency (ms)': [np.mean([r['latency'] for r in stage2_res]), np.mean([r['latency'] for r in stage3_res])],
            'Accuracy (Exact Match)': [np.mean([r['accuracy'] for r in stage2_res]), np.mean([r['accuracy'] for r in stage3_res])]
        }
        
        # Plot Latency
        save_grouped_bar(
            {'Stage 2': [metrics['Latency (ms)'][0]], 'Stage 3': [metrics['Latency (ms)'][1]]},
            ['Average'], "Stage", "Latency (ms)", "Stage Comparison - Latency", "stage_comparison_latency.pdf"
        )
        
        # Plot Accuracy
        save_grouped_bar(
            {'Stage 2': [metrics['Accuracy (Exact Match)'][0]], 'Stage 3': [metrics['Accuracy (Exact Match)'][1]]},
            ['Average'], "Stage", "Accuracy", "Stage Comparison - Accuracy", "stage_comparison_accuracy.pdf"
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage2-path", required=True)
    parser.add_argument("--stage3-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    run_experiment(args)
