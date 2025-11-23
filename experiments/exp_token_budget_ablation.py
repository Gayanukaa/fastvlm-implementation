#!/usr/bin/env python3
"""
FastVLM Token Budget Ablation Experiment (TextVQA Benchmark)
Tests different token budgets (via resolution scaling) and measures Accuracy vs Latency.
"""

import argparse
import csv
import gc
import math
import os
import sys
import time
import warnings

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_dataset import get_benchmark_dataset

# Import utilities
from utils_plot import save_dual_axis_plot, save_plot

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

warnings.filterwarnings("ignore")


def setup_model(model_path, device):
    disable_torch_init()
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path,
        None,
        model_name,
        device=device,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer, image_processor


def calculate_exact_match(prediction: str, ground_truths: list) -> float:
    if not ground_truths:
        return 0.0
    pred = prediction.lower().strip().replace(".", "")
    for gt in ground_truths:
        if pred == gt.lower().strip():
            return 1.0
    return 0.0


def calculate_token_budget_size(target_tokens: int) -> int:
    """Calculate image size based on target token budget (assuming patch size 16)."""
    patch_size = 16
    total_pixels = target_tokens * (patch_size**2)
    image_size = int(math.sqrt(total_pixels))
    # Round to nearest multiple of 14 (ViT patch size usually 14 or 16, LLaVA uses 336 which is 14*24)
    # But here we just want rough scaling.
    return image_size


def resize_image_for_budget(image, target_tokens):
    target_size = calculate_token_budget_size(target_tokens)
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_image = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    new_image.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return new_image, target_size


def measure_inference(model, tokenizer, image_processor, image, prompt, device):
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

    conv = conv_templates["qwen_2"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_formatted = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .to(device)
    )
    image_tensor = process_images([image], image_processor, model.config)[0]

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
            max_new_tokens=20,
            do_sample=False,
            use_cache=True,
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
    print(f"🚀 Starting Token Budget Ablation (TextVQA)")
    print(f"Model: {args.model_path}")

    model, tokenizer, image_processor = setup_model(args.model_path, args.device)

    print("🔥 Warming up GPU...")
    # Create a dummy image and prompt
    dummy_image = Image.new('RGB', (512, 512), color='white')
    dummy_prompt = "Warmup run"

    # Run inference once to initialize CUDA context and buffers
    try:
        _ = measure_inference(model, tokenizer, image_processor, dummy_image, dummy_prompt, args.device)
    except Exception as e:
        print(f"Warmup warning: {e}")

    # Reset stats so warmup doesn't count toward VRAM peaks
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("✅ Warmup complete. Starting experiment...")

    dataset = get_benchmark_dataset(
        "textvqa", split="validation", max_samples=args.num_samples
    )

    # Target token budgets (approximate)
    budgets = [16, 64, 144, 256, 576]
    results = []

    for budget in budgets:
        print(f"\nTesting Token Budget: ~{budget} tokens")
        latencies = []
        accuracies = []

        for i, sample in enumerate(tqdm(dataset)):
            image = sample["image"]
            prompt = (
                sample["question"] if "question" in sample else "Describe this image."
            )
            ground_truths = sample.get("answers", [])

            # Resize based on budget
            processed_image, actual_size = resize_image_for_budget(image, budget)

            try:
                latency, pred_text = measure_inference(
                    model,
                    tokenizer,
                    image_processor,
                    processed_image,
                    prompt,
                    args.device,
                )
                acc = calculate_exact_match(pred_text, ground_truths)

                latencies.append(latency)
                accuracies.append(acc)
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

        if latencies:
            avg_lat = np.mean(latencies)
            avg_acc = np.mean(accuracies)
            results.append({"budget": budget, "latency": avg_lat, "accuracy": avg_acc})
            print(f"Avg Latency: {avg_lat:.2f}ms | Avg Accuracy: {avg_acc:.2%}")

    # Save Results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/token_budget_textvqa.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["budget", "latency", "accuracy"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved results to {csv_path}")

    # Plot
    if results:
        budgets_val = [r["budget"] for r in results]
        lat_val = [r["latency"] for r in results]
        acc_val = [r["accuracy"] for r in results]

        save_dual_axis_plot(
            budgets_val,
            lat_val,
            acc_val,
            "Token Budget",
            "Latency (ms)",
            "Accuracy",
            "Token Budget Ablation (TextVQA)",
            "token_budget_ablation.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    run_experiment(args)
