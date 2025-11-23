#!/usr/bin/env python3
"""
FastVLM Resolution Scaling Experiment (TextVQA Benchmark)
Tests different input resolutions using real samples from TextVQA.
"""

import argparse
import csv
import gc
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


def resize_image(image, target_size):
    """Resize image to target square size with padding."""
    w, h = image.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    new_image = Image.new("RGB", (target_size, target_size), (128, 128, 128))
    new_image.paste(image, ((target_size - new_w) // 2, (target_size - new_h) // 2))
    return new_image


def measure_inference(model, tokenizer, image_processor, image, prompt, device):
    # Prepare input
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

    # Memory and timing
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    start_time = time.perf_counter()

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device, dtype=torch.float16),
            max_new_tokens=50,
            do_sample=False,
            use_cache=True,
        )

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        latency = start_event.elapsed_time(end_event)
        vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
    else:
        latency = (time.perf_counter() - start_time) * 1000
        vram_gb = 0

    return latency, vram_gb


def run_experiment(args):
    print(f"🚀 Loading model: {args.model_path}")
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

    print("📚 Loading TextVQA dataset...")
    dataset = get_benchmark_dataset(
        "textvqa", split="validation", max_samples=args.num_samples
    )

    resolutions = [224, 336, 448, 672, 896, 1024]
    results = []

    for res in resolutions:
        print(f"\nTesting Resolution: {res}x{res}")
        latencies = []
        vrams = []

        for i, sample in enumerate(tqdm(dataset)):
            image = sample["image"]
            # Use the question as prompt
            prompt = (
                sample["question"] if "question" in sample else "Describe this image."
            )

            # Resize
            processed_image = resize_image(image, res)

            try:
                lat, vram = measure_inference(
                    model,
                    tokenizer,
                    image_processor,
                    processed_image,
                    prompt,
                    args.device,
                )
                latencies.append(lat)
                vrams.append(vram)
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

        if latencies:
            avg_lat = np.mean(latencies)
            avg_vram = np.mean(vrams)
            results.append({"resolution": res, "latency": avg_lat, "vram": avg_vram})
            print(f"Avg Latency: {avg_lat:.2f}ms | Avg VRAM: {avg_vram:.2f}GB")

    # Save results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/resolution_scaling_textvqa.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["resolution", "latency", "vram"])
        writer.writeheader()
        writer.writerows(results)

    print(f"💾 Saved results to {csv_path}")

    # Plot
    if results:
        res_vals = [r["resolution"] for r in results]
        lat_vals = [r["latency"] for r in results]
        vram_vals = [r["vram"] for r in results]

        save_dual_axis_plot(
            res_vals,
            lat_vals,
            vram_vals,
            "Resolution (px)",
            "Latency (ms)",
            "VRAM (GB)",
            "Resolution Scaling (TextVQA)",
            "resolution_scaling_textvqa.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-samples", type=int, default=20, help="Samples per resolution"
    )
    args = parser.parse_args()
    run_experiment(args)
