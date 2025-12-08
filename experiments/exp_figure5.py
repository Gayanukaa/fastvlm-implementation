#!/usr/bin/env python3
"""
FastVLM Figure 5 Replication: Resolution Scaling Experiment
Measures Vision Encoder Latency and LLM Prefilling Latency separately
(excluding token generation/decoding time).

Reference: Figure 5 from FastVLM paper - "Vision Latency vs LLM Prefilling"
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
from utils_plot import save_grouped_bar

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


def measure_vision_and_prefill_latency(
    model, tokenizer, image_processor, image, prompt, device
):
    """
    Measure Vision Encoder latency and LLM Prefilling latency separately.
    Does NOT include token generation/decoding time (unlike model.generate()).

    IMPORTANT: CPU preprocessing and data transfer are done OUTSIDE the timer.
    Only pure GPU forward pass time is measured.

    Returns:
        vision_latency: Time for vision encoder + projector (ms) - GPU only
        prefill_latency: Time for LLM prefilling pass (ms) - GPU only
    """
    # --- 1. PREPARATION (Do not time this) ---
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

    # Prepare Image Tensor (CPU -> GPU) OUTSIDE the timer
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(device, dtype=torch.float16)

    # --- 2. MEASURE VISION ENCODER (GPU ONLY) ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_vision = time.perf_counter()

    with torch.no_grad():
        image_features = model.encode_images(image_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    vision_latency = (time.perf_counter() - start_vision) * 1000

    # --- 3. MEASURE LLM PREFILLING (GPU ONLY) ---
    # Prepare embeddings (CPU/GPU mix, do not time)
    with torch.no_grad():
        prepared = model.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=None,
            past_key_values=None,
            labels=None,
            images=image_tensor,
        )
        inputs_embeds = prepared[4] if len(prepared) > 4 else prepared[0]
        attention_mask = prepared[2] if len(prepared) > 2 else None

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_prefill = time.perf_counter()

    with torch.no_grad():
        _ = model.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    prefill_latency = (time.perf_counter() - start_prefill) * 1000

    return vision_latency, prefill_latency


def run_experiment(args):
    print(f"🚀 Loading model: {args.model_path}")
    model, tokenizer, image_processor = setup_model(args.model_path, args.device)

    # Print original image processor settings
    print(f"   Original image_processor settings:")
    if hasattr(image_processor, "crop_size"):
        print(f"      crop_size: {image_processor.crop_size}")
    if hasattr(image_processor, "size"):
        print(f"      size: {image_processor.size}")

    print("🔥 Warming up GPU...")
    # Create a dummy image and prompt
    dummy_image = Image.new("RGB", (512, 512), color="white")
    dummy_prompt = "Warmup run"

    # Run inference once to initialize CUDA context and buffers
    try:
        _ = measure_vision_and_prefill_latency(
            model, tokenizer, image_processor, dummy_image, dummy_prompt, args.device
        )
    except Exception as e:
        print(f"Warmup warning: {e}")

    # Reset stats so warmup doesn't count
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()

    print("✅ Warmup complete. Starting experiment...")

    print("📚 Loading TextVQA dataset...")
    dataset = get_benchmark_dataset(
        "textvqa", split="validation", max_samples=args.num_samples
    )

    # Convert to list for multiple passes
    samples = list(dataset)
    print(f"   ➜ Loaded {len(samples)} samples")

    # Figure 5 resolutions from the paper
    resolutions = [256, 512, 768, 1024, 1536]
    results = []

    for res in resolutions:
        print(f"\n{'='*60}")
        print(f"  Testing Resolution: {res}x{res}")
        print(f"{'='*60}")

        # FORCE the image processor to respect the current resolution
        # By default, it may resize all images to a fixed size (e.g., 336px)
        if hasattr(image_processor, "crop_size"):
            image_processor.crop_size["height"] = res
            image_processor.crop_size["width"] = res
        if hasattr(image_processor, "size"):
            # Handle different processor configurations
            if isinstance(image_processor.size, dict):
                if "shortest_edge" in image_processor.size:
                    image_processor.size["shortest_edge"] = res
                if "height" in image_processor.size:
                    image_processor.size["height"] = res
                if "width" in image_processor.size:
                    image_processor.size["width"] = res
            else:
                image_processor.size = res

        vision_latencies = []
        prefill_latencies = []
        first_sample = True

        for sample in tqdm(samples, desc=f"Eval @{res}px"):
            image = sample["image"]
            prompt = sample.get("question", "Describe this image.")

            # Resize image to target resolution
            processed_image = resize_image(image, res)

            # Debug: Show tensor shape for first sample at each resolution
            if first_sample:
                debug_tensor = process_images(
                    [processed_image], image_processor, model.config
                )
                print(f"   Tensor shape @ {res}px: {debug_tensor.shape}")
                first_sample = False

            try:
                vis_lat, prefill_lat = measure_vision_and_prefill_latency(
                    model,
                    tokenizer,
                    image_processor,
                    processed_image,
                    prompt,
                    args.device,
                )
                vision_latencies.append(vis_lat)
                prefill_latencies.append(prefill_lat)
            except Exception as e:
                print(f"Error: {e}")
                continue

        if vision_latencies:
            avg_vision = np.mean(vision_latencies)
            avg_prefill = np.mean(prefill_latencies)
            results.append(
                {
                    "resolution": res,
                    "vision_latency": avg_vision,
                    "llm_prefill_latency": avg_prefill,
                    "total_latency": avg_vision + avg_prefill,
                }
            )
            print(
                f"   Vision: {avg_vision:.2f}ms | Prefill: {avg_prefill:.2f}ms | Total: {avg_vision + avg_prefill:.2f}ms"
            )

    # Save results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/resolution_scaling_textvqa.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "resolution",
                "vision_latency",
                "llm_prefill_latency",
                "total_latency",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\n💾 Saved results to {csv_path}")

    # Print summary table
    print("\n" + "=" * 70)
    print("  Figure 5 Replication — Vision Latency vs LLM Prefilling")
    print("=" * 70)
    print(
        f"{'Resolution':<12} {'Vision (ms)':<15} {'Prefill (ms)':<15} {'Total (ms)':<12}"
    )
    print("-" * 70)
    for r in results:
        print(
            f"{r['resolution']:<12} {r['vision_latency']:<15.2f} {r['llm_prefill_latency']:<15.2f} {r['total_latency']:<12.2f}"
        )
    print("=" * 70)

    # Create grouped bar chart (Figure 5 style)
    if results:
        res_labels = [str(r["resolution"]) for r in results]

        data = {
            "Vision Latency": [r["vision_latency"] for r in results],
            "LLM Prefilling": [r["llm_prefill_latency"] for r in results],
        }

        save_grouped_bar(
            data=data,
            labels=res_labels,
            xlabel="Resolution (px)",
            ylabel="Latency (ms)",
            title="Figure 5: Vision Latency vs LLM Prefilling",
            filename="figure5_resolution_scaling.png",
        )

    print("\n📋 See experiments/limitations.md for replication limitations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastVLM Figure 5 Replication: Vision vs LLM Prefilling Latency"
    )
    parser.add_argument(
        "--model-path",
        default="../checkpoints/llava-fastvithd_0.5b_stage3",
        help="Path to FastVLM checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Samples per resolution (default: full dataset)",
    )
    args = parser.parse_args()
    run_experiment(args)
