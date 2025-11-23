#!/usr/bin/env python3
"""
FastVLM Prompt Length Effect Experiment (TextVQA Benchmark)
Tests how different prompt lengths affect inference latency and TTFT.
Uses TextVQA questions extended with context prefixes.
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


def get_prompt_prefixes():
    return {
        "Short": "",
        "Medium": "Please analyze this image carefully and answer the following question based on the visual details you observe: ",
        "Long": "As an expert visual assistant capable of detailed image analysis and OCR, please examine this image thoroughly. Pay attention to all text, objects, and context within the scene. Based on your comprehensive understanding of the visual content, please provide a precise answer to the following specific question: ",
    }


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
        first_token_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    start_time = time.perf_counter()

    with torch.no_grad():
        # Generate first token for TTFT
        _ = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(device, dtype=torch.float16),
            max_new_tokens=1,
            do_sample=False,
            use_cache=True,
        )
        if torch.cuda.is_available():
            first_token_event.record()

        first_token_time = time.perf_counter()

        # Generate full response
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
        ttft = start_event.elapsed_time(first_token_event)
        latency = start_event.elapsed_time(end_event)
    else:
        ttft = (first_token_time - start_time) * 1000
        latency = (time.perf_counter() - start_time) * 1000

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    prompt_tokens = input_ids.shape[1]

    return latency, ttft, prompt_tokens, generated_text


def run_experiment(args):
    print(f"🚀 Starting Prompt Length Effect (TextVQA)")
    print(f"Model: {args.model_path}")

    model, tokenizer, image_processor = setup_model(args.model_path, args.device)
    dataset = get_benchmark_dataset(
        "textvqa", split="validation", max_samples=args.num_samples
    )

    prefixes = get_prompt_prefixes()
    results = []

    for category, prefix in prefixes.items():
        print(f"\nTesting Prompt Length: {category}")
        latencies = []
        ttfts = []
        token_counts = []

        for i, sample in enumerate(tqdm(dataset)):
            image = sample["image"]
            base_question = (
                sample["question"] if "question" in sample else "Describe this image."
            )
            prompt = prefix + base_question

            try:
                latency, ttft, tokens, _ = measure_inference(
                    model, tokenizer, image_processor, image, prompt, args.device
                )

                latencies.append(latency)
                ttfts.append(ttft)
                token_counts.append(tokens)
            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

        if latencies:
            avg_lat = np.mean(latencies)
            avg_ttft = np.mean(ttfts)
            avg_tokens = np.mean(token_counts)

            results.append(
                {
                    "category": category,
                    "avg_tokens": avg_tokens,
                    "latency": avg_lat,
                    "ttft": avg_ttft,
                }
            )
            print(
                f"Avg Tokens: {avg_tokens:.1f} | Avg Latency: {avg_lat:.2f}ms | Avg TTFT: {avg_ttft:.2f}ms"
            )

    # Save Results
    os.makedirs("results", exist_ok=True)
    csv_path = "results/prompt_length_textvqa.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["category", "avg_tokens", "latency", "ttft"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved results to {csv_path}")

    # Plot
    if results:
        tokens_val = [r["avg_tokens"] for r in results]
        lat_val = [r["latency"] for r in results]
        ttft_val = [r["ttft"] for r in results]

        save_dual_axis_plot(
            tokens_val,
            lat_val,
            ttft_val,
            "Prompt Tokens",
            "Total Latency (ms)",
            "TTFT (ms)",
            "Prompt Length Effect (TextVQA)",
            "prompt_length_effect.pdf",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()
    run_experiment(args)
