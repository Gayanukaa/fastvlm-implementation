#!/usr/bin/env python3
"""
FastVLM Multi-Benchmark Evaluation Script

Supports replication of:
- Figure 4: Pareto curve (Avg-5 Score vs TTFT at different resolutions)
- Table 6: Full VLM benchmark comparison (Avg-5 metrics)
- Table 11: Text-Rich benchmark evaluation
- Stage Comparison: Stage-2 vs Stage-3 comparison on TextVQA

Usage Examples:
  # Figure 4 / Table 6 (Avg-5 benchmarks)
  python exp_stage_comparison.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --replication-target figure4 --resolution 1024 --num-samples 100

  # Table 11 (Text-Rich benchmarks)
  python exp_stage_comparison.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --replication-target table11 --resolution 1024

  # Stage Comparison (original mode)
  python exp_stage_comparison.py --replication-target stage-comparison --stage2-path ../checkpoints/llava-fastvithd_0.5b_stage2 --stage3-path ../checkpoints/llava-fastvithd_0.5b_stage3 --num-samples 20
"""

import argparse
import csv
import gc
import os
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_dataset import get_benchmark_dataset

# Import utilities
from utils_plot import save_grouped_bar, ensure_plot_dir

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

# ============================================================
# Benchmark Groups from the Paper
# ============================================================

# Benchmarks used in Figure 4 and Table 6 for "Avg-5"
# Note: Full Avg-5 includes TextVQA, DocVQA, ChartQA, AI2D, InfoVQA
# We support TextVQA and DocVQA (available via lmms-lab)
AVG_5_BENCHMARKS = ["textvqa", "docvqa"]

# Benchmarks used in Table 11 (Text-Rich)
# Note: Full set includes TextVQA, DocVQA, ChartQA, InfoVQA, OCRBench
# We support TextVQA and DocVQA
TABLE_11_BENCHMARKS = ["textvqa", "docvqa"]

# Resolutions used in Figure 4 for the Pareto curve
FIGURE4_RESOLUTIONS = [256, 512, 768, 1024]

# Standard metric map (for reference)
METRIC_MAP = {
    "textvqa": "accuracy",  # Paper uses accuracy/exact match
    "docvqa": "anls",       # Paper uses ANLS, but exact match is a proxy
}


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


def save_pareto_curve(results, title, filename):
    """
    Plots Accuracy (Y) vs Latency (X) for multiple resolutions.
    This replicates the Figure 4 style from the FastVLM paper.
    """
    plt.figure(figsize=(10, 6))

    latencies = [r['avg_latency'] for r in results]
    accuracies = [r['avg_accuracy'] for r in results]
    resolutions = [r['resolution'] for r in results]

    # Plot line with markers
    plt.plot(latencies, accuracies, marker='o', linestyle='-', color='#1f77b4',
             linewidth=2, markersize=10, label='FastViT-HD (Ours)')

    # Annotate points with resolution
    for i, res in enumerate(resolutions):
        plt.annotate(f"{res}²",
                     (latencies[i], accuracies[i]),
                     xytext=(8, 8), textcoords='offset points',
                     fontsize=11, fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

    plt.xscale('log')  # Figure 4 uses log scale for X-axis
    plt.xlabel("Time To First Token (ms) [Log Scale]", fontsize=12)
    plt.ylabel("Avg-N VLM Accuracy (%)", fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=11)

    # Set axis limits with some padding
    if latencies:
        plt.xlim(min(latencies) * 0.8, max(latencies) * 1.2)
    if accuracies:
        y_min = max(0, min(accuracies) - 5)
        y_max = min(100, max(accuracies) + 5)
        plt.ylim(y_min, y_max)

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   📈 Saved Pareto curve to {full_path}")


def calculate_exact_match(prediction: str, ground_truths: list) -> float:
    """Checks if prediction matches any of the ground truth answers exactly (case-insensitive)."""
    if not ground_truths:
        return 0.0
    pred = prediction.lower().strip().replace(".", "")  # Simple normalization
    for gt in ground_truths:
        if pred == gt.lower().strip():
            return 1.0
    return 0.0


def measure_inference(model, tokenizer, image_processor, image, prompt, device, target_resolution=None):
    """
    Run inference and measure latency.

    Args:
        target_resolution: If provided, forces the image processor to use this resolution.
                          Critical for replicating Figure 4 and Table 11.
    """
    # --- Resolution Override (Critical for Figure 4 / Table 11) ---
    if target_resolution:
        if hasattr(image_processor, 'crop_size'):
            image_processor.crop_size['height'] = target_resolution
            image_processor.crop_size['width'] = target_resolution
        if hasattr(image_processor, 'size'):
            if isinstance(image_processor.size, dict):
                if 'shortest_edge' in image_processor.size:
                    image_processor.size['shortest_edge'] = target_resolution
                if 'height' in image_processor.size:
                    image_processor.size['height'] = target_resolution
                if 'width' in image_processor.size:
                    image_processor.size['width'] = target_resolution
            else:
                image_processor.size = target_resolution
    # ---------------------------------------------------------------

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
            max_new_tokens=20,  # Short answers for VQA
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


def run_multi_benchmark_experiment(args):
    """
    Run evaluation across multiple benchmarks for Figure 4, Table 6, or Table 11 replication.

    For Figure 4: Loops through multiple resolutions to generate a Pareto curve.
    For Table 6/11: Runs at a single resolution.
    """
    print(f"🚀 Replicating: {args.replication_target}")
    print(f"   Model: {args.model_path}")
    print(f"   Samples per benchmark: {args.num_samples if args.num_samples else 'Full dataset'}")

    # 1. Determine Resolution Loop and Benchmarks
    if args.replication_target == "figure4":
        # Figure 4: Loop through multiple resolutions for Pareto curve
        test_resolutions = FIGURE4_RESOLUTIONS
        benchmarks = AVG_5_BENCHMARKS
        print(f"   Mode: Pareto Curve Generation")
        print(f"   Resolutions: {test_resolutions}")
        print(f"   Benchmarks (Avg-5): {benchmarks}")
    elif args.replication_target == "table6":
        # Table 6: Single resolution
        test_resolutions = [args.resolution]
        benchmarks = AVG_5_BENCHMARKS
        print(f"   Resolution: {args.resolution}px")
        print(f"   Benchmarks (Avg-5): {benchmarks}")
    elif args.replication_target == "table11":
        # Table 11: Single resolution
        test_resolutions = [args.resolution]
        benchmarks = TABLE_11_BENCHMARKS
        print(f"   Resolution: {args.resolution}px")
        print(f"   Benchmarks (Text-Rich): {benchmarks}")
    else:
        test_resolutions = [args.resolution]
        benchmarks = ["textvqa"]
        print(f"   Resolution: {args.resolution}px")
        print(f"   Benchmarks: {benchmarks}")

    # 2. Setup Model (Load once to save VRAM)
    print(f"\n🔹 Loading model...")
    model, tokenizer, image_processor = setup_model(args.model_path, args.device)

    # Initial warmup
    print("🔥 Warming up GPU...")
    dummy_image = Image.new('RGB', (512, 512), color='white')
    try:
        _ = measure_inference(model, tokenizer, image_processor, dummy_image, "Warmup", args.device, 512)
    except Exception as e:
        print(f"Warmup warning: {e}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print("✅ Warmup complete.")

    # Pre-load datasets once (to avoid re-downloading for each resolution)
    print("\n📚 Pre-loading datasets...")
    all_datasets = {}
    for dataset_name in benchmarks:
        try:
            dataset = get_benchmark_dataset(
                dataset_name,
                split="validation",
                max_samples=args.num_samples
            )
            all_datasets[dataset_name] = list(dataset)
            print(f"   ✅ {dataset_name}: {len(all_datasets[dataset_name])} samples")
        except Exception as e:
            print(f"   ⚠️ Skipping {dataset_name}: {e}")

    # Store aggregated results for Pareto curve (Figure 4)
    pareto_results = []
    all_detailed_results = []

    # --- LOOP OVER RESOLUTIONS ---
    for res in test_resolutions:
        print(f"\n{'='*70}")
        print(f"  Testing Resolution: {res}x{res}")
        print(f"{'='*70}")

        # Warmup for this specific resolution
        try:
            dummy = Image.new('RGB', (224, 224), color='white')
            _ = measure_inference(model, tokenizer, image_processor, dummy, "warmup", args.device, target_resolution=res)
        except:
            pass

        res_metrics = {"accuracies": [], "latencies": []}
        res_benchmark_results = {}

        # Iterate over benchmarks (e.g., TextVQA + DocVQA)
        for dataset_name in benchmarks:
            if dataset_name not in all_datasets:
                continue

            samples = all_datasets[dataset_name]
            print(f"   📊 Evaluating {dataset_name}...", end=" ", flush=True)

            benchmark_latencies = []
            benchmark_accuracies = []

            for sample in samples:
                image = sample["image"]
                prompt = sample.get("question", "Describe this image.")
                ground_truths = sample.get("answers", [])

                try:
                    lat, pred = measure_inference(
                        model, tokenizer, image_processor, image, prompt, args.device,
                        target_resolution=res  # Critical: Force Resolution
                    )
                    acc = calculate_exact_match(pred, ground_truths)

                    res_metrics["accuracies"].append(acc)
                    res_metrics["latencies"].append(lat)
                    benchmark_latencies.append(lat)
                    benchmark_accuracies.append(acc)

                    all_detailed_results.append({
                        "resolution": res,
                        "benchmark": dataset_name,
                        "id": sample.get("id", ""),
                        "latency": lat,
                        "accuracy": acc,
                        "prediction": pred,
                        "ground_truths": str(ground_truths),
                    })

                except Exception as e:
                    continue

            if benchmark_accuracies:
                avg_acc = np.mean(benchmark_accuracies) * 100
                avg_lat = np.mean(benchmark_latencies)
                res_benchmark_results[dataset_name] = {
                    "avg_accuracy": avg_acc,
                    "avg_latency": avg_lat,
                    "num_samples": len(benchmark_accuracies),
                }
                print(f"Acc={avg_acc:.1f}%, Lat={avg_lat:.0f}ms")
            else:
                print("No results")

        # Aggregate for this resolution point
        if res_metrics["accuracies"]:
            avg_acc = np.mean(res_metrics["accuracies"]) * 100
            avg_lat = np.mean(res_metrics["latencies"])

            print(f"\n   👉 Summary @ {res}px: Avg-{len(res_benchmark_results)} Acc={avg_acc:.2f}%, Avg Latency={avg_lat:.2f}ms")

            pareto_results.append({
                "resolution": res,
                "avg_accuracy": avg_acc,
                "avg_latency": avg_lat,
                "benchmark_details": res_benchmark_results,
            })

    # 3. Final Output & Plotting
    print("\n" + "=" * 70)

    if args.replication_target == "figure4":
        # Figure 4: Print data points and generate Pareto curve
        print("  🏆 Figure 4 Data Points (FastViT-HD)")
        print("=" * 70)
        print(f"{'Resolution':<12} {'Accuracy (%)':<15} {'Latency (ms)':<15}")
        print("-" * 70)
        for p in pareto_results:
            print(f"{p['resolution']:<12} {p['avg_accuracy']:<15.2f} {p['avg_latency']:<15.2f}")
        print("=" * 70)

        # Generate the Pareto Curve
        if pareto_results:
            save_pareto_curve(
                pareto_results,
                "Figure 4 Replication: Accuracy vs Latency (FastViT-HD)",
                "figure4_pareto_curve.png"
            )

        # Also save CSV with all resolution data
        csv_path = "results/figure4_pareto_data.csv"
        os.makedirs("results", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["resolution", "avg_accuracy", "avg_latency"])
            writer.writeheader()
            for p in pareto_results:
                writer.writerow({
                    "resolution": p["resolution"],
                    "avg_accuracy": p["avg_accuracy"],
                    "avg_latency": p["avg_latency"],
                })
        print(f"\n💾 Saved Pareto data to {csv_path}")

    elif args.replication_target in ["table6", "table11"]:
        # Table 6/11: Single resolution output
        if pareto_results:
            result = pareto_results[0]
            table_name = "Table 6" if args.replication_target == "table6" else "Table 11"
            print(f"  🏆 {table_name} Replication @ {result['resolution']}px")
            print("=" * 70)
            print(f"{'Benchmark':<15} {'Accuracy (%)':<15} {'Latency (ms)':<15} {'Samples':<10}")
            print("-" * 70)
            for name, data in result.get("benchmark_details", {}).items():
                print(f"{name:<15} {data['avg_accuracy']:<15.2f} {data['avg_latency']:<15.2f} {data['num_samples']:<10}")
            print("-" * 70)
            print(f"{'Avg-' + str(len(result.get('benchmark_details', {}))):<15} {result['avg_accuracy']:<15.2f} {result['avg_latency']:<15.2f}")
            print("=" * 70)

        # Save detailed results
        csv_path = f"results/{args.replication_target}_{args.resolution}px_results.csv"
        os.makedirs("results", exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["resolution", "benchmark", "id", "latency", "accuracy", "prediction", "ground_truths"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detailed_results)
        print(f"\n💾 Saved detailed results to {csv_path}")

        # Create bar chart visualization
        if pareto_results and pareto_results[0].get("benchmark_details"):
            data = {name: [res["avg_accuracy"]] for name, res in pareto_results[0]["benchmark_details"].items()}
            save_grouped_bar(
                data,
                ["Accuracy"],
                "Benchmark",
                "Accuracy (%)",
                f"{args.replication_target} @ {args.resolution}px",
                f"{args.replication_target}_{args.resolution}px_accuracy.png",
            )

    # Save all detailed results
    if all_detailed_results:
        csv_path = f"results/{args.replication_target}_detailed_results.csv"
        with open(csv_path, "w", newline="") as f:
            fieldnames = ["resolution", "benchmark", "id", "latency", "accuracy", "prediction", "ground_truths"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_detailed_results)
        print(f"💾 Saved all detailed results to {csv_path}")

    # Cleanup
    del model, tokenizer, image_processor
    gc.collect()
    torch.cuda.empty_cache()

    print("\n📋 See experiments/limitations.md for replication limitations.")


def run_stage_comparison(args):
    """
    Original stage comparison mode: Compare Stage-2 vs Stage-3 on TextVQA.
    """
    print(f"🚀 Starting Stage Comparison (TextVQA)")
    print(f"Stage 2: {args.stage2_path}")
    print(f"Stage 3: {args.stage3_path}")

    # Load Dataset
    dataset = get_benchmark_dataset(
        "textvqa", split="validation", max_samples=args.num_samples
    )
    samples = list(dataset)

    results = []

    # Define stages to test
    stages = [("Stage 2", args.stage2_path), ("Stage 3", args.stage3_path)]

    for stage_name, model_path in stages:
        print(f"\n🔄 Testing {stage_name}...")

        # Load model (one at a time to save VRAM)
        model, tokenizer, image_processor = setup_model(model_path, args.device)

        print("🔥 Warming up GPU...")
        dummy_image = Image.new('RGB', (512, 512), color='white')
        dummy_prompt = "Warmup run"

        try:
            _ = measure_inference(model, tokenizer, image_processor, dummy_image, dummy_prompt, args.device)
        except Exception as e:
            print(f"Warmup warning: {e}")

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        print("✅ Warmup complete. Starting experiment...")

        stage_latencies = []
        stage_accuracies = []

        for i, sample in enumerate(tqdm(samples, desc=f"Eval {stage_name}")):
            image = sample["image"]
            prompt = sample.get("question", "What is in this image?")
            ground_truths = sample.get("answers", [])

            try:
                latency, pred_text = measure_inference(
                    model, tokenizer, image_processor, image, prompt, args.device
                )

                acc = calculate_exact_match(pred_text, ground_truths)

                stage_latencies.append(latency)
                stage_accuracies.append(acc)

                results.append({
                    "stage": stage_name,
                    "id": sample["id"],
                    "latency": latency,
                    "accuracy": acc,
                    "prediction": pred_text,
                    "ground_truths": str(ground_truths),
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
    os.makedirs("results", exist_ok=True)
    csv_path = "results/stage_comparison_textvqa.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["stage", "id", "latency", "accuracy", "prediction", "ground_truths"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"💾 Saved results to {csv_path}")

    # Plotting
    if results:
        stage2_res = [r for r in results if r["stage"] == "Stage 2"]
        stage3_res = [r for r in results if r["stage"] == "Stage 3"]

        metrics = {
            "Latency (ms)": [
                np.mean([r["latency"] for r in stage2_res]) if stage2_res else 0,
                np.mean([r["latency"] for r in stage3_res]) if stage3_res else 0,
            ],
            "Accuracy (Exact Match)": [
                np.mean([r["accuracy"] for r in stage2_res]) if stage2_res else 0,
                np.mean([r["accuracy"] for r in stage3_res]) if stage3_res else 0,
            ],
        }

        save_grouped_bar(
            {
                "Stage 2": [metrics["Latency (ms)"][0]],
                "Stage 3": [metrics["Latency (ms)"][1]],
            },
            ["Average"],
            "Stage",
            "Latency (ms)",
            "Stage Comparison - Latency",
            "stage_comparison_latency.png",
        )

        save_grouped_bar(
            {
                "Stage 2": [metrics["Accuracy (Exact Match)"][0]],
                "Stage 3": [metrics["Accuracy (Exact Match)"][1]],
            },
            ["Average"],
            "Stage",
            "Accuracy",
            "Stage Comparison - Accuracy",
            "stage_comparison_accuracy.png",
        )

    print("\n📋 See experiments/limitations.md for replication limitations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastVLM Multi-Benchmark Evaluation (Figure 4, Table 6, Table 11, Stage Comparison)"
    )

    # Replication Target
    parser.add_argument(
        "--replication-target",
        default="stage-comparison",
        choices=["figure4", "table6", "table11", "stage-comparison"],
        help="Which paper result to replicate"
    )

    # Model paths
    parser.add_argument(
        "--model-path",
        default="../checkpoints/llava-fastvithd_0.5b_stage3",
        help="Path to model checkpoint (for figure4/table6/table11)"
    )
    parser.add_argument(
        "--stage2-path",
        default="../checkpoints/llava-fastvithd_0.5b_stage2",
        help="Path to Stage-2 checkpoint (for stage-comparison)"
    )
    parser.add_argument(
        "--stage3-path",
        default="../checkpoints/llava-fastvithd_0.5b_stage3",
        help="Path to Stage-3 checkpoint (for stage-comparison)"
    )

    # Resolution (Critical for Figure 4 and Table 11)
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help="Input resolution in pixels (e.g., 256, 512, 768, 1024, 1536)"
    )

    # Samples
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of samples per benchmark. Set to None for full dataset."
    )

    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    # Route to appropriate experiment
    if args.replication_target == "stage-comparison":
        run_stage_comparison(args)
    else:
        run_multi_benchmark_experiment(args)
