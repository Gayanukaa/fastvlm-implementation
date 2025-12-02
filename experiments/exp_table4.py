"""
exp_table4.py

Replicate Table 4 from the FastVLM paper - Visual Token Efficiency benchmark.

Encoders & Resolutions (matching paper order):
- FastViT-HD: 256, 512, 768, 1024
- ConvNeXt-L: 320, 512

Datasets: TextVQA, DocVQA, GQA

Metrics per encoder/resolution:
- #Visual Tokens (based on downsampling factor)
- Latency (ms) - single column
- Accuracy on each dataset (TextVQA, DocVQA, GQA as columns)
"""

import os
import sys
import time
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for llava imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from utils_dataset import get_benchmark_dataset
from utils_plot import save_table_image

# ---------------- CONFIG ---------------- #

# Model checkpoint path
MODEL_PATH = "../checkpoints/llava-fastvithd_0.5b_stage3"

# Format: (encoder_name, display_name, resolution, downsample_factor)
# Downsample factors from paper: ConvNeXt=32x, FastViT-HD=64x
# Order matches paper Table 4
ENCODER_CONFIGS: List[Tuple[str, str, int, int]] = [
    ("fastvit", "FastViT-HD", 256, 64),   # 16 tokens
    ("convnext", "ConvNeXt-L", 320, 32),  # 100 tokens
    ("fastvit", "FastViT-HD", 512, 64),   # 64 tokens
    ("fastvit", "FastViT-HD", 768, 64),   # 144 tokens
    ("convnext", "ConvNeXt-L", 512, 32),  # 256 tokens
    ("fastvit", "FastViT-HD", 1024, 64),  # 256 tokens
]

BENCHMARKS = [
    ("textvqa", "TextVQA"),
    ("docvqa", "DocVQA"),
    # ("gqa", "GQA"),  # Commented out - dataset loading issues
]

# Set to None to use full dataset, or a number for quick testing
MAX_SAMPLES: Optional[int] = 10

N_WARMUP = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CONV_MODE = "qwen_2"


# ---------------- ACCURACY METRICS ---------------- #

def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
    import re
    import string

    s = s.lower().strip()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Remove extra whitespace
    s = " ".join(s.split())
    return s


def vqa_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """
    VQA accuracy metric: prediction matches if it matches any ground truth.
    Uses relaxed matching (normalized comparison).
    """
    pred_norm = normalize_answer(prediction)

    for gt in ground_truths:
        gt_norm = normalize_answer(gt)
        if pred_norm == gt_norm:
            return 1.0
        # Also check if prediction contains the answer or vice versa
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return 1.0

    return 0.0


def exact_match_accuracy(prediction: str, ground_truths: List[str]) -> float:
    """Exact match accuracy for GQA."""
    pred_norm = normalize_answer(prediction)

    for gt in ground_truths:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0


# ---------------- MODEL UTILITIES ---------------- #

def calculate_visual_tokens(resolution: int, downsample_factor: int) -> int:
    """Calculate number of visual tokens based on resolution and downsampling."""
    return (resolution // downsample_factor) ** 2


def load_vlm_model():
    """Load the full VLM model for inference."""
    print(f"🔹 Loading VLM model from {MODEL_PATH}...")

    disable_torch_init()
    model_path = os.path.expanduser(MODEL_PATH)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device=DEVICE
    )

    model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"✅ Model loaded successfully")
    return tokenizer, model, image_processor


def run_inference(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str
) -> Tuple[str, float]:
    """
    Run VLM inference and return (answer, latency_ms).
    """
    # Construct prompt
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(device=DEVICE)

    # Process image
    image_tensor = process_images([image], image_processor, model.config)[0]

    # Run inference with timing
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0).to(dtype=DTYPE),
            image_sizes=[image.size],
            do_sample=False,  # Greedy decoding for consistency
            max_new_tokens=64,
            use_cache=True,
        )

    if DEVICE == "cuda":
        torch.cuda.synchronize()

    latency_ms = (time.time() - start_time) * 1000

    # Decode output
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return answer, latency_ms


# ---------------- BENCHMARK FUNCTIONS ---------------- #

def evaluate_on_dataset(
    model,
    tokenizer,
    image_processor,
    benchmark_name: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate model on a benchmark dataset.
    Returns accuracy and average latency.
    """
    # Determine sample count
    if max_samples is None:
        sample_limit = 100000
        print(f"\n📚 Loading dataset: {benchmark_name} (split={split}, FULL dataset)")
    else:
        sample_limit = max_samples
        print(f"\n📚 Loading dataset: {benchmark_name} (split={split}, max_samples={max_samples})")

    samples = get_benchmark_dataset(benchmark_name, split, sample_limit)
    if len(samples) == 0:
        print(f"⚠️ No samples loaded for {benchmark_name}.")
        return {"accuracy": None, "avg_latency_ms": None, "num_samples": 0}

    print(f"   ➜ Loaded {len(samples)} samples.")

    # Warmup
    if len(samples) > 0:
        print("   ➜ Warming up...")
        for _ in range(min(N_WARMUP, len(samples))):
            _ = run_inference(model, tokenizer, image_processor,
                            samples[0]["image"], samples[0]["question"])

    # Evaluate
    correct = 0
    total = 0
    latencies = []

    # Choose accuracy metric based on dataset
    if benchmark_name.lower() == "gqa":
        acc_func = exact_match_accuracy
    else:
        acc_func = vqa_accuracy

    for sample in tqdm(samples, desc=f"Evaluating {benchmark_name}"):
        try:
            prediction, latency = run_inference(
                model, tokenizer, image_processor,
                sample["image"], sample["question"]
            )

            acc = acc_func(prediction, sample["answers"])
            correct += acc
            total += 1
            latencies.append(latency)

        except Exception as e:
            print(f"⚠️ Error on sample: {e}")
            continue

    accuracy = (correct / total * 100) if total > 0 else None
    avg_latency = sum(latencies) / len(latencies) if latencies else None

    print(f"   ➜ Accuracy: {accuracy:.1f}% | Avg Latency: {avg_latency:.1f}ms")

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "num_samples": total,
    }


def benchmark_table4() -> List[Dict[str, Any]]:
    """Run Table 4 benchmark and return structured results."""
    results = []

    print(f"\n🚀 Running Table 4 benchmark")
    print(f"   Device: {DEVICE} | dtype: {DTYPE}")
    print(f"   Max samples: {'FULL' if MAX_SAMPLES is None else MAX_SAMPLES}\n")

    # Load VLM model once
    try:
        tokenizer, model, image_processor = load_vlm_model()
    except Exception as e:
        print(f"❌ Failed to load VLM model: {e}")
        return []

    for enc_name, enc_display, resolution, downsample in ENCODER_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {enc_display} @ {resolution}px")
        print(f"{'='*60}")

        tokens = calculate_visual_tokens(resolution, downsample)
        print(f"  Visual tokens: {tokens}")

        # Evaluate on each benchmark
        dataset_results = {}
        total_latency = []

        for ds_name, ds_display in BENCHMARKS:
            try:
                stats = evaluate_on_dataset(
                    model,
                    tokenizer,
                    image_processor,
                    benchmark_name=ds_name,
                    split="validation",
                    max_samples=MAX_SAMPLES,
                )
                dataset_results[ds_name] = stats["accuracy"]
                if stats["avg_latency_ms"]:
                    total_latency.append(stats["avg_latency_ms"])
            except Exception as e:
                print(f"❌ Error on {ds_display}: {e}")
                dataset_results[ds_name] = None

        # Average latency across all datasets
        avg_latency = sum(total_latency) / len(total_latency) if total_latency else None

        results.append({
            "encoder": enc_display,
            "resolution": resolution,
            "tokens": tokens,
            "latency": avg_latency,
            "textvqa_acc": dataset_results.get("textvqa"),
            "docvqa_acc": dataset_results.get("docvqa"),
            "gqa_acc": dataset_results.get("gqa"),
        })

    # Clean up
    del model, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


def print_table(results: List[Dict[str, Any]]):
    """Print markdown table matching paper's Table 4 format."""
    print("\n" + "=" * 80)
    print("### Table 4 (Replication) — Visual Token Efficiency")
    print("=" * 80 + "\n")

    # Header: Encoder | Resolution | #Tokens | Latency | TextVQA | DocVQA | GQA
    print("| Image Encoder | Input Res. | #Tokens | Latency (ms) | TextVQA | DocVQA | GQA |")
    print("|---------------|------------|---------|--------------|---------|--------|-----|")

    for r in results:
        enc = r["encoder"]
        res = r["resolution"]
        tok = r["tokens"]
        lat = f"{r['latency']:.1f}" if r.get("latency") else "-"
        tvqa = f"{r['textvqa_acc']:.1f}" if r.get("textvqa_acc") is not None else "-"
        dvqa = f"{r['docvqa_acc']:.1f}" if r.get("docvqa_acc") is not None else "-"
        gqa = f"{r['gqa_acc']:.1f}" if r.get("gqa_acc") is not None else "-"

        print(f"| {enc} | {res} | {tok} | {lat} | {tvqa} | {dvqa} | {gqa} |")


def save_table_as_image(results: List[Dict[str, Any]]):
    """Save results as a LaTeX-style table image."""
    headers = ["Image Encoder", "Input Res.", "#Tokens", "Latency (ms)", "TextVQA", "DocVQA", "GQA"]
    rows = []

    for r in results:
        lat = f"{r['latency']:.1f}" if r.get("latency") else "-"
        tvqa = f"{r['textvqa_acc']:.1f}" if r.get("textvqa_acc") is not None else "-"
        dvqa = f"{r['docvqa_acc']:.1f}" if r.get("docvqa_acc") is not None else "-"
        gqa = f"{r['gqa_acc']:.1f}" if r.get("gqa_acc") is not None else "-"

        rows.append([
            r["encoder"],
            str(r["resolution"]),
            str(r["tokens"]),
            lat,
            tvqa,
            dvqa,
            gqa,
        ])

    save_table_image(
        headers,
        rows,
        "table4_visual_token_efficiency.png",
        title="Table 4: Visual Token Efficiency",
    )


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    results = benchmark_table4()
    print_table(results)
    save_table_as_image(results)


