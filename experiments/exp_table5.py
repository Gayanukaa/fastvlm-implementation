"""
exp_table5.py

Table 5 Replication (FastViT-HD Visual Token Efficiency):

For FastViT-HD, measure for each resolution:
- Visual token count (based on 64x downsampling): (resolution / 64)²
- Benchmark accuracy (SQA, TextVQA, POPE, SeedBench) using full VLM inference at each resolution

Key: Images are RESIZED to each target resolution before inference to properly
measure how accuracy scales with visual token count.

Uses:
- FastViT-HD encoder from the VLM checkpoint
- Full VLM inference for accuracy measurement
- SQA, TextVQA, POPE, SeedBench from HuggingFace datasets (via utils_dataset)

Output (printed/saved):

| Model | Resolution | Visual Tokens | SQA Acc (%) | TextVQA Acc (%) | POPE Acc (%) | SeedBench Acc (%) |
"""

import os
import re
import string
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory to path for llava imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from utils_dataset import get_benchmark_dataset
from utils_plot import save_table_image

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

# ---------------- CONFIG ---------------- #

# Model checkpoint path (default, can be overridden via args)
DEFAULT_MODEL_PATH = "../checkpoints/llava-fastvithd_0.5b_stage3"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CONV_MODE = "qwen_2"

# Resolutions to evaluate (matching paper Table 5)
# These are the input resolutions to the vision encoder
RESOLUTIONS = [256, 512, 768, 1024]

# FastViT-HD downsampling factor (64x)
# Visual tokens = (resolution / 64)²
DOWNSAMPLE_FACTOR = 64

# Default: None = use full dataset, or a number for quick testing
# Can be overridden via --num-samples argument
MAX_SAMPLES: Optional[int] = None

N_WARMUP = 3


# ---------------- ACCURACY METRICS ---------------- #


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison."""
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
    VQA-style accuracy metric: prediction matches if it matches any ground truth.
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

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    print(f"✅ Model loaded successfully")
    return tokenizer, model, image_processor


def resize_image(image: Image.Image, target_resolution: int) -> Image.Image:
    """
    Resize image to target resolution (square).
    Uses LANCZOS resampling for high quality downscaling.
    """
    image = image.convert("RGB")
    return image.resize((target_resolution, target_resolution), Image.LANCZOS)


def run_inference_at_resolution(
    model,
    tokenizer,
    image_processor,
    image: Image.Image,
    question: str,
    target_resolution: int,
) -> Tuple[str, float, float]:
    """
    Run VLM inference at a specific resolution and return:
        (answer, full_latency_ms, encoder_latency_ms).

    - The image is resized to target_resolution before being processed.
    - encoder_latency_ms measures only the vision tower forward pass,
      similar to "Latency Enc. (ms)" in the paper.
    """
    # Resize image to target resolution
    resized_image = resize_image(image, target_resolution)

    # Construct prompt
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize
    input_ids = (
        tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
        .unsqueeze(0)
        .to(device=DEVICE)
    )

    # Process the resized image -> tensor on DEVICE with correct dtype
    image_tensor = process_images([resized_image], image_processor, model.config)[0]
    image_tensor = image_tensor.to(device=DEVICE, dtype=DTYPE)

    # 1) Encoder-only latency (vision tower forward)
    vision_tower = model.get_vision_tower()

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    enc_start = time.time()
    with torch.inference_mode():
        _ = vision_tower(image_tensor.unsqueeze(0))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    encoder_latency_ms = (time.time() - enc_start) * 1000

    # 2) Full VLM generation latency (for reference)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    gen_start = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.unsqueeze(0),
            image_sizes=[resized_image.size],  # Use resized image size
            do_sample=False,  # Greedy decoding for consistency
            max_new_tokens=64,
            use_cache=True,
        )
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    full_latency_ms = (time.time() - gen_start) * 1000

    # Decode output
    answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

    return answer, full_latency_ms, encoder_latency_ms


# ---------------- BENCHMARK FUNCTIONS ---------------- #


def evaluate_at_resolution(
    model,
    tokenizer,
    image_processor,
    samples: List[Dict[str, Any]],
    target_resolution: int,
) -> Dict[str, Any]:
    """
    Evaluate model on samples at a specific resolution.
    Returns:
        {
          "accuracy": accuracy_percent,
          "avg_latency_ms": avg_full_vlm_latency,
          "avg_encoder_latency_ms": avg_encoder_latency,
          "num_samples": total,
        }
    """
    print(
        f"\n   📐 Evaluating at {target_resolution}x{target_resolution} resolution..."
    )

    if len(samples) == 0:
        print("      ⚠️ No samples for this benchmark.")
        return {
            "accuracy": None,
            "avg_latency_ms": None,
            "avg_encoder_latency_ms": None,
            "num_samples": 0,
        }

    # Warmup at this resolution
    print(f"      Warming up...")
    for _ in range(min(N_WARMUP, len(samples))):
        _ = run_inference_at_resolution(
            model,
            tokenizer,
            image_processor,
            samples[0]["image"],
            samples[0]["question"],
            target_resolution,
        )

    # Evaluate
    correct = 0.0
    total = 0
    full_latencies = []
    encoder_latencies = []

    for sample in tqdm(samples, desc=f"Eval @{target_resolution}px"):
        try:
            prediction, full_lat, enc_lat = run_inference_at_resolution(
                model,
                tokenizer,
                image_processor,
                sample["image"],
                sample["question"],
                target_resolution,
            )

            acc = vqa_accuracy(prediction, sample["answers"])
            correct += acc
            total += 1
            full_latencies.append(full_lat)
            encoder_latencies.append(enc_lat)

        except Exception as e:
            print(f"⚠️ Error on sample: {e}")
            continue

    accuracy = (correct / total * 100) if total > 0 else None
    avg_full_latency = (
        sum(full_latencies) / len(full_latencies) if full_latencies else None
    )
    avg_encoder_latency = (
        sum(encoder_latencies) / len(encoder_latencies) if encoder_latencies else None
    )

    if accuracy is not None:
        print(
            f"      ➜ Accuracy: {accuracy:.1f}% | "
            f"Avg Encoder Latency: {avg_encoder_latency:.1f}ms | "
            f"Avg Full VLM Latency: {avg_full_latency:.1f}ms"
        )
    else:
        print("      ➜ Accuracy / latency could not be computed (no valid samples).")

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_full_latency,
        "avg_encoder_latency_ms": avg_encoder_latency,
        "num_samples": total,
    }


# ---------------- MAIN TABLE 5 ---------------- #


def build_table5() -> List[Dict[str, Any]]:
    """Run Table 5 benchmark and return results."""
    rows: List[Dict[str, Any]] = []

    print(f"\n🚀 Running Table 5 benchmark (FastViT-HD Visual Token Efficiency)")
    print(f"   Device: {DEVICE} | dtype: {DTYPE}")
    print(f"   Max samples: {'FULL' if MAX_SAMPLES is None else MAX_SAMPLES}")
    print(f"   Resolutions: {RESOLUTIONS}")
    print(f"   Downsampling factor: {DOWNSAMPLE_FACTOR}x\n")

    # Load VLM model once
    try:
        tokenizer, model, image_processor = load_vlm_model()
    except Exception as e:
        print(f"❌ Failed to load VLM model: {e}")
        return []

    # ---------- Load benchmark datasets (SQA, TextVQA, POPE, SeedBench) ---------- #

    print("\n" + "=" * 60)
    print("  Loading benchmark samples...")
    print("=" * 60)

    # Order matters: SQA, TextVQA, POPE, SeedBench
    benchmark_specs = [
        ("sqa", "validation"),  # SQA
        ("textvqa", "validation"),  # TextVQA
        ("pope", "validation"),  # POPE
        ("seedbench", "validation"),  # SeedBench
    ]

    benchmark_samples: Dict[str, List[Dict[str, Any]]] = {}

    for bench_name, split in benchmark_specs:
        try:
            print(f"   ➜ Loading {bench_name} ({split})...")
            # If MAX_SAMPLES is None, get_benchmark_dataset should use full dataset.
            benchmark_samples[bench_name] = get_benchmark_dataset(
                bench_name, split, MAX_SAMPLES
            )
            print(f"      Loaded {len(benchmark_samples[bench_name])} samples.")
        except Exception as e:
            print(f"      ⚠️ Failed to load {bench_name}: {e}")
            benchmark_samples[bench_name] = []

    if len(benchmark_samples.get("textvqa", [])) == 0:
        print("⚠️ No TextVQA samples loaded. Exiting.")
        return []

    # ---------- Evaluate at each resolution for all benchmarks ---------- #

    print("\n" + "=" * 60)
    print("  Evaluating accuracy at each resolution...")
    print("=" * 60)

    for res in RESOLUTIONS:
        tokens = calculate_visual_tokens(res, DOWNSAMPLE_FACTOR)
        print(f"\n{'─'*50}")
        print(f"  Resolution: {res}px → {tokens} visual tokens")
        print(f"{'─'*50}")

        # For each benchmark, run evaluation if we have samples
        sqa_res = {"accuracy": None}
        textvqa_res = {"accuracy": None}
        pope_res = {"accuracy": None}
        seedbench_res = {"accuracy": None}

        if benchmark_samples.get("sqa"):
            print("\n[SQA]")
            sqa_res = evaluate_at_resolution(
                model, tokenizer, image_processor, benchmark_samples["sqa"], res
            )

        if benchmark_samples.get("textvqa"):
            print("\n[TextVQA]")
            textvqa_res = evaluate_at_resolution(
                model, tokenizer, image_processor, benchmark_samples["textvqa"], res
            )

        if benchmark_samples.get("pope"):
            print("\n[POPE]")
            pope_res = evaluate_at_resolution(
                model, tokenizer, image_processor, benchmark_samples["pope"], res
            )

        if benchmark_samples.get("seedbench"):
            print("\n[SeedBench]")
            seedbench_res = evaluate_at_resolution(
                model, tokenizer, image_processor, benchmark_samples["seedbench"], res
            )

        # We store one latency (e.g., TextVQA full latency) for reference if needed.
        rows.append(
            {
                "model": "FastViT-HD",
                "resolution": res,
                "tokens": tokens,
                # accuracies in requested order: SQA, TextVQA, POPE, SeedBench
                "sqa_acc": sqa_res.get("accuracy"),
                "textvqa_acc": textvqa_res.get("accuracy"),
                "pope_acc": pope_res.get("accuracy"),
                "seedbench_acc": seedbench_res.get("accuracy"),
                "avg_latency_ms": textvqa_res.get("avg_encoder_latency_ms"),
            }
        )

    # Clean up
    del model, tokenizer
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return rows


def print_markdown_table(rows: List[Dict[str, Any]]):
    """Print results as markdown table."""
    print("\n" + "=" * 80)
    print("### Table 5 (Replication)")
    print("=" * 80 + "\n")

    print(
        "| Model | Resolution | Visual Tokens | SQA Acc (%) | TextVQA Acc (%) | POPE Acc (%) | SeedBench Acc (%) |"
    )
    print(
        "|-------|------------|---------------|-------------|-----------------|--------------|-------------------|"
    )

    for r in rows:
        tok = "-" if r["tokens"] is None or r["tokens"] == -1 else str(r["tokens"])
        sqa = "-" if r.get("sqa_acc") is None else f"{r['sqa_acc']:.1f}"
        txt = "-" if r.get("textvqa_acc") is None else f"{r['textvqa_acc']:.1f}"
        pope = "-" if r.get("pope_acc") is None else f"{r['pope_acc']:.1f}"
        seed = "-" if r.get("seedbench_acc") is None else f"{r['seedbench_acc']:.1f}"
        print(
            f"| {r['model']} | {r['resolution']} | {tok} | {sqa} | {txt} | {pope} | {seed} |"
        )


def save_table_as_image(rows: List[Dict[str, Any]]):
    """Save results as a LaTeX-style table image."""
    headers = [
        "Model",
        "Resolution",
        "Visual Tokens",
        "SQA Acc (%)",
        "TextVQA Acc (%)",
        "POPE Acc (%)",
        "SeedBench Acc (%)",
    ]
    table_rows = []

    for r in rows:
        tok = "-" if r["tokens"] is None or r["tokens"] == -1 else str(r["tokens"])
        sqa = "-" if r.get("sqa_acc") is None else f"{r['sqa_acc']:.1f}"
        txt = "-" if r.get("textvqa_acc") is None else f"{r['textvqa_acc']:.1f}"
        pope = "-" if r.get("pope_acc") is None else f"{r['pope_acc']:.1f}"
        seed = "-" if r.get("seedbench_acc") is None else f"{r['seedbench_acc']:.1f}"
        table_rows.append(
            [
                r["model"],
                str(r["resolution"]),
                tok,
                sqa,
                txt,
                pope,
                seed,
            ]
        )

    save_table_image(
        headers,
        table_rows,
        "table5_fastvithd_efficiency.png",
        title="Table 5: FastViT-HD Visual Token Efficiency",
    )


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FastVLM Table 5 Replication: FastViT-HD Visual Token Efficiency"
    )
    parser.add_argument(
        "--model-path",
        default=DEFAULT_MODEL_PATH,
        help="Path to FastVLM checkpoint",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Samples for evaluation (default: full dataset)",
    )
    args = parser.parse_args()

    # Override module-level config with args
    MODEL_PATH = args.model_path
    MAX_SAMPLES = args.num_samples
    DEVICE = args.device
    DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

    rows = build_table5()

    if rows:
        print_markdown_table(rows)
        save_table_as_image(rows)

    print("\n📋 See experiments/limitations.md for replication limitations.")
