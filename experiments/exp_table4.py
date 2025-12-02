"""
make_table4_fastvlm.py

Replicate Table 4 from the FastVLM paper - Visual Token Efficiency benchmark.

Encoders & Resolutions (matching paper):
- ConvNeXt-L: 320, 512
- FastViT-HD: 512, 768, 1024

Datasets: TextVQA, DocVQA

Metrics per encoder/resolution:
- #Visual Tokens (based on downsampling factor)
- Accuracy on each dataset (TextVQA, DocVQA as columns)
"""

import time
from typing import List, Dict, Any, Optional, Tuple

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_large
from transformers import AutoModel

from utils_dataset import get_benchmark_dataset
from utils_plot import save_table_image

# Path where your encoder weights are stored
MODEL_PATH = "encoder_models/"

# ---------------- CONFIG ---------------- #

# Format: (encoder_name, display_name, resolution, downsample_factor)
# Downsample factors from paper: ConvNeXt=32x, FastViT-HD=64x
ENCODER_CONFIGS: List[Tuple[str, str, int, int]] = [
    # ConvNeXt-L resolutions from paper
    ("convnext", "ConvNeXt-L", 320, 32),
    ("convnext", "ConvNeXt-L", 512, 32),
    # FastViT-HD resolutions from paper
    ("fastvit", "FastViT-HD", 512, 64),
    ("fastvit", "FastViT-HD", 768, 64),
    ("fastvit", "FastViT-HD", 1024, 64),
]

BENCHMARKS = [
    ("textvqa", "TextVQA"),
    ("docvqa", "DocVQA"),
    ("gqa", "GQA"),
]

# Set to None to use full dataset, or a number for quick testing
MAX_SAMPLES: Optional[int] = 10

N_WARMUP = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------- UTILITIES ---------------- #

def make_preprocess(resolution: int) -> transforms.Compose:
    """Create preprocessing pipeline for given resolution."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def load_convnext_encoder() -> torch.nn.Module:
    """Load ConvNeXt-L with local checkpoint."""
    print("🔹 Loading ConvNeXt-L encoder (local weights)...")
    model = convnext_large(weights=None)
    weights = torch.load(MODEL_PATH + "convnext_large.pt", map_location="cpu")
    model.load_state_dict(weights)
    model.eval()
    return model


def load_fastvit_encoder() -> torch.nn.Module:
    """Load FastViT-HD from HuggingFace."""
    print("🔹 Loading FastViT-HD encoder (HuggingFace kevin510/fast-vit-hd)...")
    model = AutoModel.from_pretrained(
        "kevin510/fast-vit-hd",
        trust_remote_code=True
    ).eval()
    return model


def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
    """Unified forward for all encoders."""
    if hasattr(model, "encode_image"):
        return model.encode_image(x)
    return model(x)


def calculate_visual_tokens(resolution: int, downsample_factor: int) -> int:
    """Calculate number of visual tokens based on resolution and downsampling."""
    return (resolution // downsample_factor) ** 2


def measure_latency_on_dataset(
    model: torch.nn.Module,
    preprocess: transforms.Compose,
    benchmark_name: str,
    split: str = "validation",
    max_samples: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Measure average latency per sample on a given dataset.
    If max_samples is None, uses full dataset.
    """
    # Determine sample count
    if max_samples is None:
        # Use a large number to get "all" samples (streaming will stop when exhausted)
        sample_limit = 100000
        print(f"\n📚 Loading dataset: {benchmark_name} (split={split}, FULL dataset)")
    else:
        sample_limit = max_samples
        print(f"\n📚 Loading dataset: {benchmark_name} (split={split}, max_samples={max_samples})")

    samples = get_benchmark_dataset(benchmark_name, split, sample_limit)
    if len(samples) == 0:
        print(f"⚠️ No samples loaded for {benchmark_name}.")
        return {"avg_latency_ms": None, "num_samples": 0}

    print(f"   ➜ Loaded {len(samples)} samples.")

    model.eval().to(DEVICE)
    latencies = []

    # Warmup on first sample
    if len(samples) > 0:
        img0 = samples[0]["image"]
        inp0 = preprocess(img0).unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                encode_image_generic(model, inp0)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

    # Timed runs
    with torch.no_grad():
        for sample in samples:
            img = sample["image"]
            inp = preprocess(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = encode_image_generic(model, inp)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            latencies.append((end - start) * 1000.0)

    avg_latency = sum(latencies) / len(latencies)
    print(f"   ➜ Avg latency: {avg_latency:.2f} ms over {len(samples)} samples.")
    return {"avg_latency_ms": avg_latency, "num_samples": len(samples)}


def benchmark_table4() -> List[Dict[str, Any]]:
    """Run Table 4 benchmark and return structured results."""
    results = []

    print(f"\n🚀 Running Table 4 benchmark")
    print(f"   Device: {DEVICE} | dtype: {DTYPE}")
    print(f"   Max samples: {'FULL' if MAX_SAMPLES is None else MAX_SAMPLES}\n")

    # Cache loaded models to avoid reloading
    loaded_models: Dict[str, torch.nn.Module] = {}

    for enc_name, enc_display, resolution, downsample in ENCODER_CONFIGS:
        print(f"\n{'='*60}")
        print(f"  {enc_display} @ {resolution}px")
        print(f"{'='*60}")

        # Load model if not cached
        if enc_name not in loaded_models:
            try:
                if enc_name == "convnext":
                    model = load_convnext_encoder()
                elif enc_name == "fastvit":
                    model = load_fastvit_encoder()
                else:
                    raise ValueError(f"Unknown encoder '{enc_name}'")

                model.to(DEVICE).to(dtype=DTYPE)
                loaded_models[enc_name] = model
            except Exception as e:
                print(f"❌ Failed to load {enc_display}: {e}")
                # Add error entries for all benchmarks
                results.append({
                    "encoder": enc_display,
                    "resolution": resolution,
                    "tokens": calculate_visual_tokens(resolution, downsample),
                    "textvqa_latency": None,
                    "docvqa_latency": None,
                    "gqa_latency": None,
                    "error": str(e),
                })
                continue

        model = loaded_models[enc_name]
        preprocess = make_preprocess(resolution)
        tokens = calculate_visual_tokens(resolution, downsample)

        print(f"  Visual tokens: {tokens}")

        # Benchmark on each dataset
        dataset_latencies = {}
        for ds_name, ds_display in BENCHMARKS:
            try:
                stats = measure_latency_on_dataset(
                    model,
                    preprocess,
                    benchmark_name=ds_name,
                    split="validation",
                    max_samples=MAX_SAMPLES,
                )
                dataset_latencies[ds_name] = stats["avg_latency_ms"]
            except Exception as e:
                print(f"❌ Error on {ds_display}: {e}")
                dataset_latencies[ds_name] = None

        results.append({
            "encoder": enc_display,
            "resolution": resolution,
            "tokens": tokens,
            "textvqa_latency": dataset_latencies.get("textvqa"),
            "docvqa_latency": dataset_latencies.get("docvqa"),
            "gqa_latency": dataset_latencies.get("gqa"),
        })

    # Clean up
    for model in loaded_models.values():
        del model
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return results


def print_table(results: List[Dict[str, Any]]):
    """Print markdown table matching paper's Table 4 format."""
    print("\n" + "=" * 70)
    print("### Table 4 (Replication) — Visual Token Efficiency")
    print("=" * 70 + "\n")

    # Header matching paper format: Encoder | Resolution | #Tokens | TextVQA | DocVQA | GQA
    print("| Image Encoder | Input Res. | #Visual Tokens | TextVQA (ms) | DocVQA (ms) | GQA (ms) |")
    print("|---------------|------------|----------------|--------------|-------------|----------|")

    for r in results:
        enc = r["encoder"]
        res = r["resolution"]
        tok = r["tokens"]
        tvqa = f"{r['textvqa_latency']:.1f}" if r.get("textvqa_latency") else "-"
        dvqa = f"{r['docvqa_latency']:.1f}" if r.get("docvqa_latency") else "-"
        gqa = f"{r['gqa_latency']:.1f}" if r.get("gqa_latency") else "-"

        print(f"| {enc} | {res} | {tok} | {tvqa} | {dvqa} | {gqa} |")


def save_table_as_image(results: List[Dict[str, Any]]):
    """Save results as a LaTeX-style table image."""
    headers = ["Image Encoder", "Input Res.", "#Visual Tokens", "TextVQA (ms)", "DocVQA (ms)", "GQA (ms)"]
    rows = []

    for r in results:
        tvqa = f"{r['textvqa_latency']:.1f}" if r.get("textvqa_latency") else "-"
        dvqa = f"{r['docvqa_latency']:.1f}" if r.get("docvqa_latency") else "-"
        gqa = f"{r['gqa_latency']:.1f}" if r.get("gqa_latency") else "-"

        rows.append([
            r["encoder"],
            str(r["resolution"]),
            str(r["tokens"]),
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

