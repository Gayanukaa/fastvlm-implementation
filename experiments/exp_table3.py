"""
exp_table3.py

Replicates Table 3 from the FastVLM paper - Encoder comparison benchmark.

Encoders:
- ViT-L/14 (OpenCLIP) - 224px resolution
- ConvNeXt-L - 320px resolution (as per paper)
- FastViT-HD - 224px resolution (using local efficient implementation)

Outputs:
- Encoder Size (parameters in millions) - matches paper's "Encoder Size (M)"
- Input Resolution
- Latency (ms/image)

Hardware: RTX 4070 8GB (or your GPU)
"""

import time
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image

from utils_encoder_models import load_encoder
from utils_plot import save_table_image

# Enable optimizations for convolutional networks
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------- CONFIG ---------------- #

# Order matches the paper: ViT-L/14, ConvNeXt-L, FastViT-HD
# Format: (internal_name, display_name, resolution)
ENCODER_CONFIGS: List[Tuple[str, str, int]] = [
    ("vit", "ViT-L/14", 224),          # Paper: 304M params, 224px, 47.2ms
    ("convnext", "ConvNeXt-L", 320),   # Paper: 200M params, 320px, 34.4ms
    ("fastvit", "FastViT-HD", 224),    # Paper: 125M params, 224px, 6.8ms
]

N_WARMUP = 20
N_RUNS = 100  # More runs for stable measurements

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------- UTILITIES ---------------- #

def count_parameters(model: torch.nn.Module) -> float:
    """Return total parameters in millions (matches paper's 'Encoder Size (M)')."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_dummy_image(res: int) -> Image.Image:
    """Create a synthetic image for benchmarking."""
    return Image.new("RGB", (res, res), color=(128, 128, 128))


def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
    """Unified forward pass for all encoders."""
    # CLIP models use encode_image()
    if hasattr(model, "encode_image"):
        return model.encode_image(x)
    # Standard forward for FastViT-HD, ConvNeXt
    return model(x)


def measure_latency(
    model: torch.nn.Module,
    preprocess,
    res: int,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
) -> float:
    """
    Measure average encode latency (ms/image).
    Uses CUDA events for precise GPU timing.

    Note: Model should already be on correct device and dtype before calling.
    """
    model.eval()

    dummy = make_dummy_image(res)
    with torch.no_grad():
        inp = preprocess(dummy).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Warmup runs
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = encode_image_generic(model, inp)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    # Timed runs using CUDA events for accuracy
    if DEVICE == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        latencies = []
        with torch.no_grad():
            for _ in range(n_runs):
                start_event.record()
                _ = encode_image_generic(model, inp)
                end_event.record()
                torch.cuda.synchronize()
                latencies.append(start_event.elapsed_time(end_event))

        return sum(latencies) / len(latencies)
    else:
        # CPU fallback
        total = 0
        with torch.no_grad():
            for _ in range(n_runs):
                start = time.time()
                _ = encode_image_generic(model, inp)
                total += time.time() - start
        return (total / n_runs) * 1000


# ---------------- MAIN BENCHMARK ---------------- #

def benchmark() -> List[Tuple]:
    """Run benchmark and return results."""
    results = []

    print(f"\n🚀 Running Table 3 replication benchmark")
    print(f"   Device: {DEVICE} | dtype: {DTYPE}")
    print(f"   Warmup: {N_WARMUP} runs | Timed: {N_RUNS} runs per encoder\n")

    for name, display, res in ENCODER_CONFIGS:
        print(f"===== {display} ({res}px) =====")

        try:
            model, preprocess = load_encoder(name, resolution=res)
            model.to(DEVICE).to(dtype=DTYPE)

            params = count_parameters(model)
            latency = measure_latency(model, preprocess, res)

            print(f"  ✓ Params: {params:.1f}M | Latency: {latency:.2f}ms\n")
            results.append((display, params, res, latency, "Measured"))

            # Free memory
            del model
            torch.cuda.empty_cache() if DEVICE == "cuda" else None

        except Exception as e:
            print(f"  ❌ Failed: {e}\n")
            results.append((display, "-", res, "-", f"Error: {e}"))

    return results


def print_table(results: List[Tuple]):
    """Print markdown table matching paper's Table 3 format."""
    print("\n" + "=" * 70)
    print("### Table 3 (Replication) — Encoder Comparison")
    print("=" * 70 + "\n")

    # Header matching paper
    print("| Image Encoder | Encoder Size (M) | Input Res. | Latency (ms) |")
    print("|---------------|------------------|------------|--------------|")

    for enc, params, res, latency, note in results:
        if isinstance(params, float):
            print(f"| {enc} | {params:.0f} | {res} | {latency:.1f} |")
        else:
            print(f"| {enc} | {params} | {res} | {latency} |")

    # print("\n" + "-" * 70)
    # print("Reference (Original Paper Table 3):")
    # print("-" * 70)
    # print("| Image Encoder | Encoder Size (M) | Input Res. | Latency (ms) |")
    # print("|---------------|------------------|------------|--------------|")
    # print("| ViT-L/14      | 304              | 224        | 47.2         |")
    # print("| ConvNeXt-L    | 200              | 320        | 34.4         |")
    # print("| FastViT-HD    | 125              | 224        | 6.8          |")


def save_table_as_image(results: List[Tuple]):
    """Save results as a LaTeX-style table image."""
    headers = ["Image Encoder", "Encoder Size (M)", "Input Res.", "Latency (ms)"]
    rows = []

    for enc, params, res, latency, note in results:
        p_str = f"{params:.0f}" if isinstance(params, float) else str(params)
        l_str = f"{latency:.1f}" if isinstance(latency, float) else str(latency)
        rows.append([enc, p_str, str(res), l_str])

    save_table_image(
        headers,
        rows,
        "table3_encoder_comparison.png",
        title="Table 3: Encoder Comparison",
    )


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    results = benchmark()
    print_table(results)
    save_table_as_image(results)
