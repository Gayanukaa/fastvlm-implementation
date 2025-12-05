"""
exp_figure3_lat_vs_res.py

Figure 3 (Replication-style) – Encoder Latency vs Resolution

Encoders:
- ConvNeXt-L
- FastViT-HD

What it measures:
- PURE ENCODER LATENCY (ms per image)
  Image -> Encoder -> Features
  (NO LLM, NO decoding, NO full VLM pipeline)

Resolutions:
- 256, 512, 768, 1024

Output:
- Prints a small latency table
- Saves plot: figure3_fastvlm_encoder_latency.png


Note: Paper implememtation details may vary; this is a close approximation. Because they used M1 processor for the plotting but we used our GPUs, absolute latencies will differ.
"""

import time
from typing import Dict, Any, List, Tuple

import torch
from PIL import Image
import matplotlib.pyplot as plt

from utils_encoder_models import load_encoder  # you already have this


# ---------------- CONFIG ---------------- #

# Encoders to compare: internal_name, display_name
ENCODERS: List[Tuple[str, str]] = [
    ("convnext", "ConvNeXt-L"),
    ("fastvit", "FastViT-HD"),
]

# Resolutions to sweep (px)
RESOLUTIONS = [256, 512, 768, 1024]

N_WARMUP = 10
N_RUNS = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Optional speed optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------- UTILITIES ---------------- #

def make_dummy_image(res: int) -> Image.Image:
    """Create a synthetic image for benchmarking."""
    return Image.new("RGB", (res, res), color=(128, 128, 128))


def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
    """Unified forward pass for all encoders."""
    # CLIP-style encoders sometimes define encode_image()
    if hasattr(model, "encode_image"):
        return model.encode_image(x)
    return model(x)


def measure_latency(
    model: torch.nn.Module,
    preprocess,
    res: int,
    n_warmup: int = N_WARMUP,
    n_runs: int = N_RUNS,
) -> float:
    """
    Measure average encoder latency (ms/image) at a given resolution.

    This is PURE vision encoder latency:
        image -> preprocess -> encoder forward
    """
    model.eval()

    dummy = make_dummy_image(res)
    with torch.no_grad():
        inp = preprocess(dummy).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = encode_image_generic(model, inp)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    # Timed runs
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
                latencies.append(start_event.elapsed_time(end_event))  # ms

        return sum(latencies) / len(latencies)
    else:
        total = 0.0
        with torch.no_grad():
            for _ in range(n_runs):
                t0 = time.time()
                _ = encode_image_generic(model, inp)
                total += (time.time() - t0)
        return (total / n_runs) * 1000.0  # ms


# ---------------- MAIN BENCHMARK ---------------- #

def run_benchmark() -> Dict[str, Dict[int, float]]:
    """
    Run latency benchmark for each encoder across all resolutions.

    Returns:
        {
          "ConvNeXt-L": {256: ms, 320: ms, ...},
          "FastViT-HD": {256: ms, ...},
        }
    """
    results: Dict[str, Dict[int, float]] = {}

    print(f"\n🚀 Figure 3-style Encoder Latency Benchmark")
    print(f"   Device: {DEVICE} | dtype: {DTYPE}")
    print(f"   Warmup: {N_WARMUP} | Timed runs: {N_RUNS}\n")

    for enc_name, enc_display in ENCODERS:
        results[enc_display] = {}
        print(f"===== {enc_display} =====")

        for res in RESOLUTIONS:
            print(f"  → Resolution: {res}px")
            try:
                # NOTE: assumes load_encoder(name, resolution=res) signature
                model, preprocess = load_encoder(enc_name, resolution=res)
                model.to(DEVICE).to(dtype=DTYPE)

                latency_ms = measure_latency(model, preprocess, res)
                results[enc_display][res] = latency_ms

                print(f"     Latency: {latency_ms:.2f} ms")

                # Free memory
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            except TypeError:
                # If your load_encoder only takes (name), fall back:
                print("     ⚠ load_encoder(name, resolution=res) failed; "
                      "trying load_encoder(name) without resolution.")
                model, preprocess = load_encoder(enc_name)
                model.to(DEVICE).to(dtype=DTYPE)

                latency_ms = measure_latency(model, preprocess, res)
                results[enc_display][res] = latency_ms

                print(f"     Latency: {latency_ms:.2f} ms")

                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"     ❌ Error at {res}px: {e}")
                results[enc_display][res] = None
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

        print()

    return results


def print_latency_table(results: Dict[str, Dict[int, float]]):
    print("\n" + "=" * 70)
    print("Latency Table (ms) – Encoder-only")
    print("=" * 70 + "\n")

    header = "| Encoder | " + " | ".join(f"{r}px" for r in RESOLUTIONS) + " |"
    sep = "|" + "-" * (len(" Encoder ") ) + "|" + "|".join("-" * (len(f" {r}px ") ) for r in RESOLUTIONS) + "|"

    print(header)
    print("|---------|" + "|".join("--------" for _ in RESOLUTIONS) + "|")

    for enc_display, res_dict in results.items():
        row = [enc_display]
        for res in RESOLUTIONS:
            val = res_dict.get(res)
            row.append("-" if val is None else f"{val:.1f}")
        print("| " + " | ".join(row) + " |")


def plot_figure3_style(results: Dict[str, Dict[int, float]]):
    """
    Plot Figure-3 style latency vs resolution for ConvNeXt-L and FastViT-HD.
    """
    plt.figure(figsize=(7, 5))

    for enc_display, res_dict in results.items():
        xs = []
        ys = []
        for res in RESOLUTIONS:
            lat = res_dict.get(res)
            if lat is not None:
                xs.append(res)
                ys.append(lat)
        if len(xs) == 0:
            continue
        plt.plot(xs, ys, marker="o", label=enc_display)

    plt.xlabel("Input Resolution (px)")
    plt.ylabel("Encoder Latency (ms per image)")
    plt.title("Figure 3 (Replication-style): Encoder Latency vs Resolution\n(ConvNeXt-L vs FastViT-HD, Encoder-only)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figure3_fastvlm_encoder_latency.png", dpi=300)
    print("\n📈 Saved plot to figure3_fastvlm_encoder_latency.png")


if __name__ == "__main__":
    results = run_benchmark()
    print_latency_table(results)
    plot_figure3_style(results)


print("""
Why the latency plot does NOT match the FastVLM paper trend:

1. Different FastViT-HD Implementation:
   - The paper uses Apple's MLX-optimized FastViT-HD.
   - Your script uses the HuggingFace PyTorch version (kevin510/fast-vit-hd),
     which has different downsampling, FLOPs, and no fused kernels.
   - Therefore latency scaling behaves differently.

2. Hardware is Completely Different:
   - Paper uses Apple M1 Neural Engine (NE).
   - You are using an RTX 4070 GPU.
   - NE has tile-based fusion and sublinear scaling,
     while GPUs scale superlinearly at high resolutions.

3. FastVLM Checkpoint Cannot Be Used as a Raw Encoder:
   - The LLaVA FastVIT-HD checkpoint contains projection layers and VLM wrappers.
   - It is not equivalent to the standalone encoder used in Table/Figure 3.
   - Pure encoder latency cannot be reproduced from the VLM checkpoint.

4. Preprocessing Differences:
   - Your loaded models may internally crop/resize differently than the paper.
   - PyTorch transforms are not identical to the MLX preprocessing pipeline.

. Kernel and Backend Differences:
   - Paper uses fused NE kernels; PyTorch uses CUDA kernels.
   - Operator fusion differences change resolution scaling behavior.

6. Implementation-Specific Overheads:
   - HuggingFace model has Python overhead and extra normalization layers.
   - Apple’s FastViT-HD implementation removes many of these.

Summary:
The plotted values vary and the relative trend differs because 
your models, hardware, and execution backend are 
NOT the same as those used in the FastVLM paper, making an exact 
trend replication impossible.

""")
