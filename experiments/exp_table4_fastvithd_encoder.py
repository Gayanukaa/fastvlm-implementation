"""
exp_table4_fastvithd_encoder.py

Table 4 – FastViT-HD Encoder Benchmarks (HuggingFace Model)

This script measures:
- Visual token count from encoder output
- Average encoder latency (ms) for each tested input resolution

Model:
- HuggingFace: "kevin510/fast-vit-hd" (encoder-only evaluation)

Notes:
- This is NOT a VLM test (no text component).
- No TextVQA dataset used.
- Latency is computed using a single synthetic dummy image per resolution.

Output Table Columns:
| Model | Resolution | Visual Tokens | Avg Encoder Latency (ms) |
"""

import time
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


# ---------------- CONFIG ---------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Resolutions to evaluate for Table 3
RESOLUTIONS = [256, 512, 768, 1024]

N_WARMUP = 5      # warmup runs
N_RUNS = 100       # timed inference runs per resolution


# ---------------- MODEL LOADING ---------------- #

def load_fastvithd():
    print("🔹 Loading FastViT-HD encoder from 'kevin510/fast-vit-hd' ...")
    model = AutoModel.from_pretrained(
        "kevin510/fast-vit-hd",
        trust_remote_code=True
    ).eval()

    processor = AutoImageProcessor.from_pretrained(
        "kevin510/fast-vit-hd",
        trust_remote_code=True
    )

    model.to(DEVICE).to(dtype=DTYPE)
    return model, processor


# ---------------- UTILITIES ---------------- #

def make_dummy_image(res: int) -> Image.Image:
    """Create a black dummy image of resolution res x res."""
    return Image.new("RGB", (res, res), color=(0, 0, 0))


def prepare_input(processor, img: Image.Image, res: int) -> torch.Tensor:
    """Prepare input for the encoder."""
    img_resized = img.resize((res, res))
    batch = processor(
        img_resized,
        do_resize=False,
        do_center_crop=False,
        return_tensors="pt",
    )
    return batch["pixel_values"].to(DEVICE, dtype=DTYPE)


def encode_image(model, x: torch.Tensor):
    return model(x)


def estimate_tokens(model, processor, res: int) -> int:
    """Estimate number of visual tokens using one dummy image."""
    img = make_dummy_image(res)
    pixel_values = prepare_input(processor, img, res)

    with torch.no_grad():
        out = encode_image(model, pixel_values)

    # HF output styles
    if hasattr(out, "last_hidden_state"):
        return int(out.last_hidden_state.shape[1])
    if isinstance(out, dict) and "last_hidden_state" in out:
        return int(out["last_hidden_state"].shape[1])

    # Generic fallback
    if hasattr(out, "shape") and out.ndim >= 3:
        return int(out.shape[1])
    if isinstance(out, tuple) and len(out) > 0 and hasattr(out[0], "shape"):
        return int(out[0].shape[1])

    return -1


def measure_latency(model, processor, res: int) -> float:
    """Measure encoder latency using dummy images."""
    img = make_dummy_image(res)
    pixel_values = prepare_input(processor, img, res)

    # Warmup
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = encode_image(model, pixel_values)
            if DEVICE == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(N_RUNS):
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = encode_image(model, pixel_values)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()
            latencies.append((t1 - t0) * 1000.0)

    return sum(latencies) / len(latencies)


# ---------------- MAIN TABLE 3 ---------------- #

def build_table3_fastvithd() -> List[Dict[str, Any]]:
    model, processor = load_fastvithd()

    rows: List[Dict[str, Any]] = []

    print(f"\n🚀 Table 3 — FastViT-HD Encoder Benchmarks\n")

    for res in RESOLUTIONS:
        print(f"=== Resolution: {res} x {res} ===")

        try:
            tokens = estimate_tokens(model, processor, res)
            avg_lat = measure_latency(model, processor, res)

            print(f"  ➜ Tokens: {tokens}, Avg Latency: {avg_lat:.2f} ms\n")

            rows.append({
                "model": "FastViT-HD",
                "resolution": res,
                "tokens": tokens,
                "avg_latency_ms": avg_lat,
            })

        except Exception as e:
            print(f"  ❌ Error at resolution {res}: {e}\n")
            rows.append({
                "model": "FastViT-HD",
                "resolution": res,
                "tokens": None,
                "avg_latency_ms": None,
                "notes": f"Error: {e}",
            })

    return rows


def print_markdown_table(rows: List[Dict[str, Any]]):
    print("\n\n### Table 3 — FastViT-HD Encoder Benchmarks\n")
    print("| Model | Resolution | Visual Tokens | Avg Encoder Latency (ms) |")
    print("|-------|-----------:|--------------:|--------------------------:|")

    for r in rows:
        tok = "-" if r["tokens"] is None or r["tokens"] == -1 else str(r["tokens"])
        lat = "-" if r["avg_latency_ms"] is None else f"{r['avg_latency_ms']:.2f}"
        print(f"| {r['model']} | {r['resolution']} | {tok} | {lat} |")


if __name__ == "__main__":
    rows = build_table3_fastvithd()
    print_markdown_table(rows)
