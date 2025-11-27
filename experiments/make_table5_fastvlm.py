"""
make_table5_fastvithd_textvqa_encoder.py

Mini Table 5 (Encoder-only, using TextVQA images):

For FastViT-HD, measure for each resolution:
- Visual token count (from encoder output)
- Average encoder latency (ms) on TextVQA images

Uses:
- FastViT-HD from HuggingFace: "kevin510/fast-vit-hd"
- TextVQA from utils_datasets.get_benchmark_dataset()

Output:

| Model | Resolution | Visual Tokens | Avg Encoder Latency (ms) | #Samples | Notes |
"""

import time
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoModel, AutoImageProcessor

from utils_dataset import get_benchmark_dataset


# ---------------- CONFIG ---------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Resolutions to evaluate
RESOLUTIONS = [256, 512, ]

MAX_SAMPLES = 10       # number of TextVQA samples per resolution
N_WARMUP = 5           # warmup runs per resolution


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

def prepare_input(processor, img: Image.Image, res: int) -> torch.Tensor:
    """
    Resize the PIL image to res x res ourselves,
    then call processor with resizing disabled so the resolution is preserved.
    """
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


def estimate_tokens(model, processor, res: int, sample_img: Image.Image) -> int:
    """
    Estimate number of visual tokens for a given resolution using one real TextVQA image.
    """
    pixel_values = prepare_input(processor, sample_img, res)
    with torch.no_grad():
        out = encode_image(model, pixel_values)

    # Try HF-style outputs
    if hasattr(out, "last_hidden_state"):
        return int(out.last_hidden_state.shape[1])
    if isinstance(out, dict) and "last_hidden_state" in out:
        return int(out["last_hidden_state"].shape[1])

    if hasattr(out, "shape") and out.ndim >= 3:
        return int(out.shape[1])
    if isinstance(out, tuple) and len(out) > 0 and hasattr(out[0], "shape"):
        return int(out[0].shape[1])

    return -1


def measure_latency_on_textvqa(
    model,
    processor,
    res: int,
    samples: List[Dict[str, Any]],
) -> float:
    """
    Measure average encoder latency on TextVQA images for given resolution.
    """
    latencies = []

    # Warmup using first image
    if len(samples) > 0:
        img0 = samples[0]["image"]
        pv0 = prepare_input(processor, img0, res)
        with torch.no_grad():
            for _ in range(N_WARMUP):
                _ = encode_image(model, pv0)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()

    # Timed runs
    with torch.no_grad():
        for s in samples:
            img = s["image"]
            if not isinstance(img, Image.Image):
                continue

            pixel_values = prepare_input(processor, img, res)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t0 = time.time()
            _ = encode_image(model, pixel_values)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t1 = time.time()

            lat_ms = (t1 - t0) * 1000.0
            latencies.append(lat_ms)

    if not latencies:
        return None

    return sum(latencies) / len(latencies)


# ---------------- MAIN TABLE 5 (ENCODER + TEXTVQA) ---------------- #

def build_table5_encoder_textvqa() -> List[Dict[str, Any]]:
    model, processor = load_fastvithd()

    print(f"\n📚 Loading TextVQA samples (max={MAX_SAMPLES})...")
    samples = get_benchmark_dataset("textvqa", "validation", MAX_SAMPLES)
    if len(samples) == 0:
        print("⚠️ No TextVQA samples loaded.")
        return []

    print(f"✅ Loaded {len(samples)} TextVQA samples.")

    rows: List[Dict[str, Any]] = []

    # Use the first sample image for token estimation
    sample_img = samples[0]["image"]
    if not isinstance(sample_img, Image.Image):
        raise RuntimeError("First TextVQA sample image is not a PIL.Image.Image")

    print(f"\n🚀 Mini Table 5 (Encoder-only) — FastViT-HD on TextVQA images\n")

    for res in RESOLUTIONS:
        print(f"=== Resolution: {res} x {res} ===")

        try:
            tokens = estimate_tokens(model, processor, res, sample_img)
            avg_lat = measure_latency_on_textvqa(model, processor, res, samples)

            print(f"  ➜ Visual tokens: {tokens}, Avg encoder latency: {avg_lat:.2f} ms\n")

            rows.append({
                "model": "FastViT-HD",
                "resolution": res,
                "tokens": tokens,
                "avg_latency_ms": avg_lat,
                "num_samples": len(samples),
                "notes": "Encoder-only, TextVQA images",
            })
        except Exception as e:
            print(f"  ❌ Error at resolution {res}: {e}\n")
            rows.append({
                "model": "FastViT-HD",
                "resolution": res,
                "tokens": None,
                "avg_latency_ms": None,
                "num_samples": len(samples),
                "notes": f"Error: {e}",
            })

    return rows


def print_markdown_table(rows: List[Dict[str, Any]]):
    print("\n\n### Mini-Table-5 (Encoder-only, Replication) — FastViT-HD on TextVQA\n")
    print("| Model | Resolution | Visual Tokens | Avg Encoder Latency (ms) | #Samples | Notes |")
    print("|-------|-----------:|--------------:|--------------------------:|---------:|-------|")

    for r in rows:
        tok = "-" if r["tokens"] is None or r["tokens"] == -1 else str(r["tokens"])
        lat = "-" if r["avg_latency_ms"] is None else f"{r['avg_latency_ms']:.2f}"
        print(
            f"| {r['model']} "
            f"| {r['resolution']} "
            f"| {tok} "
            f"| {lat} "
            f"| {r['num_samples']} "
            f"| {r['notes']} |"
        )


limitations_msg = """
[Mini Table 5 – Replication Limitations]

This experiment reproduces only the encoder-level behavior of FastViT-HD, not
the full Table 5 from the FastVLM paper.

1) No pruning baselines:
   The original Table 5 compares FastViT-HD to several token pruning /
   sparsification methods from other papers. Those baseline results are
   reported directly from the respective papers and were not re-trained by
   the FastVLM authors. In this replication, we do not re-implement or
   re-train those pruning methods, so we only report results for FastViT-HD.

2) Encoder-only setup:
   The original Table 5 is based on full VLM models trained with the
   LLaVA-1.5 setup and Vicuna-7B. Here, we evaluate only the FastViT-HD
   vision encoder (token count and encoder latency). We do not run the
   full vision-language model with Vicuna-7B, so we cannot reproduce the
   exact end-to-end benchmark scores.

3) Hardware and data limitations:
   The original paper uses its own hardware configuration and full
   benchmark datasets. Our replication is run on a single RTX 4070 8 GB
   GPU and, where applicable, only a small subset of dataset samples
   (e.g., a limited number of TextVQA images). Therefore, our absolute
   latency numbers may differ from the paper, and our results should be
   interpreted as a qualitative replication of the trends, not as an exact
   reproduction of the reported scores.
"""




if __name__ == "__main__":
    rows = build_table5_encoder_textvqa()
    print_markdown_table(rows)


    print("----------------------------------------------------------------------------------")
    print(limitations_msg)




