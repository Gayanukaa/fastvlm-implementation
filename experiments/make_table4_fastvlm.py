"""
make_table4_fastvlm.py

Replicate a Table-4-style encoder benchmark using:
- Encoders: ConvNeXt-L, FastViT-HD
- Resolution: 256 x 256
- Datasets: TextVQA, DocVQA (from utils_datasets.get_benchmark_dataset)

Metrics:
- Encoder size (visual tokens) at 256x256
- Average encoding latency per image (ms) on each dataset
"""

import time
from typing import List, Dict, Any

import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import convnext_large
from transformers import AutoModel

from utils_dataset import get_benchmark_dataset

# Path where your encoder weights are stored (same as in utils_encoder_models.py)
MODEL_PATH = "encoder_models/"

# ---------------- CONFIG ---------------- #

ENCODERS = [
    ("convnext", "ConvNeXt-L"),
    ("fastvit", "FastViT-HD"),
]

BENCHMARKS = [
    ("textvqa", "TextVQA"),
    ("docvqa", "DocVQA"),
]

INPUT_RES = 256           # final resolution given to encoder
MAX_SAMPLES = 10          # you can increase later if you want
N_WARMUP = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# ---------------- UTILITIES ---------------- #

def make_preprocess_256() -> transforms.Compose:
    """Common preprocessing: resize to 256x256 and convert to tensor (0-1)."""
    return transforms.Compose([
        transforms.Resize((INPUT_RES, INPUT_RES)),
        transforms.ToTensor(),  # [0,1]
    ])


def load_convnext_encoder() -> Any:
    """Load ConvNeXt-L with your local checkpoint, using 256x256 input."""
    print("🔹 Loading ConvNeXt-L encoder (local weights)...")
    model = convnext_large(weights=None)
    weights = torch.load(MODEL_PATH + "convnext_large.pt", map_location="cpu")
    model.load_state_dict(weights)
    model.eval()
    preprocess = make_preprocess_256()
    return model, preprocess


def load_fastvit_encoder() -> Any:
    """
    Load FastViT-HD from HuggingFace.
    We don't apply your local fastvithd.pt here to avoid mismatch issues.
    """
    print("🔹 Loading FastViT-HD encoder (HuggingFace kevin510/fast-vit-hd)...")
    model = AutoModel.from_pretrained(
        "kevin510/fast-vit-hd",
        trust_remote_code=True
    ).eval()
    preprocess = make_preprocess_256()
    return model, preprocess


def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
    """
    Unified forward for all encoders:
    - CLIP-like models: use encode_image()
    - Others: call model(x)
    """
    if hasattr(model, "encode_image"):
        return model.encode_image(x)
    return model(x)


def make_dummy_image() -> Image.Image:
    """Create dummy image for token estimation."""
    return Image.new("RGB", (INPUT_RES, INPUT_RES), color=(128, 128, 128))


def estimate_tokens_for_encoder(
    encoder_name: str,
    model: torch.nn.Module,
    preprocess,
) -> int:
    """Estimate number of visual tokens at 256x256 resolution."""

    encoder_name = encoder_name.lower()

    if encoder_name == "convnext":
        # ConvNeXt downsamples by 32x → (256/32)^2 = 8x8 = 64 tokens
        return (INPUT_RES // 32) ** 2

    if encoder_name == "fastvit":
        # Infer dynamically from one forward pass
        dummy = make_dummy_image()
        inp = preprocess(dummy).unsqueeze(0).to(DEVICE, dtype=DTYPE)
        with torch.no_grad():
            out = encode_image_generic(model, inp)

        # Handle typical shapes / HF outputs
        if hasattr(out, "last_hidden_state"):
            return int(out.last_hidden_state.shape[1])

        if isinstance(out, dict) and "last_hidden_state" in out:
            return int(out["last_hidden_state"].shape[1])

        if hasattr(out, "shape") and out.ndim >= 3:
            return int(out.shape[1])

        if isinstance(out, tuple) and len(out) > 0 and hasattr(out[0], "shape"):
            return int(out[0].shape[1])

        return -1  # unknown

    return -1


def measure_latency_on_dataset(
    model: torch.nn.Module,
    preprocess,
    benchmark_name: str,
    split: str = "validation",
    max_samples: int = MAX_SAMPLES,
) -> Dict[str, Any]:
    """
    Measure average latency per sample on a given dataset:
    - Uses get_benchmark_dataset to fetch images.
    - Only uses the 'image' field (ignores question/answers).
    """
    print(f"\n📚 Loading dataset: {benchmark_name} (split={split}, max_samples={max_samples})")
    samples = get_benchmark_dataset(benchmark_name, split, max_samples)
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
        for i, sample in enumerate(samples):
            img = sample["image"]
            inp = preprocess(img).unsqueeze(0).to(DEVICE, dtype=DTYPE)

            if DEVICE == "cuda":
                torch.cuda.synchronize()
            start = time.time()
            _ = encode_image_generic(model, inp)
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            end = time.time()

            latencies.append((end - start) * 1000.0)  # ms

    avg_latency = sum(latencies) / len(latencies)
    print(f"   ➜ Avg latency: {avg_latency:.2f} ms over {len(samples)} samples.")
    return {"avg_latency_ms": avg_latency, "num_samples": len(samples)}


def benchmark_table4() -> List[Dict[str, Any]]:
    results = []

    print(f"🚀 Running Table-4-style benchmark on device={DEVICE}, dtype={DTYPE}, res={INPUT_RES}")

    # Load both encoders once
    encoder_models = {}
    encoder_preprocs = {}
    encoder_tokens = {}

    for name, disp in ENCODERS:
        try:
            if name == "convnext":
                model, preprocess = load_convnext_encoder()
            elif name == "fastvit":
                model, preprocess = load_fastvit_encoder()
            else:
                raise ValueError(f"Unknown encoder '{name}'")

            model.to(DEVICE).to(dtype=DTYPE)
            encoder_models[name] = model
            encoder_preprocs[name] = preprocess

            tokens = estimate_tokens_for_encoder(name, model, preprocess)
            encoder_tokens[name] = tokens

            print(f"✅ {disp}: encoder_size_tokens={tokens}")
        except Exception as e:
            print(f"❌ Failed to load {disp}: {e}")
            encoder_models[name] = None
            encoder_preprocs[name] = None
            encoder_tokens[name] = None

    # For each encoder + dataset
    for enc_name, enc_disp in ENCODERS:
        model = encoder_models[enc_name]
        preprocess = encoder_preprocs[enc_name]
        tokens = encoder_tokens[enc_name]

        if model is None or preprocess is None:
            for ds_name, ds_disp in BENCHMARKS:
                results.append({
                    "encoder": enc_disp,
                    "dataset": ds_disp,
                    "resolution": INPUT_RES,
                    "tokens": tokens,
                    "avg_latency_ms": None,
                    "num_samples": 0,
                    "notes": "Encoder load failed",
                })
            continue

        for ds_name, ds_disp in BENCHMARKS:
            try:
                stats = measure_latency_on_dataset(
                    model,
                    preprocess,
                    benchmark_name=ds_name,
                    split="validation",
                    max_samples=MAX_SAMPLES,
                )

                results.append({
                    "encoder": enc_disp,
                    "dataset": ds_disp,
                    "resolution": INPUT_RES,
                    "tokens": tokens,
                    "avg_latency_ms": stats["avg_latency_ms"],
                    "num_samples": stats["num_samples"],
                    "notes": "OK" if stats["avg_latency_ms"] is not None else "No samples",
                })
            except Exception as e:
                print(f"❌ Error during {enc_disp} on {ds_disp}: {e}")
                results.append({
                    "encoder": enc_disp,
                    "dataset": ds_disp,
                    "resolution": INPUT_RES,
                    "tokens": tokens,
                    "avg_latency_ms": None,
                    "num_samples": 0,
                    "notes": f"Error: {e}",
                })

    return results


def print_markdown_table(results: List[Dict[str, Any]]):
    print("\n\n### Table 4 (Replication) — Encoder Latency on TextVQA & DocVQA (256x256)\n")
    print("| Encoder | Dataset | Resolution | Encoder Size (tokens) | Avg Latency (ms) | #Samples | Notes |")
    print("|---------|---------|-----------:|----------------------:|-----------------:|---------:|-------|")

    for r in results:
        lat = "-" if r["avg_latency_ms"] is None else f"{r['avg_latency_ms']:.2f}"
        tok = "-" if r["tokens"] is None or r["tokens"] == -1 else str(r["tokens"])
        print(
            f"| {r['encoder']} "
            f"| {r['dataset']} "
            f"| {r['resolution']} "
            f"| {tok} "
            f"| {lat} "
            f"| {r['num_samples']} "
            f"| {r['notes']} |"
        )


if __name__ == "__main__":
    results = benchmark_table4()
    print_markdown_table(results)
