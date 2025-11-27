# # """
# # make_table3_fastvlm.py

# # Replicates a Table-3-style encoder benchmark for FastVLM using your local encoders:
# # - ConvNeXt-L
# # - ViT-L/14 (OpenCLIP)
# # - FastViT-HD

# # What it measures (on *your* hardware, e.g., Windows + RTX 4070 8GB):
# # - #Params (millions)
# # - Input resolution used by the encoder
# # - Average image encoding latency (ms per image)

# # NOTE:
# # - Original Table 3 also has CLIP retrieval / 38-task averages.
# #   Here we only measure things you can do locally: params + latency.
# #   You can later extend this script to add retrieval metrics if needed.
# # """

# # import time
# # from typing import Dict, Any, List

# # import torch
# # from PIL import Image

# # from utils_encoder_models import load_encoder


# # # ---------- CONFIG ----------

# # # Encoders to benchmark for "Table 3"
# # # name: (internal_name_for_load_encoder, display_name, input_resolution)
# # ENCODER_CONFIGS = [
# #     ("convnext", "ConvNeXt-L", 224),
# #     ("vit", "ViT-L/14 (OpenCLIP)", 224),
# #     ("fastvit", "FastViT-HD", 224),
# # ]

# # # Number of warmup + timed runs for latency measurement
# # N_WARMUP = 5
# # N_RUNS = 20

# # DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# # DTYPE = torch.float16 if (DEVICE == "cuda" and torch.cuda.is_bf16_supported() is False) else torch.float32
# # # (You can change DTYPE to torch.float16 for faster timing once things work)


# # # ---------- UTILS ----------

# # def count_parameters(model: torch.nn.Module) -> float:
# #     """Return total parameters in millions."""
# #     return sum(p.numel() for p in model.parameters()) / 1e6


# # def make_dummy_image(res: int) -> Image.Image:
# #     """Create a dummy PIL image with given resolution (RGB)."""
# #     return Image.new("RGB", (res, res), color=(128, 128, 128))


# # def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
# #     """
# #     Generic "encode image" call.

# #     For ConvNeXt & vanilla torchvision models: model(x) is fine.
# #     For OpenCLIP: we should call model.encode_image(x).
# #     For FastViT-HD: depends on implementation; we try a few common patterns.

# #     Adjust this to match your actual project if needed.
# #     """
# #     # OpenCLIP CLIP models typically have `encode_image`
# #     if hasattr(model, "encode_image"):
# #         return model.encode_image(x)

# #     # HuggingFace vision models often accept pixel_values=...
# #     try:
# #         return model(pixel_values=x)
# #     except TypeError:
# #         pass

# #     # Fallback: standard forward
# #     return model(x)


# # def measure_latency(
# #     model: torch.nn.Module,
# #     preprocess,
# #     input_res: int,
# #     n_warmup: int = N_WARMUP,
# #     n_runs: int = N_RUNS,
# # ) -> float:
# #     """
# #     Measure average encode latency (ms/image) for a given encoder.
# #     """
# #     model.eval()
# #     model.to(DEVICE)

# #     # Create one dummy image and preprocess it
# #     dummy_img = make_dummy_image(input_res)
# #     with torch.no_grad():
# #         inp = preprocess(dummy_img).unsqueeze(0)  # (1, C, H, W)
# #         inp = inp.to(DEVICE, dtype=DTYPE)

# #     # Warmup
# #     with torch.no_grad():
# #         for _ in range(n_warmup):
# #             _ = encode_image_generic(model, inp)
# #             if DEVICE == "cuda":
# #                 torch.cuda.synchronize()

# #     # Timed runs
# #     total_time = 0.0
# #     with torch.no_grad():
# #         for _ in range(n_runs):
# #             if DEVICE == "cuda":
# #                 torch.cuda.synchronize()
# #             start = time.time()
# #             _ = encode_image_generic(model, inp)
# #             if DEVICE == "cuda":
# #                 torch.cuda.synchronize()
# #             end = time.time()
# #             total_time += (end - start)

# #     avg_time_ms = (total_time / n_runs) * 1000.0
# #     return avg_time_ms


# # def benchmark_encoders() -> List[Dict[str, Any]]:
# #     results = []

# #     print(f"🚀 Running Table-3-style encoder benchmark on device: {DEVICE} (dtype={DTYPE})")
# #     print(f"   Warmup: {N_WARMUP} runs, Timed: {N_RUNS} runs per encoder\n")

# #     for internal_name, display_name, res in ENCODER_CONFIGS:
# #         print(f"===== {display_name} (load name: '{internal_name}', res={res}) =====")
# #         try:
# #             model, preprocess = load_encoder(internal_name)
# #             # Put model to correct device & dtype once
# #             model.to(DEVICE)
# #             model = model.to(dtype=DTYPE)

# #             num_params_m = count_parameters(model)
# #             latency_ms = measure_latency(model, preprocess, res)

# #             results.append({
# #                 "encoder": display_name,
# #                 "internal_name": internal_name,
# #                 "resolution": res,
# #                 "params_m": num_params_m,
# #                 "latency_ms": latency_ms,
# #             })

# #             print(f"  ➜ Params: {num_params_m:.1f}M | Latency: {latency_ms:.2f} ms\n")
# #         except Exception as e:
# #             print(f"  ❌ Error benchmarking {display_name}: {e}\n")
# #             results.append({
# #                 "encoder": display_name,
# #                 "internal_name": internal_name,
# #                 "resolution": res,
# #                 "params_m": None,
# #                 "latency_ms": None,
# #                 "error": str(e),
# #             })

# #     return results


# # def print_markdown_table(results: List[Dict[str, Any]]):
# #     """
# #     Print a Markdown table you can paste into your report as "Table 3 (Replication)".
# #     """
# #     print("\n\n### Table 3 (Replication) – Encoder Latency on Your Hardware\n")
# #     print("| Encoder | Resolution | Params (M) | Latency Enc. (ms) | Notes |")
# #     print("|---------|-----------:|----------:|-------------------:|-------|")

# #     for r in results:
# #         if r.get("latency_ms") is None:
# #             note = f"Error: {r.get('error', 'N/A')}"
# #             print(f"| {r['encoder']} | {r['resolution']} | - | - | {note} |")
# #         else:
# #             print(
# #                 f"| {r['encoder']} "
# #                 f"| {r['resolution']} "
# #                 f"| {r['params_m']:.1f} "
# #                 f"| {r['latency_ms']:.2f} "
# #                 f"| Measured on {DEVICE} |"
# #             )


# # if __name__ == "__main__":
# #     results = benchmark_encoders()
# #     print_markdown_table(results)


# """
# make_table3_fastvlm.py

# Replicates a Table-3-style encoder benchmark for FastVLM using your local encoders:
# - ConvNeXt-L
# - ViT-L/14 (OpenCLIP)
# - FastViT-HD (loaded directly from Hugging Face)

# Metrics:
# - Encoder Size (≈ #visual tokens)
# - Params (M)
# - Input resolution
# - Average encoding latency (ms / image) on YOUR GPU
# """

# import time
# from typing import Dict, Any, List

# import torch
# from PIL import Image

# from utils_encoder_models import load_encoder
# from transformers import AutoModel, AutoImageProcessor


# # ---------- CONFIG ----------

# # name: (internal_name, display_name, resolution)
# ENCODER_CONFIGS = [
#     ("convnext", "ConvNeXt-L", 224),
#     ("vit", "ViT-L/14 (OpenCLIP)", 224),
#     ("fastvit", "FastViT-HD", 224),
# ]

# N_WARMUP = 5
# N_RUNS = 20

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


# # ---------- UTILS ----------

# def count_parameters(model: torch.nn.Module) -> float:
#     """Return total parameters in millions."""
#     return sum(p.numel() for p in model.parameters()) / 1e6


# def make_dummy_image(res: int) -> Image.Image:
#     """Create a dummy PIL image with given resolution (RGB)."""
#     return Image.new("RGB", (res, res), color=(128, 128, 128))


# def load_fastvit_direct():
#     """
#     Load FastViT-HD directly from Hugging Face in the 'official' way,
#     without using your local fastvithd.pt checkpoint.
#     """
#     print("🔹 [HF] Loading FastViT-HD encoder from 'kevin510/fast-vit-hd' ...")

#     model = AutoModel.from_pretrained(
#         "kevin510/fast-vit-hd",
#         trust_remote_code=True
#     ).eval()

#     processor = AutoImageProcessor.from_pretrained(
#         "kevin510/fast-vit-hd",
#         trust_remote_code=True
#     )

#     # Wrap processor into a torchvision-like preprocess function
#     def preprocess(img: Image.Image) -> torch.Tensor:
#         batch = processor(
#             img,
#             do_center_crop=False,
#             return_tensors="pt"
#         )
#         # batch["pixel_values"]: (1, 3, H, W) → we return (3, H, W)
#         return batch["pixel_values"][0]

#     return model, preprocess


# def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
#     """
#     Generic 'encode image' call that works for:
#     - FastViT-HD HF wrapper: model(pixel_values=x)
#     - OpenCLIP: model.encode_image(x)
#     - ConvNeXt / torchvision: model(x)
#     """
#     # 1) Try HuggingFace-style call first (FastViT-HD)
#     try:
#         return model(pixel_values=x)
#     except TypeError:
#         pass

#     # 2) CLIP-style encoders
#     if hasattr(model, "encode_image"):
#         return model.encode_image(x)

#     # 3) Fallback: plain forward
#     return model(x)


# def estimate_tokens(
#     encoder_name: str,
#     model: torch.nn.Module,
#     preprocess,
#     input_res: int,
# ) -> int:
#     """
#     Estimate encoder size as the number of visual tokens.

#     - For ConvNeXt-L: approximate tokens ≈ (res / 32)^2  (final 7x7 at 224)
#     - For ViT-L/14: tokens = (res / 14)^2
#     - For FastViT-HD: infer from model output shape.
#     """
#     encoder_name = encoder_name.lower()

#     # Analytic approximations for convnext / vit
#     if encoder_name == "convnext":
#         # ConvNeXt downsamples by factor 32 → 224/32 = 7, tokens = 7*7 = 49
#         tokens = (input_res // 32) ** 2
#         return tokens

#     if encoder_name == "vit":
#         # ViT-L/14: patch size 14
#         patch = 14
#         tokens = (input_res // patch) ** 2
#         return tokens

#     # For FastViT-HD, try to infer dynamically
#     if encoder_name == "fastvit":
#         model.eval()
#         model.to(DEVICE)

#         dummy_img = make_dummy_image(input_res)
#         with torch.no_grad():
#             inp = preprocess(dummy_img).unsqueeze(0)  # (1, C, H, W)
#             inp = inp.to(DEVICE, dtype=DTYPE)

#             out = model(pixel_values=inp)

#             # Common HF outputs
#             if hasattr(out, "last_hidden_state"):
#                 # shape: (B, T, D)
#                 return int(out.last_hidden_state.shape[1])
#             if isinstance(out, dict) and "last_hidden_state" in out:
#                 return int(out["last_hidden_state"].shape[1])
#             if isinstance(out, tuple) and len(out) > 0:
#                 # assume first element is [B, T, D] or [B, T]
#                 t = out[0]
#                 if hasattr(t, "shape") and t.ndim >= 2:
#                     return int(t.shape[1])

#         # Fallback if we can't infer
#         return -1

#     # Unknown encoder type
#     return -1


# def measure_latency(
#     model: torch.nn.Module,
#     preprocess,
#     input_res: int,
#     n_warmup: int = N_WARMUP,
#     n_runs: int = N_RUNS,
# ) -> float:
#     """
#     Measure average encode latency (ms/image) for a given encoder.
#     """
#     model.eval()
#     model.to(DEVICE)

#     dummy_img = make_dummy_image(input_res)
#     with torch.no_grad():
#         inp = preprocess(dummy_img).unsqueeze(0)  # (1, C, H, W)
#         inp = inp.to(DEVICE, dtype=DTYPE)

#     # Warmup
#     with torch.no_grad():
#         for _ in range(n_warmup):
#             _ = encode_image_generic(model, inp)
#             if DEVICE == "cuda":
#                 torch.cuda.synchronize()

#     # Timed runs
#     total_time = 0.0
#     with torch.no_grad():
#         for _ in range(n_runs):
#             if DEVICE == "cuda":
#                 torch.cuda.synchronize()
#             start = time.time()
#             _ = encode_image_generic(model, inp)
#             if DEVICE == "cuda":
#                 torch.cuda.synchronize()
#             end = time.time()
#             total_time += (end - start)

#     avg_time_ms = (total_time / n_runs) * 1000.0
#     return avg_time_ms


# def benchmark_encoders() -> List[Dict[str, Any]]:
#     results = []

#     print(f"🚀 Running Table-3-style encoder benchmark on device: {DEVICE} (dtype={DTYPE})")
#     print(f"   Warmup: {N_WARMUP} runs, Timed: {N_RUNS} runs per encoder\n")

#     for internal_name, display_name, res in ENCODER_CONFIGS:
#         print(f"===== {display_name} (load name: '{internal_name}', res={res}) =====")
#         try:
#             # Special-case FastViT-HD: load directly via HF
#             if internal_name == "fastvit":
#                 model, preprocess = load_fastvit_direct()
#             else:
#                 model, preprocess = load_encoder(internal_name)

#             model.to(DEVICE)
#             model = model.to(dtype=DTYPE)

#             num_params_m = count_parameters(model)
#             tokens = estimate_tokens(internal_name, model, preprocess, res)
#             latency_ms = measure_latency(model, preprocess, res)

#             results.append({
#                 "encoder": display_name,
#                 "internal_name": internal_name,
#                 "resolution": res,
#                 "params_m": num_params_m,
#                 "encoder_size_tokens": tokens,
#                 "latency_ms": latency_ms,
#             })

#             print(
#                 f"  ➜ Params: {num_params_m:.1f}M | "
#                 f"Encoder Size (tokens): {tokens} | "
#                 f"Latency: {latency_ms:.2f} ms\n"
#             )
#         except Exception as e:
#             print(f"  ❌ Error benchmarking {display_name}: {repr(e)}\n")
#             results.append({
#                 "encoder": display_name,
#                 "internal_name": internal_name,
#                 "resolution": res,
#                 "params_m": None,
#                 "encoder_size_tokens": None,
#                 "latency_ms": None,
#                 "error": repr(e),
#             })

#     return results


# def print_markdown_table(results: List[Dict[str, Any]]):
#     """
#     Print a Markdown table you can paste into your report as "Table 3 (Replication)".
#     """
#     print("\n\n### Table 3 (Replication) – Encoder Size & Latency on Your Hardware\n")
#     print("| Encoder | Resolution | Encoder Size (tokens) | Params (M) | Latency Enc. (ms) | Notes |")
#     print("|---------|-----------:|----------------------:|----------:|-------------------:|-------|")

#     for r in results:
#         if r.get("latency_ms") is None:
#             note = f"Error: {r.get('error', 'N/A')}"
#             print(f"| {r['encoder']} | {r['resolution']} | - | - | - | {note} |")
#         else:
#             tokens = r["encoder_size_tokens"]
#             token_str = "-" if tokens is None or tokens < 0 else str(tokens)
#             print(
#                 f"| {r['encoder']} "
#                 f"| {r['resolution']} "
#                 f"| {token_str} "
#                 f"| {r['params_m']:.1f} "
#                 f"| {r['latency_ms']:.2f} "
#                 f"| Measured on {DEVICE} |"
#             )


# if __name__ == "__main__":
#     results = benchmark_encoders()
#     print_markdown_table(results)


"""
make_table3_fastvlm.py

Replicates a Table-3-style encoder benchmark for FastVLM using available models:
- ConvNeXt-L
- ViT-L/14 (OpenCLIP)
- FastViT-HD (HuggingFace)

Outputs:
- Encoder Size (visual tokens)
- Params (M)
- Latency (ms/image)

Paste output table directly into your report as:
"Table 3 — Replication Results (RTX 4070 8GB)"
"""

import time
from typing import Dict, Any, List

import torch
from PIL import Image

from utils_encoder_models import load_encoder
from transformers import AutoModel, AutoImageProcessor


# ---------------- CONFIG ---------------- #

ENCODER_CONFIGS = [
    ("convnext", "ConvNeXt-L", 224),
    ("vit", "ViT-L/14 (OpenCLIP)", 224),
    ("fastvit", "FastViT-HD", 224),
]

N_WARMUP = 5
N_RUNS = 20

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32



# ---------------- UTILITIES ---------------- #

def count_parameters(model: torch.nn.Module) -> float:
    """Return total parameters in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def make_dummy_image(res: int) -> Image.Image:
    """Create a synthetic image for benchmarking (no external dataset needed)."""
    return Image.new("RGB", (res, res), color=(128, 128, 128))


def load_fastvit_direct():
    """Loads FastViT-HD properly from HuggingFace."""
    print("🔹 Loading FastViT-HD from HuggingFace...")

    model = AutoModel.from_pretrained(
        "kevin510/fast-vit-hd",
        trust_remote_code=True
    ).eval()

    processor = AutoImageProcessor.from_pretrained(
        "kevin510/fast-vit-hd",
        trust_remote_code=True
    )

    def preprocess(img: Image.Image) -> torch.Tensor:
        batch = processor(
            img,
            return_tensors="pt"
        )
        return batch["pixel_values"][0]

    return model, preprocess



def encode_image_generic(model: torch.nn.Module, x: torch.Tensor):
    """Unified forward pass for all encoders."""
    # CLIP models use encode_image()
    if hasattr(model, "encode_image"):
        return model.encode_image(x)

    # Otherwise standard forward (FastViT-HD, ConvNeXt)
    return model(x)



def estimate_tokens(
    encoder_name: str,
    model: torch.nn.Module,
    preprocess,
    res: int
):
    """Estimate visual token count for encoder."""

    encoder_name = encoder_name.lower()

    # ConvNeXt → final spatial grid ≈ (res / 32)^2
    if encoder_name == "convnext":
        return (res // 32) ** 2

    # ViT → patch-based grid (res / 14)^2
    if encoder_name == "vit":
        return (res // 14) ** 2

    # FastViT → infer dynamically
    dummy = make_dummy_image(res)
    inp = preprocess(dummy).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out = encode_image_generic(model, inp)

    # Try common HF output structures
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state.shape[1]

    if isinstance(out, dict) and "last_hidden_state" in out:
        return out["last_hidden_state"].shape[1]

    # If tensor or tuple of tensors
    if hasattr(out, "shape") and out.ndim >= 3:
        return out.shape[1]

    if isinstance(out, tuple) and hasattr(out[0], "shape"):
        return out[0].shape[1]

    return -1  # fallback


def measure_latency(
    model, preprocess, res, n_warmup=N_WARMUP, n_runs=N_RUNS
):
    """Benchmark encoding speed."""
    model.eval().to(DEVICE)

    dummy = make_dummy_image(res)
    inp = preprocess(dummy).unsqueeze(0).to(DEVICE, dtype=DTYPE)

    # Warmup
    for _ in range(n_warmup):
        encode_image_generic(model, inp)
        torch.cuda.synchronize()

    # Timed runs
    total = 0
    for _ in range(n_runs):
        torch.cuda.synchronize()
        start = time.time()
        encode_image_generic(model, inp)
        torch.cuda.synchronize()
        total += time.time() - start

    return (total / n_runs) * 1000  # ms



# ---------------- MAIN BENCHMARK ---------------- #

def benchmark():
    results = []

    print(f"\n🚀 Running encoder benchmarking on: {DEVICE} (dtype={DTYPE})\n")

    for name, disp, res in ENCODER_CONFIGS:
        print(f"===== {disp} ({res}px) =====")

        try:
            if name == "fastvit":
                model, preprocess = load_fastvit_direct()
            else:
                model, preprocess = load_encoder(name)

            model.to(DEVICE).to(dtype=DTYPE)

            params = count_parameters(model)
            tokens = estimate_tokens(name, model, preprocess, res)
            latency = measure_latency(model, preprocess, res)

            print(f"  Params: {params:.1f}M | Tokens: {tokens} | Latency: {latency:.2f}ms\n")

            results.append((disp, res, tokens, params, latency, "Measured"))

        except Exception as e:
            print(f"  ❌ Failed: {e}\n")
            results.append((disp, res, "-", "-", "-", f"Error: {e}"))

    return results



def print_table(results):
    print("\n\n### Table 3 (Replication) — Encoder Efficiency on RTX 4070 8GB\n")
    print("| Encoder | Resolution | Encoder Size (tokens) | Params (M) | Latency (ms) | Notes |")
    print("|---------|-----------:|----------------------:|-----------:|-------------:|-------|")

    for enc, res, tok, p, lat, note in results:
        print(f"| {enc} | {res} | {tok} | {p} | {lat} | {note} |")


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    results = benchmark()
    print_table(results)
