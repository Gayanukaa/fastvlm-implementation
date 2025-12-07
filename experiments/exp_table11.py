"""
exp_table11.py

Mini Table 11 Replication (FastVLM 0.5B vs 1.5B)

Same output format style + saving functions as other tables:
- print_markdown_table(rows)
- save_table_as_image(rows)

Columns:
| Model | Resolution | Visual Tokens | TextVQA Acc (%) | DocVQA Acc (%) |
"""

import os
import sys
from typing import List, Dict, Any, Tuple

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory for llava imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

from utils_dataset import get_benchmark_dataset
from utils_plot import save_table_image

# ---------------- CONFIG ---------------- #

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
CONV_MODE = "qwen_2"

MODELS = [
    ("FastVLM-0.5B", "../checkpoints/llava-fastvithd_0.5b_stage3"),
    ("FastVLM-1.5B", "../checkpoints/llava-fastvithd_1.5b_stage3"),
]

# Resolutions & visual tokens (matching paper)
RESOLUTIONS = [1024, 2048]
CUSTOM_VISUAL_TOKENS = {
    1024: 256,
    2048: 1280,
}

# Number of samples per dataset (quick replication)
MAX_SAMPLES = 100
N_WARMUP = 1


# ---------------- UTILITY FUNCTIONS ---------------- #

def normalize_answer(s: str) -> str:
    import re, string
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def vqa_accuracy(pred: str, gts: List[str]) -> float:
    """
    Relaxed VQA-style accuracy:
    - normalize prediction and each ground truth
    - exact match or substring match counts as correct
    """
    pred_norm = normalize_answer(pred)
    if not pred_norm:
        return 0.0

    for gt in gts:
        gt_norm = normalize_answer(gt)
        if not gt_norm:
            continue
        if pred_norm == gt_norm:
            return 1.0
        if pred_norm in gt_norm or gt_norm in pred_norm:
            return 1.0
    return 0.0


def load_vlm_model(path: str):
    """Load VLM checkpoint (tokenizer, model, image_processor)."""
    disable_torch_init()
    model_name = get_model_name_from_path(path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        path, None, model_name, device=DEVICE
    )
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model, image_processor


def resize_image(img: Image.Image, res: int) -> Image.Image:
    img = img.convert("RGB")
    return img.resize((res, res), Image.LANCZOS)


def build_prompt(question: str) -> str:
    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


def run_inference(
    model,
    tokenizer,
    image_processor,
    img: Image.Image,
    question: str,
    res: int,
) -> str:
    """
    Run full VLM inference (no timing).
    Returns: generated answer string.
    """
    img_resized = resize_image(img, res)
    prompt = build_prompt(question)

    # Text tokens
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
    ).unsqueeze(0).to(DEVICE)

    # Image tensor
    img_tensor = process_images([img_resized], image_processor, model.config)[0]
    img_tensor = img_tensor.to(device=DEVICE, dtype=DTYPE)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            images=img_tensor.unsqueeze(0),
            image_sizes=[img_resized.size],
            max_new_tokens=64,
            do_sample=False,
            use_cache=True,
        )

    answer = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return answer


def evaluate_dataset(
    model,
    tokenizer,
    image_processor,
    samples: List[Dict[str, Any]],
    res: int,
) -> float:
    """
    Evaluate a dataset at a specific resolution.
    Returns: accuracy percentage (or None if failed).
    """
    if len(samples) == 0:
        return None

    # Warmup on first sample (no timing)
    for _ in range(min(N_WARMUP, len(samples))):
        _ = run_inference(
            model, tokenizer, image_processor,
            samples[0]["image"], samples[0]["question"], res
        )

    correct = 0.0
    total = 0

    for s in tqdm(samples, desc=f"Eval@{res}px"):
        try:
            pred = run_inference(
                model, tokenizer, image_processor,
                s["image"], s["question"], res
            )
            acc = vqa_accuracy(pred, s["answers"])
            correct += acc
            total += 1
        except Exception:
            continue

    if total == 0:
        return None

    accuracy = (correct / total) * 100.0
    return accuracy


# ---------------- MAIN TABLE 11 LOGIC ---------------- #

def build_table11() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    print("\n📚 Loading datasets...")
    textvqa = get_benchmark_dataset("textvqa", "validation", MAX_SAMPLES)
    docvqa = get_benchmark_dataset("docvqa", "validation", MAX_SAMPLES)

    for model_name, ckpt in MODELS:
        print(f"\n🔹 Loading {model_name} from {ckpt}")
        tokenizer, model, image_processor = load_vlm_model(ckpt)

        for res in RESOLUTIONS:
            tokens = CUSTOM_VISUAL_TOKENS[res]
            print(f"\n===== {model_name} @ {res}px ({tokens} tokens) =====")

            text_acc = evaluate_dataset(
                model, tokenizer, image_processor, textvqa, res
            )
            doc_acc = evaluate_dataset(
                model, tokenizer, image_processor, docvqa, res
            )

            rows.append({
                "model": model_name,
                "resolution": res,
                "tokens": tokens,
                "textvqa_acc": text_acc,
                "docvqa_acc": doc_acc,
            })

        del model, tokenizer
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    return rows


# ---------------- PRINT & SAVE (SAME FORMAT STYLE) ---------------- #

def print_markdown_table(rows: List[Dict[str, Any]]):
    """Print markdown table (style consistent with other experiments)."""
    print("\n" + "=" * 80)
    print("### Table 11 (Replication) — FastVLM Model Comparison")
    print("=" * 80 + "\n")

    print("| Model | Resolution | Visual Tokens | TextVQA Acc (%) | DocVQA Acc (%) |")
    print("|-------|------------|---------------|------------------|----------------|")

    for r in rows:
        textvqa = "-" if r["textvqa_acc"] is None else f"{r['textvqa_acc']:.1f}"
        docvqa = "-" if r["docvqa_acc"] is None else f"{r['docvqa_acc']:.1f}"
        print(
            f"| {r['model']} | {r['resolution']} | {r['tokens']} | "
            f"{textvqa} | {docvqa} |"
        )


def save_table_as_image(rows: List[Dict[str, Any]]):
    """Save results as a LaTeX-style table image (PNG), same style as other tables."""
    headers = [
        "Model",
        "Resolution",
        "Visual Tokens",
        "TextVQA Acc (%)",
        "DocVQA Acc (%)",
    ]
    table_rows = []

    for r in rows:
        table_rows.append([
            r["model"],
            str(r["resolution"]),
            str(r["tokens"]),
            "-" if r["textvqa_acc"] is None else f"{r['textvqa_acc']:.1f}",
            "-" if r["docvqa_acc"] is None else f"{r['docvqa_acc']:.1f}",
        ])

    save_table_image(
        headers,
        table_rows,
        "table11_fastvlm_comparison.png",   # <- same PNG style/flow as others
        title="Table 11: FastVLM Model Comparison",
    )


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    rows = build_table11()
    print_markdown_table(rows)
    save_table_as_image(rows)
