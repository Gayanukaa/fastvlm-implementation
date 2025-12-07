"""
exp_table11.py

Mini Table 11 Replication (FastVLM 0.5B vs 1.5B)

Same output format + saving functions as Table 5:
- print_markdown_table(rows)
- save_table_as_image(rows)

Columns:
| Model | Params | Resolution | Visual Tokens | TextVQA Acc (%) | DocVQA Acc (%) | Avg Latency (ms) |
"""

import os
import sys
import time
from typing import Dict, List, Any

import torch
from PIL import Image
from tqdm import tqdm

# Add parent directory for llava imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path
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

RESOLUTIONS = [1024, 2048]

# Your custom visual token mapping
CUSTOM_VISUAL_TOKENS = {
    1024: 256,
    2048: 1280,
}

MAX_SAMPLES = 100
N_WARMUP = 2


# ---------------- UTILS ---------------- #

def normalize_answer(s: str) -> str:
    import re, string
    s = s.lower().strip()
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def vqa_accuracy(pred: str, gts: List[str]) -> float:
    pred_norm = normalize_answer(pred)
    for gt in gts:
        if normalize_answer(gt) == pred_norm:
            return 1.0
    return 0.0


def load_vlm_model(path: str):
    disable_torch_init()
    model_name = get_model_name_from_path(path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        path, None, model_name, device=DEVICE
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return tokenizer, model, image_processor


def resize_image(img: Image.Image, res: int):
    return img.resize((res, res), Image.LANCZOS)


def run_inference(model, tokenizer, image_processor, img, question, res: int):
    img_resized = resize_image(img, res)

    qs = DEFAULT_IMAGE_TOKEN + "\n" + question
    conv = conv_templates[CONV_MODE].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors="pt").unsqueeze(0).to(DEVICE)

    img_tensor = process_images([img_resized], image_processor, model.config)[0]

    torch.cuda.synchronize() if DEVICE == "cuda" else None
    start = time.time()

    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            images=img_tensor.unsqueeze(0).to(dtype=DTYPE),
            image_sizes=[img_resized.size],
            max_new_tokens=64,
            do_sample=False
        )

    torch.cuda.synchronize() if DEVICE == "cuda" else None

    latency = (time.time() - start) * 1000
    ans = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    return ans, latency


def evaluate_dataset(model, tokenizer, image_processor, samples, res: int):
    if len(samples) == 0:
        return None, None

    # Warmup
    for _ in range(N_WARMUP):
        _ = run_inference(model, tokenizer, image_processor,
                          samples[0]["image"], samples[0]["question"], res)

    acc = 0.0
    latencies = []

    for s in tqdm(samples, desc=f"Eval@{res}px"):
        try:
            pred, lt = run_inference(model, tokenizer, image_processor,
                                     s["image"], s["question"], res)
            acc += vqa_accuracy(pred, s["answers"])
            latencies.append(lt)
        except:
            continue

    accuracy = (acc / len(latencies)) * 100 if latencies else None
    avg_latency = sum(latencies) / len(latencies) if latencies else None
    return accuracy, avg_latency


# ---------------- MAIN TABLE 11 ---------------- #

def build_table11():
    rows = []

    print("\n📚 Loading datasets...")
    textvqa = get_benchmark_dataset("textvqa", "validation", MAX_SAMPLES)
    docvqa = get_benchmark_dataset("docvqa", "validation", MAX_SAMPLES)

    for model_name, ckpt in MODELS:
        print(f"\n🔹 Loading {model_name} from {ckpt}")
        tokenizer, model, image_processor = load_vlm_model(ckpt)
        params_m = sum(p.numel() for p in model.parameters()) / 1e6

        for res in RESOLUTIONS:
            tokens = CUSTOM_VISUAL_TOKENS[res]

            print(f"\n===== Resolution {res}px ({tokens} tokens) =====")

            txt_acc, txt_lat = evaluate_dataset(
                model, tokenizer, image_processor, textvqa, res
            )

            doc_acc, doc_lat = evaluate_dataset(
                model, tokenizer, image_processor, docvqa, res
            )

            rows.append({
                "model": model_name,
                "params_m": params_m,
                "resolution": res,
                "tokens": tokens,
                "textvqa_acc": txt_acc,
                "docvqa_acc": doc_acc,
                "avg_latency": txt_lat,   # using TextVQA latency consistently
            })

        del model, tokenizer
        torch.cuda.empty_cache()

    return rows


# ---------------- OUTPUT (Same as Table 5) ---------------- #

def print_markdown_table(rows):
    print("\n" + "="*80)
    print("### Table 11 (Replication) — FastVLM Model Comparison")
    print("="*80 + "\n")

    print("| Model | Params (M) | Resolution | Visual Tokens | TextVQA Acc (%) | DocVQA Acc (%) | Avg Latency (ms) |")
    print("|-------|------------|------------|----------------|------------------|----------------|------------------|")

    for r in rows:
        textvqa = '-' if r['textvqa_acc'] is None else f"{r['textvqa_acc']:.1f}"
        docvqa = '-' if r['docvqa_acc'] is None else f"{r['docvqa_acc']:.1f}"
        avglat = '-' if r['avg_latency'] is None else f"{r['avg_latency']:.1f}"
        print(
            f"| {r['model']} | {r['params_m']:.1f} | {r['resolution']} | {r['tokens']} | {textvqa} | {docvqa} | {avglat} |"
        )


def save_table_as_image(rows):
    headers = ["Model", "Params (M)", "Resolution", "Visual Tokens",
               "TextVQA Acc (%)", "DocVQA Acc (%)", "Avg Latency (ms)"]

    table_rows = []
    for r in rows:
        table_rows.append([
            r["model"],
            f"{r['params_m']:.1f}",
            str(r["resolution"]),
            str(r["tokens"]),
            "-" if r["textvqa_acc"] is None else f"{r['textvqa_acc']:.1f}",
            "-" if r["docvqa_acc"] is None else f"{r['docvqa_acc']:.1f}",
            "-" if r["avg_latency"] is None else f"{r['avg_latency']:.1f}",
        ])

    save_table_image(
        headers,
        table_rows,
        "table11_fastvlm_comparison.png",
        title="Table 11: FastVLM Model Comparison"
    )


# ---------------- ENTRY POINT ---------------- #

if __name__ == "__main__":
    rows = build_table11()
    print_markdown_table(rows)
    save_table_as_image(rows)

