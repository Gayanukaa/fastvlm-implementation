"""
Utility functions for loading FastVLM benchmark datasets.
Supports TextVQA, DocVQA and GQA for benchmarking visual QA performance.
"""

import os
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def get_benchmark_dataset(
    benchmark_name: str = "textvqa",
    split: str = "validation",
    max_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Loads a benchmark dataset and returns a simplified list of samples.

    Args:
        benchmark_name: Name of the dataset (textvqa | docvqa | gqa)
        split: Dataset split to load
        max_samples: Number of samples to load (for quick testing)

    Returns:
        List of dicts formatted as:
        [{'image': PIL.Image, 'question': str, 'id': str, 'answers': List[str]}]
    """

    benchmark_name = benchmark_name.lower()
    print(f"📚 Loading {benchmark_name} ({split}) - Max samples: {max_samples}...")

    formatted_data = []

    try:
        # ======== TextVQA (Standard) ========
        if benchmark_name == "textvqa":
            dataset = load_dataset("textvqa", split=split, streaming=True)

            for idx, sample in tqdm(enumerate(dataset), total=max_samples, desc="Processing TextVQA"):
                if idx >= max_samples:
                    break

                if not isinstance(sample.get("image"), Image.Image):
                    continue

                img = sample["image"].convert("RGB")

                formatted_data.append({
                    "id": sample.get("image_id", str(idx)),
                    "image": img,
                    "question": sample["question"],
                    "answers": sample.get("answers", []),
                })

        # ======== DocVQA (Subset for Testing) ========
        elif benchmark_name == "docvqa":
            # Using the small public DocVQA sample on HF
            # For full replication, you might need a larger/official dataset source
            hf_name = "nielsr/docvqa_1200_examples"

            # Map "validation" -> "test" since this specific subset only has train/test
            actual_split = split
            if split.lower() in ["validation", "val"]:
                actual_split = "test"

            dataset = load_dataset(hf_name, split=actual_split, streaming=True)

            for idx, sample in tqdm(enumerate(dataset), total=max_samples, desc="Processing DocVQA"):
                if idx >= max_samples:
                    break

                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue

                img = img.convert("RGB")

                # In this dataset:
                # - question text is under "query"
                # - answers is typically a list under "answers"
                question = sample.get("query") or sample.get("question") or ""
                # Ensure question is a string (not dict)
                if isinstance(question, dict):
                    question = question.get("text", "") or str(list(question.values())[0]) if question else ""
                question = str(question)
                answers = sample.get("answers", [])

                # Normalize to List[str] - handle various formats
                if answers is None:
                    answers = []
                elif isinstance(answers, str):
                    answers = [answers]
                elif isinstance(answers, dict):
                    # Handle dict format - extract string values
                    answers = [str(v) for v in answers.values() if v]
                elif isinstance(answers, list):
                    # Handle list that may contain dicts or strings
                    normalized = []
                    for a in answers:
                        if isinstance(a, dict):
                            normalized.extend([str(v) for v in a.values() if v])
                        elif a is not None:
                            normalized.append(str(a))
                    answers = normalized

                formatted_data.append({
                    "id": sample.get("id", str(idx)),
                    "image": img,
                    "question": question,
                    "answers": answers,
                })

        # ======== GQA (Corrected) ========
        elif benchmark_name == "gqa":
            # Use vikhyatk/gqa which has images+questions merged
            hf_name = "vikhyatk/gqa"

            # Map splits - available: ['train_balanced', 'val_balanced']
            if split.lower() in ["validation", "val", "test", "testdev"]:
                actual_split = "val_balanced"
            else:
                actual_split = "train_balanced"

            dataset = load_dataset(hf_name, split=actual_split, streaming=True)

            for idx, sample in tqdm(enumerate(dataset), total=max_samples, desc="Processing GQA"):
                if idx >= max_samples:
                    break

                # GQA image is stored as PIL under "image"
                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue

                img = img.convert("RGB")

                # GQA usually provides a single 'answer' string
                answer = sample.get("answer")
                answers_list = [answer] if answer else []

                formatted_data.append({
                    "id": str(sample.get("question_id", str(idx))),
                    "image": img,
                    "question": sample["question"],
                    "answers": answers_list,
                })

        else:
            raise ValueError(f"❌ Unsupported benchmark: {benchmark_name}")

        print(f"✅ Loaded {len(formatted_data)} samples from {benchmark_name}")
        return formatted_data

    except Exception as e:
        print(f"❌ Error while loading dataset: {e}")
        print("⚠️ Check internet connection or dataset configuration.")
        return []


def save_debug_image(image: Image.Image, run_id: str):
    """Helper to save input images for verification."""
    os.makedirs("results/debug", exist_ok=True)
    image.save(f"results/debug/input_{run_id}.png")