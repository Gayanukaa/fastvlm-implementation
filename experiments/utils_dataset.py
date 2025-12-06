"""
Utility functions for loading multiple FastVLM-compatible visual QA benchmark datasets.

Supports streaming-based loading for:
- TextVQA (lmms-lab/textvqa)
- DocVQA (lmms-lab/DocVQA)
- SEED-Bench (lmms-lab/SEED-Bench)
- ScienceQA-IMG / SQA (lmms-lab/ScienceQA-IMG)
- VQAv2 (lmms-lab/VQAv2)
- POPE (lmms-lab/POPE)
- GQA (lmms-lab/GQA)  [images loaded normally, instructions streamed]

This module enables fast debugging by allowing:
- Partial dataset loading via max_samples (streams shards instead of full download)
- Uniform return format across all datasets:
    {
        'id': str,
        'image': PIL.Image.Image,
        'question': str,
        'answers': List[str]
    }

Intended for use in FastVLM evaluation scripts, quick benchmarking,
and encoder comparison experiments.
"""


import os
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def get_benchmark_dataset(
    benchmark_name: str = "textvqa",
    split: str = "validation",
    max_samples: Optional[int] = 10
) -> List[Dict[str, Any]]:
    """
    Loads a benchmark dataset and returns a simplified list of samples.

    Args:
        benchmark_name: Name of the dataset (textvqa | docvqa | gqa)
        split: Dataset split to load
        max_samples: Number of samples to load (for quick testing).
                     Set to None to load the full dataset.

    Returns:
        List of dicts formatted as:
        [{'image': PIL.Image, 'question': str, 'id': str, 'answers': List[str]}]
    """

    benchmark_name = benchmark_name.lower()

    # Determine if we should stream or load full dataset
    use_streaming = max_samples is not None
    sample_limit = max_samples if max_samples is not None else float('inf')

    if max_samples is None:
        print(f"📚 Loading {benchmark_name} ({split}) - FULL dataset (no streaming)...")
    else:
        print(f"📚 Loading {benchmark_name} ({split}) - Max samples: {max_samples} (streaming)...")

    formatted_data = []

    try:
        # ======== TextVQA (lmms-lab) ========
        if benchmark_name == "textvqa":
            dataset = load_dataset("lmms-lab/textvqa", split=split, streaming=use_streaming)

            iterator = enumerate(dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing TextVQA"):
                if idx >= sample_limit:
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

        # ======== DocVQA (lmms-lab) ========
        elif benchmark_name == "docvqa":
            # lmms-lab/DocVQA requires config name: 'DocVQA' or 'InfographicVQA'
            dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split, streaming=use_streaming)

            iterator = enumerate(dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing DocVQA"):
                if idx >= sample_limit:
                    break

                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue

                img = img.convert("RGB")

                # lmms-lab/DocVQA format:
                # - question: str
                # - answers: list of strings
                question = sample.get("question", "")
                answers = sample.get("answers", [])

                # Normalize answers to List[str]
                if answers is None:
                    answers = []
                elif isinstance(answers, str):
                    answers = [answers]
                elif isinstance(answers, list):
                    answers = [str(a) for a in answers if a is not None]

                formatted_data.append({
                    "id": sample.get("questionId", str(idx)),
                    "image": img,
                    "question": str(question),
                    "answers": answers,
                })


        # ======== SEED-Bench (lmms-lab) ========
        elif benchmark_name in ["seedbench", "seed-bench", "seed"]:
            # lmms-lab/SEED-Bench has only a 'test' split
            actual_split = "test"

            dataset = load_dataset(
                "lmms-lab/SEED-Bench",
                split=actual_split,
                streaming=use_streaming,
            )

            iterator = enumerate(dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing SEED-Bench"):
                if idx >= sample_limit:
                    break

                imgs = sample.get("image")
                img = None
                # image column is a list of images; take the first if present
                if isinstance(imgs, list) and len(imgs) > 0 and isinstance(imgs[0], Image.Image):
                    img = imgs[0]
                elif isinstance(imgs, Image.Image):
                    img = imgs

                if img is None:
                    continue

                img = img.convert("RGB")

                question = sample.get("question", "")
                answer_letter = str(sample.get("answer", "")).strip().upper()

                # Choices: A/B/C/D
                choice_map = {
                    "A": sample.get("choice_a"),
                    "B": sample.get("choice_b"),
                    "C": sample.get("choice_c"),
                    "D": sample.get("choice_d"),
                }

                answers_list: List[str] = []
                if answer_letter in choice_map and choice_map[answer_letter] is not None:
                    answers_list = [str(choice_map[answer_letter])]

                formatted_data.append({
                    "id": str(sample.get("question_id", sample.get("data_id", idx))),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                })


        # ======== ScienceQA-IMG / SQA (lmms-lab) ========
        elif benchmark_name == "sqa" or benchmark_name == "scienceqa" or benchmark_name == "scienceqa-img":
        # elif benchmark_name in ["sqa", "scienceqa", "scienceqa-img"]:
            # lmms-lab/ScienceQA-IMG has train/validation/test
            if split.lower() in ["val", "validation"]:
                actual_split = "validation"
            elif split.lower() in ["test", "testing"]:
                actual_split = "test"
            elif split.lower() in ["train", "training"]:
                actual_split = "train"
            else:
                actual_split = split  # fall back to what user provided

            dataset = load_dataset(
                "lmms-lab/ScienceQA-IMG",
                split=actual_split,
                streaming=use_streaming,
            )

            iterator = enumerate(dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing ScienceQA-IMG"):
                if idx >= sample_limit:
                    break

                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")

                question = sample.get("question", "")
                choices = sample.get("choices", [])
                answer_idx = sample.get("answer", None)

                answers_list: List[str] = []
                if isinstance(choices, list) and isinstance(answer_idx, int):
                    if 0 <= answer_idx < len(choices):
                        # Store the *text* of the correct answer
                        answers_list = [str(choices[answer_idx])]

                formatted_data.append({
                    "id": str(sample.get("data_id", sample.get("question", idx))),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                })

                # ======== VQAv2 (lmms-lab) ========
        elif benchmark_name in ["vqav2", "vqa2", "vqa-v2"]:
            # Map user-friendly split names to lmms-lab/VQAv2 splits
            split_lower = split.lower()
            if split_lower in ["val", "validation"]:
                actual_split = "validation"
            elif split_lower in ["testdev", "test-dev"]:
                actual_split = "testdev"
            elif split_lower in ["test", "testing"]:
                actual_split = "test"
            else:
                actual_split = "validation"  # safe default

            dataset = load_dataset(
                "lmms-lab/VQAv2",
                split=actual_split,
                streaming=use_streaming,
            )

            iterator = enumerate(dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing VQAv2"):
                if idx >= sample_limit:
                    break

                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")

                question = sample.get("question", "")
                raw_answers = sample.get("answers", [])

                answers_list: List[str] = []
                # HF schema: answers is a list of dicts with key "answer"
                if isinstance(raw_answers, list):
                    for a in raw_answers:
                        if isinstance(a, dict) and "answer" in a:
                            ans_str = a["answer"]
                        else:
                            ans_str = a
                        if ans_str is not None:
                            answers_list.append(str(ans_str))

                formatted_data.append({
                    "id": str(sample.get("question_id", idx)),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                })


        # ======== POPE (lmms-lab) ========
        elif benchmark_name == "pope":
            # Only "test" split exists for lmms-lab/POPE
            actual_split = "test"

            dataset = load_dataset(
                "lmms-lab/POPE",
                split=actual_split,
                streaming=use_streaming,
            )

            iterator = enumerate(dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing POPE"):
                if idx >= sample_limit:
                    break

                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue
                img = img.convert("RGB")

                question = sample.get("question", "")
                answer = sample.get("answer", None)

                answers_list: List[str] = []
                if isinstance(answer, str):
                    answers_list = [answer]

                formatted_data.append({
                    # prefer question_id, fall back to id / idx
                    "id": str(sample.get("question_id", sample.get("id", idx))),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                })



        # ======== GQA (lmms-lab) ========
        elif benchmark_name == "gqa":
            # lmms-lab/GQA has separate configs for images and instructions
            # We need to load both and join them by imageId

            if split.lower() in ["validation", "val", "test", "testdev"]:
                actual_split = "testdev"
                images_config = "testdev_balanced_images"
                instructions_config = "testdev_balanced_instructions"
            elif split.lower() == "train":
                actual_split = "train"
                images_config = "train_balanced_images"
                instructions_config = "train_balanced_instructions"
            else:
                actual_split = "testdev"
                images_config = "testdev_balanced_images"
                instructions_config = "testdev_balanced_instructions"

            # Load images first (not streaming, we need to index by id)
            print(f"   Loading GQA images ({images_config})...")
            images_dataset = load_dataset("lmms-lab/GQA", images_config, split=actual_split)

            # Build image lookup dict
            image_lookup = {}
            for img_sample in tqdm(images_dataset, desc="Building image index"):
                img_id = img_sample.get("id")
                img = img_sample.get("image")
                if img_id and isinstance(img, Image.Image):
                    image_lookup[img_id] = img

            print(f"   Loaded {len(image_lookup)} images")

            # Load instructions with streaming
            print(f"   Loading GQA instructions ({instructions_config})...")
            instructions_dataset = load_dataset("lmms-lab/GQA", instructions_config, split=actual_split, streaming=use_streaming)

            iterator = enumerate(instructions_dataset)
            pbar_total = max_samples if max_samples else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing GQA"):
                if idx >= sample_limit:
                    break

                # Get image from lookup
                image_id = sample.get("imageId")
                img = image_lookup.get(image_id)

                if img is None:
                    continue

                img = img.convert("RGB")

                # lmms-lab/GQA format:
                # - question: str
                # - answer: str (single answer)
                answer = sample.get("answer", "")
                answers_list = [answer] if answer else []

                formatted_data.append({
                    "id": str(sample.get("id", str(idx))),
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