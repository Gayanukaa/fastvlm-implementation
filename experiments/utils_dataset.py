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
import json
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


# Root cache directory for benchmark datasets
# You can also override this via an env var if you want:
#   FASTVLM_BENCHMARK_CACHE_DIR


# Base directory for caching datasets (relative to this script)
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_ROOT = os.path.join(THIS_DIR, "Benchmark_datasets")
os.makedirs(CACHE_ROOT, exist_ok=True)




def _get_cache_dir(benchmark_name: str) -> str:
    """Returns a cache directory path for a given benchmark."""
    # Make benchmark_name path-safe
    safe_name = benchmark_name.lower().replace("/", "_").replace("\\", "_")
    return os.path.join(CACHE_ROOT, safe_name)


def _load_from_cache(
    benchmark_name: str,
    max_samples: Optional[int],
) -> List[Dict[str, Any]]:
    """
    Load up to max_samples samples from disk cache for this benchmark.
    Returns a list of dicts in the same format as get_benchmark_dataset.
    """
    cache_dir = _get_cache_dir(benchmark_name)
    meta_path = os.path.join(cache_dir, "metadata.jsonl")

    if not os.path.exists(meta_path):
        return []

    formatted_data: List[Dict[str, Any]] = []
    sample_limit = max_samples if max_samples is not None else float("inf")

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                if line_idx >= sample_limit:
                    break
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)

                img_path = os.path.join(cache_dir, rec["image_path"])
                if not os.path.exists(img_path):
                    continue

                img = Image.open(img_path).convert("RGB")

                formatted_data.append({
                    "id": rec["id"],
                    "image": img,
                    "question": rec["question"],
                    "answers": rec["answers"],
                })
    except Exception as e:
        print(f"⚠️ Error reading cache for {benchmark_name}: {e}")
        return []

    if formatted_data:
        print(f"✅ Loaded {len(formatted_data)} cached samples for {benchmark_name}")

    return formatted_data


def _append_to_cache(benchmark_name: str, sample: Dict[str, Any]) -> None:
    """
    Append a single sample to the on-disk cache for this benchmark.

    sample format:
        {
            "id": str,
            "image": PIL.Image.Image,
            "question": str,
            "answers": List[str],
        }
    """
    cache_dir = _get_cache_dir(benchmark_name)
    images_dir = os.path.join(cache_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    meta_path = os.path.join(cache_dir, "metadata.jsonl")

    # Determine next local index by counting existing lines
    local_idx = 0
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            for _ in f:
                local_idx += 1

    # Save image
    image_filename = f"{local_idx:06d}.png"
    image_path_rel = os.path.join("images", image_filename)
    image_path_abs = os.path.join(cache_dir, image_path_rel)

    img: Image.Image = sample["image"]
    img.save(image_path_abs)

    # Save metadata (with relative image path)
    record = {
        "local_idx": local_idx,
        "id": sample["id"],
        "image_path": image_path_rel,
        "question": sample["question"],
        "answers": sample["answers"],
    }

    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_benchmark_dataset(
    benchmark_name: str = "textvqa",
    split: str = "validation",
    max_samples: Optional[int] = 10,
) -> List[Dict[str, Any]]:
    """
    Loads a benchmark dataset and returns a simplified list of samples.

    Args:
        benchmark_name: Name of the dataset (textvqa | docvqa | gqa | ...)
        split: Dataset split to load
        max_samples: Number of samples to load (for quick testing).
                     Set to None to load the full dataset (no streaming, no cache).

    Returns:
        List of dicts formatted as:
        [{'image': PIL.Image, 'question': str, 'id': str, 'answers': List[str]}]
    """

    benchmark_name = benchmark_name.lower()

    # Determine if we should stream or load full dataset
    use_streaming = max_samples is not None
    sample_limit = max_samples if max_samples is not None else float("inf")

    # For full dataset mode (max_samples is None), keep old behavior:
    if max_samples is None:
        print(f"📚 Loading {benchmark_name} ({split}) - FULL dataset (no streaming, no cache)...")
        formatted_data: List[Dict[str, Any]] = []
    else:
        print(f"📚 Loading {benchmark_name} ({split}) - Max samples: {max_samples} (streaming + cache)...")
        # Try loading from cache first
        formatted_data = _load_from_cache(benchmark_name, max_samples)
        if len(formatted_data) >= sample_limit:
            # Already have enough cached samples
            print(f"✅ Returning {len(formatted_data)} samples from cache for {benchmark_name}")
            return formatted_data

    try:
        # Helper to check stopping condition based on how many samples we have collected
        def has_enough_samples() -> bool:
            return len(formatted_data) >= sample_limit

        # Helper to add a sample (append + write to cache if streaming mode)
        def add_sample(sample_dict: Dict[str, Any]):
            if has_enough_samples():
                return
            formatted_data.append(sample_dict)
            # Only cache when using streaming (i.e., partial, network-based loading)
            if use_streaming:
                _append_to_cache(benchmark_name, sample_dict)

        # ======== TextVQA (lmms-lab) ========
        if benchmark_name == "textvqa":
            dataset = load_dataset("lmms-lab/textvqa", split=split, streaming=use_streaming)

            iterator = enumerate(dataset)
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing TextVQA"):
                if has_enough_samples():
                    break

                if not isinstance(sample.get("image"), Image.Image):
                    continue

                img = sample["image"].convert("RGB")

                sample_dict = {
                    "id": sample.get("image_id", str(idx)),
                    "image": img,
                    "question": sample["question"],
                    "answers": sample.get("answers", []),
                }
                add_sample(sample_dict)

        # ======== DocVQA (lmms-lab) ========
        elif benchmark_name == "docvqa":
            # lmms-lab/DocVQA requires config name: 'DocVQA' or 'InfographicVQA'
            dataset = load_dataset("lmms-lab/DocVQA", "DocVQA", split=split, streaming=use_streaming)

            iterator = enumerate(dataset)
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing DocVQA"):
                if has_enough_samples():
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

                sample_dict = {
                    "id": sample.get("questionId", str(idx)),
                    "image": img,
                    "question": str(question),
                    "answers": answers,
                }
                add_sample(sample_dict)

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
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing SEED-Bench"):
                if has_enough_samples():
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

                sample_dict = {
                    "id": str(sample.get("question_id", sample.get("data_id", idx))),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                }
                add_sample(sample_dict)

        # ======== ScienceQA-IMG / SQA (lmms-lab) ========
        elif benchmark_name in ["sqa", "scienceqa", "scienceqa-img"]:
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
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing ScienceQA-IMG"):
                if has_enough_samples():
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

                sample_dict = {
                    "id": str(sample.get("data_id", sample.get("question", idx))),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                }
                add_sample(sample_dict)

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
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing VQAv2"):
                if has_enough_samples():
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

                sample_dict = {
                    "id": str(sample.get("question_id", idx)),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                }
                add_sample(sample_dict)

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
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing POPE"):
                if has_enough_samples():
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

                sample_dict = {
                    # prefer question_id, fall back to id / idx
                    "id": str(sample.get("question_id", sample.get("id", idx))),
                    "image": img,
                    "question": str(question),
                    "answers": answers_list,
                }
                add_sample(sample_dict)

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
            instructions_dataset = load_dataset(
                "lmms-lab/GQA",
                instructions_config,
                split=actual_split,
                streaming=use_streaming,
            )

            iterator = enumerate(instructions_dataset)
            pbar_total = max_samples if (use_streaming and max_samples is not None) else None

            for idx, sample in tqdm(iterator, total=pbar_total, desc="Processing GQA"):
                if has_enough_samples():
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

                sample_dict = {
                    "id": str(sample.get("id", str(idx))),
                    "image": img,
                    "question": sample["question"],
                    "answers": answers_list,
                }
                add_sample(sample_dict)

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
