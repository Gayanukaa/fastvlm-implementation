# """
# Utility functions for loading FastVLM benchmark datasets.
# Focuses on TextVQA as the primary benchmark for resolution/OCR testing.
# """

# import os
# from typing import Any, Dict, List

# from datasets import load_dataset
# from PIL import Image
# from tqdm import tqdm


# def get_benchmark_dataset(
#     benchmark_name: str = "textvqa", split: str = "validation", max_samples: int = 10
# ) -> List[Dict[str, Any]]:
#     """
#     Loads a benchmark dataset and returns a simplified list of samples.

#     Args:
#         benchmark_name: Name of the dataset (default: textvqa)
#         split: Dataset split to load (default: validation)
#         max_samples: Number of samples to load (for quick testing)

#     Returns:
#         List of dicts: [{'image': PIL.Image, 'question': str, 'id': str, 'answers': List[str]}]
#     """
#     print(f"📚 Loading {benchmark_name} ({split}) - Max samples: {max_samples}...")

#     formatted_data = []

#     try:
#         if benchmark_name.lower() == "textvqa":
#             # Load TextVQA from Hugging Face
#             dataset = load_dataset("textvqa", split=split, streaming=True)

#             counter = 0
#             for sample in tqdm(dataset, total=max_samples, desc="Processing samples"):
#                 if counter >= max_samples:
#                     break

#                 # Extract relevant fields
#                 try:
#                     # TextVQA structure: 'image', 'question', 'answers'
#                     img = sample["image"]
#                     if not isinstance(img, Image.Image):
#                         continue

#                     # Convert to RGB to ensure consistency
#                     img = img.convert("RGB")

#                     formatted_data.append(
#                         {
#                             "id": sample.get("image_id", str(counter)),
#                             "image": img,
#                             "question": sample["question"],
#                             "answers": sample.get("answers", []),
#                         }
#                     )
#                     counter += 1
#                 except Exception as e:
#                     print(f"⚠️ Skipping sample due to error: {e}")
#                     continue

#         else:
#             raise ValueError(f"Benchmark {benchmark_name} not implemented yet.")

#         print(
#             f"✅ Successfully loaded {len(formatted_data)} samples from {benchmark_name}"
#         )
#         return formatted_data

#     except Exception as e:
#         print(f"❌ Error loading dataset: {e}")
#         print(
#             "Falling back to local dummy mode if needed, or check internet connection."
#         )
#         return []


# def save_debug_image(image: Image.Image, run_id: str):
#     """Optional helper to save input images for verification."""
#     os.makedirs("results/debug", exist_ok=True)
#     image.save(f"results/debug/input_{run_id}.png")






#-------------------------------------------------------------------------------------------------------------------------
# Now it can work with the TextVQA and  DocVQA benchmark datasets.not for GQA , POPE, PopQA



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
        # ======== TextVQA (unchanged) ========
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

        # ======== DocVQA (NEW) ========
        elif benchmark_name == "docvqa":
            # Using the small public DocVQA sample on HF
            # Docs: load_dataset("nielsr/docvqa_1200_examples")
            hf_name = "nielsr/docvqa_1200_examples"

            # Map "validation" -> "test" since this dataset has train/test only
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
                answers = sample.get("answers", [])

                # Normalize to List[str]
                if isinstance(answers, str):
                    answers = [answers]
                elif answers is None:
                    answers = []

                formatted_data.append({
                    "id": sample.get("id", str(idx)),
                    "image": img,
                    "question": question,
                    "answers": answers,
                })

        # ======== GQA (unchanged) ========
        elif benchmark_name == "gqa":
            dataset = load_dataset("gqa", "balanced", split=split, streaming=True)

            for idx, sample in tqdm(enumerate(dataset), total=max_samples, desc="Processing GQA"):
                if idx >= max_samples:
                    break

                # GQA image is stored as PIL under "image"
                img = sample.get("image")
                if not isinstance(img, Image.Image):
                    continue
                    
                img = img.convert("RGB")

                formatted_data.append({
                    "id": sample.get("question_id", str(idx)),
                    "image": img,
                    "question": sample["question"],
                    "answers": [sample.get("answer")] if sample.get("answer") else [],
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
