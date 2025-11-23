"""
Utility functions for loading FastVLM benchmark datasets.
Focuses on TextVQA as the primary benchmark for resolution/OCR testing.
"""

import os
from typing import Any, Dict, List

from datasets import load_dataset
from PIL import Image
from tqdm import tqdm


def get_benchmark_dataset(
    benchmark_name: str = "textvqa", split: str = "validation", max_samples: int = 10
) -> List[Dict[str, Any]]:
    """
    Loads a benchmark dataset and returns a simplified list of samples.

    Args:
        benchmark_name: Name of the dataset (default: textvqa)
        split: Dataset split to load (default: validation)
        max_samples: Number of samples to load (for quick testing)

    Returns:
        List of dicts: [{'image': PIL.Image, 'question': str, 'id': str, 'answers': List[str]}]
    """
    print(f"📚 Loading {benchmark_name} ({split}) - Max samples: {max_samples}...")

    formatted_data = []

    try:
        if benchmark_name.lower() == "textvqa":
            # Load TextVQA from Hugging Face
            dataset = load_dataset("textvqa", split=split, streaming=True)

            counter = 0
            for sample in tqdm(dataset, total=max_samples, desc="Processing samples"):
                if counter >= max_samples:
                    break

                # Extract relevant fields
                try:
                    # TextVQA structure: 'image', 'question', 'answers'
                    img = sample["image"]
                    if not isinstance(img, Image.Image):
                        continue

                    # Convert to RGB to ensure consistency
                    img = img.convert("RGB")

                    formatted_data.append(
                        {
                            "id": sample.get("image_id", str(counter)),
                            "image": img,
                            "question": sample["question"],
                            "answers": sample.get("answers", []),
                        }
                    )
                    counter += 1
                except Exception as e:
                    print(f"⚠️ Skipping sample due to error: {e}")
                    continue

        else:
            raise ValueError(f"Benchmark {benchmark_name} not implemented yet.")

        print(
            f"✅ Successfully loaded {len(formatted_data)} samples from {benchmark_name}"
        )
        return formatted_data

    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        print(
            "Falling back to local dummy mode if needed, or check internet connection."
        )
        return []


def save_debug_image(image: Image.Image, run_id: str):
    """Optional helper to save input images for verification."""
    os.makedirs("results/debug", exist_ok=True)
    image.save(f"results/debug/input_{run_id}.png")
