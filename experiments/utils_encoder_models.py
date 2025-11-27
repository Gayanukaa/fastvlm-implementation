# #!/usr/bin/env python3
# """
# Utility helpers for model comparison experiments.

# Provides model loading and evaluation helpers used to compare two models
# (e.g., FastViT-HD vs ConvNeXt-L) on benchmarks such as TextVQA.

# Contents:
# - load_model: load a pretrained model via the project's builder
# - measure_inference: run one sample through the model and measure latency
# - evaluate_model: run evaluation over a dataset (latency + exact-match accuracy)
# - save_results_csv: persist per-sample results to CSV

# This module intentionally keeps functions small and dependency-minimal so the
# experiment scripts can import only what they need.
# """

# from typing import Any, Dict, Iterable, List, Optional, Tuple
# import csv
# import gc
# import time

# import numpy as np
# import torch
# from PIL import Image
# from tqdm import tqdm

# from llava.constants import (
#     DEFAULT_IM_END_TOKEN,
#     DEFAULT_IM_START_TOKEN,
#     DEFAULT_IMAGE_TOKEN,
#     IMAGE_TOKEN_INDEX,
# )
# from llava.conversation import conv_templates
# from llava.mm_utils import (
#     get_model_name_from_path,
#     process_images,
#     tokenizer_image_token,
# )
# from llava.model.builder import load_pretrained_model
# from llava.utils import disable_torch_init


# def load_model(model_path: str, device: str = "cuda"):
#     """Load a pretrained model bundle (tokenizer, model, image_processor).

#     Returns (model, tokenizer, image_processor).
#     """
#     disable_torch_init()
#     model_name = get_model_name_from_path(model_path)
#     tokenizer, model, image_processor, _ = load_pretrained_model(
#         model_path,
#         None,
#         model_name,
#         device=device,
#         device_map="auto",
#         torch_dtype=torch.float16,
#     )
#     model.eval()
#     return model, tokenizer, image_processor


# def calculate_exact_match(prediction: str, ground_truths: List[str]) -> float:
#     """Return 1.0 if prediction exactly matches any ground truth (case-insensitive).

#     Very small/fast normalization is applied (lowercase + strip + remove '.')
#     to match simple exact-match behavior used in many benchmarks.
#     """
#     if not ground_truths:
#         return 0.0
#     pred = (prediction or "").lower().strip().replace(".", "")
#     for gt in ground_truths:
#         if pred == (gt or "").lower().strip():
#             return 1.0
#     return 0.0


# def measure_inference(
#     model, tokenizer, image_processor, image: Image.Image, prompt: str, device: str = "cuda"
# ) -> Tuple[float, str]:
#     """Run the model on a single image+prompt and return (latency_ms, generated_text).

#     This function follows the project's tokenization/format conventions used by
#     the experiments (image-token, conversation templates, etc.). It uses CUDA
#     timing events if CUDA is available and falls back to wall-clock timing.
#     """
#     qs = prompt
#     if getattr(model.config, "mm_use_im_start_end", False):
#         qs = (
#             DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
#         )
#     else:
#         qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

#     conv = conv_templates.get("qwen_2")
#     if conv is None:
#         # fallback to first template if name changed
#         conv = list(conv_templates.values())[0].copy()
#     else:
#         conv = conv.copy()

#     conv.append_message(conv.roles[0], qs)
#     conv.append_message(conv.roles[1], None)
#     prompt_formatted = conv.get_prompt()

#     input_ids = (
#         tokenizer_image_token(prompt_formatted, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")
#         .unsqueeze(0)
#         .to(device)
#     )

#     image_tensor = process_images([image], image_processor, model.config)[0]

#     # GPU timing
#     if torch.cuda.is_available() and device.startswith("cuda"):
#         torch.cuda.empty_cache()
#         start_event = torch.cuda.Event(enable_timing=True)
#         end_event = torch.cuda.Event(enable_timing=True)
#         start_event.record()

#     start_time = time.perf_counter()

#     with torch.no_grad():
#         outputs = model.generate(
#             input_ids,
#             images=image_tensor.unsqueeze(0).to(device, dtype=torch.float16),
#             max_new_tokens=20,
#             do_sample=False,
#             use_cache=True,
#         )

#     if torch.cuda.is_available() and device.startswith("cuda"):
#         end_event.record()
#         torch.cuda.synchronize()
#         latency = start_event.elapsed_time(end_event)
#     else:
#         latency = (time.perf_counter() - start_time) * 1000.0

#     generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
#     return float(latency), generated_text


# def evaluate_model(
#     model,
#     tokenizer,
#     image_processor,
#     dataset: Iterable[Dict[str, Any]],
#     device: str = "cuda",
#     max_samples: Optional[int] = None,
#     show_progress: bool = True,
# ) -> Dict[str, Any]:
#     """Evaluate model on `dataset`.

#     dataset may be a list or an iterator (streaming). The function will iterate
#     up to `max_samples` if provided. Returns a dictionary with summary metrics and
#     the raw per-sample results under the `results` key.
#     """
#     results: List[Dict[str, Any]] = []

#     iterator = dataset
#     if show_progress:
#         iterator = tqdm(dataset)

#     for i, sample in enumerate(iterator):
#         if max_samples and i >= max_samples:
#             break

#         image = sample.get("image")
#         question = sample.get("question") or sample.get("question_text") or "What is in this image?"
#         answers = sample.get("answers") or sample.get("answer", [])

#         # Expectation: caller provides dataset items with PIL Image objects.
#         if not isinstance(image, Image.Image):
#             # skip if image not ready; experiments should prepare images via dataset_loader
#             continue

#         try:
#             latency, pred = measure_inference(model, tokenizer, image_processor, image, question, device)
#             acc = calculate_exact_match(pred, answers)

#             results.append(
#                 {
#                     "id": sample.get("image_id", i),
#                     "latency": latency,
#                     "accuracy": acc,
#                     "prediction": pred,
#                     "ground_truths": answers,
#                 }
#             )
#         except Exception as e:
#             # Keep going on individual sample errors
#             results.append({"id": sample.get("image_id", i), "error": str(e)})
#             continue

#     # Summary metrics
#     latencies = [r["latency"] for r in results if "latency" in r]
#     accs = [r["accuracy"] for r in results if "accuracy" in r]

#     summary = {
#         "num_samples": len(results),
#         "avg_latency_ms": float(np.mean(latencies)) if latencies else 0.0,
#         "avg_accuracy": float(np.mean(accs)) if accs else 0.0,
#         "results": results,
#     }

#     return summary


# def save_results_csv(results: List[Dict[str, Any]], csv_path: str):
#     """Save per-sample results to CSV. Results is a list of dicts."""
#     if not results:
#         return
#     fieldnames = sorted({k for r in results for k in r.keys()})
#     with open(csv_path, "w", newline="") as f:
#         writer = csv.DictWriter(f, fieldnames=fieldnames)
#         writer.writeheader()
#         writer.writerows(results)


# if __name__ == "__main__":
#     # Minimal demonstration if invoked directly
#     print("This module provides helpers for model comparison experiments.")




import torch
import open_clip
from torchvision import transforms
from torchvision.models import convnext_large
from transformers import AutoModel

MODEL_PATH = "encoder_models/"   # folder where .pt files are stored

def load_encoder(name):
    name = name.lower()

    if name == "convnext":
        print("🔹 Loading ConvNeXt-L encoder...")
        model = convnext_large(weights=None)
        weights = torch.load(MODEL_PATH + "convnext_large.pt", map_location="cpu")
        model.load_state_dict(weights)
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return model.eval(), preprocess

    elif name == "vit":
        print("🔹 Loading ViT-L/14 (OpenCLIP) encoder...")
        model, preprocess, _ = open_clip.create_model_and_transforms(
            "ViT-L-14", pretrained=None
        )
        weights = torch.load(MODEL_PATH + "vit_l_14_openclip.pt", map_location="cpu")
        model.load_state_dict(weights)
        return model.eval(), preprocess

    elif name == "fastvit":
        print("🔹 Loading FastViT-HD encoder...")
        model = AutoModel.from_pretrained(
            "kevin510/fast-vit-hd",
            trust_remote_code=True
        )
        weights = torch.load(MODEL_PATH + "fastvithd.pt", map_location="cpu")
        model.load_state_dict(weights)

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        return model.eval(), preprocess

    else:
        raise ValueError(f"Unknown encoder name '{name}'. Use: convnext | vit | fastvit")
