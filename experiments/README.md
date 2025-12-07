# FastVLM Replication Experiments

This folder contains scripts to replicate key results from the FastVLM paper (CVPR 2025), including Tables 3, 4, 5 and Figure 5.

## Experiments Overview

| Script           | Paper Reference | Description                                                     | Datasets            |
| ---------------- | --------------- | --------------------------------------------------------------- | ------------------- |
| `exp_table3.py`  | Table 3         | Encoder comparison benchmark (ViT-L/14, ConvNeXt-L, FastViT-HD) | None (latency only) |
| `exp_table4.py`  | Table 4         | Visual token efficiency across encoders and resolutions         | TextVQA, DocVQA     |
| `exp_table5.py`  | Table 5         | FastViT-HD visual token scaling (256-1024px)                    | TextVQA             |
| `exp_table11.py` | Table 11        | FastVLM 0.5B vs 1.5B comparison                                 | TextVQA, DocVQA     |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Encoder Models (for Table 3)

```bash
pip install open_clip_torch
./download_encoder_models.sh
```

### 3. Run All Experiments

```bash
# Full dataset evaluation
./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage3 cuda

# Quick test with 10 samples
./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage3 cuda 10
```

**Arguments:**

```bash
./run_all_experiments.sh [MODEL_PATH] [DEVICE] [NUM_SAMPLES]
```

- `MODEL_PATH`: Path to FastVLM checkpoint (default: `../checkpoints/llava-fastvithd_0.5b_stage3`)
- `DEVICE`: `cuda` or `cpu` (default: `cuda`)
- `NUM_SAMPLES`: Number of samples per benchmark. Omit for full dataset.

## Individual Experiments

### Table 3: Encoder Comparison

Benchmarks vision encoder latency and parameter count. No dataset required.

```bash
python exp_table3.py
```

**Output:**

- `results/table3_encoder_comparison.png`

**Encoders tested:**

- ViT-L/14 (224px)
- ConvNeXt-L (320px)
- FastViT-HD (224px)

---

### Table 4: Visual Token Efficiency

Evaluates accuracy and latency across different encoder/resolution combinations.

```bash
# Full dataset
python exp_table4.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --device cuda

# Quick test
python exp_table4.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --device cuda --num-samples 10
```

**Output:**

- `results/table4_visual_token_efficiency.png`

**Configurations tested:**

| Encoder    | Resolution | Visual Tokens |
| ---------- | ---------- | ------------- |
| FastViT-HD | 256px      | 16            |
| ConvNeXt-L | 320px      | 100           |
| FastViT-HD | 512px      | 64            |
| FastViT-HD | 768px      | 144           |
| ConvNeXt-L | 512px      | 256           |
| FastViT-HD | 1024px     | 256           |

---

### Table 5: FastViT-HD Visual Token Scaling

Measures how accuracy scales with resolution/visual tokens for FastViT-HD.

```bash
# Full dataset
python exp_table5.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --device cuda

# Quick test
python exp_table5.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --device cuda --num-samples 10
```

**Output:**

- `results/table5_fastvithd_efficiency.png`

**Resolutions tested:** 256, 512, 768, 1024px

**Visual token formula:** `(resolution / 64)^2`

---

### Table 11: FastVLM 0.5B vs 1.5B

Compares FastVLM 0.5B and 1.5B models across different resolutions.

```bash
# Full dataset
python exp_table11.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --device cuda

# Quick test
python exp_table11.py --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 --device cuda --num-samples 10
```

**Output:**

- `results/table11_model_comparison.png`

**Models tested:** FastVLM-0.5B, FastVLM-1.5B
**Resolutions tested:** 1024, 2048px

---

## Common Arguments

| Argument        | Description                                   | Default                                      |
| --------------- | --------------------------------------------- | -------------------------------------------- |
| `--model-path`  | Path to FastVLM checkpoint                    | `../checkpoints/llava-fastvithd_0.5b_stage3` |
| `--device`      | Device (`cuda` or `cpu`)                      | `cuda`                                       |
| `--num-samples` | Samples per benchmark (omit for full dataset) | `None`                                       |

## Output Structure

```
results/
    table3_encoder_comparison.png
    table4_visual_token_efficiency.png
    table5_fastvithd_efficiency.png
    table11_model_comparison.png
```

## Hardware Notes

Scripts are optimized for NVIDIA GPUs:

- FP16 precision (`torch.float16`)
- CUDA event timing for accurate latency measurement
- Memory cleanup between runs

See `limitations.md` for detailed notes on replication limitations and hardware differences from the original paper.

## Datasets

| Dataset | Source             | Split      |
| ------- | ------------------ | ---------- |
| TextVQA | `lmms-lab/textvqa` | validation |
| DocVQA  | `lmms-lab/DocVQA`  | validation |

Datasets are streamed from HuggingFace and do not require local download.

## Citation

```bibtex
@inproceedings{fastvlm2025,
  title={FastVLM: Efficient Vision Encoding for Vision Language Models},
  author={Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li, Cem Koc, Nate True, Albert Antony, Gokul Santhanam, James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  booktitle={CVPR},
  year={2025}
}
```
