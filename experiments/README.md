# FastVLM Ablation Experiments

This folder contains standalone scripts to reproduce and visualize FastVLM ablation experiments using the **TextVQA benchmark**, optimized for RTX 4070 8GB GPU.

## Experiments Overview

| Experiment                | Script                         | Purpose                                    | Key Metrics                  |
| ------------------------- | ------------------------------ | ------------------------------------------ | ---------------------------- |
| **Resolution Scaling**    | `exp_resolution_scaling.py`    | Test resolutions [224, 336, ..., 1024]     | Latency, VRAM                |
| **Token Budget Ablation** | `exp_token_budget_ablation.py` | Test token budgets [16, 64, 144, 256, 576] | Accuracy vs Latency          |
| **Stage Comparison**      | `exp_stage_comparison.py`      | Compare Stage-2 vs Stage-3 models          | Accuracy (Exact Match), TTFT |
| **Prompt Length Effect**  | `exp_prompt_length_effect.py`  | Test prompts [Short, Medium, Long]         | Latency, TTFT                |

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run All Experiments

**Linux/Mac:**

```bash
# Usage: ./run_all_experiments.sh [stage2_path] [stage3_path] [device] [num_samples]
./run_all_experiments.sh ../checkpoints/llava-fastvithd_0.5b_stage2 ../checkpoints/llava-fastvithd_0.5b_stage3 cuda 20
```

## Individual Experiments

### Resolution Scaling

Measures inference latency and VRAM usage across different input resolutions using TextVQA samples.

```bash
python exp_resolution_scaling.py \
    --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --device cuda \
    --num-samples 20
```

### Token Budget Ablation

Evaluates the trade-off between accuracy (Exact Match) and latency by varying the visual token budget (via resolution).

```bash
python exp_token_budget_ablation.py \
    --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --device cuda \
    --num-samples 20
```

### Stage Comparison

Compares the performance of Stage 2 (Pre-training) vs Stage 3 (Fine-tuning) models on the TextVQA benchmark.

```bash
python exp_stage_comparison.py \
    --stage2-path ../checkpoints/llava-fastvithd_0.5b_stage2 \
    --stage3-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --device cuda \
    --num-samples 20
```

### Prompt Length Effect

Analyzes how different prompt lengths (Short, Medium, Long) affect Time-to-First-Token (TTFT) and Total Latency.

```bash
python exp_prompt_length_effect.py \
    --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --device cuda \
    --num-samples 20
```

## Configuration Options

All scripts support these common arguments:

| Argument        | Description               | Default  |
| --------------- | ------------------------- | -------- |
| `--model-path`  | Path to model checkpoint  | Required |
| `--device`      | Device (cuda/cpu)         | `cuda`   |
| `--num-samples` | Number of TextVQA samples | `20`     |

### Stage Comparison Specific:

- `--stage2-path`: Path to Stage 2 model
- `--stage3-path`: Path to Stage 3 model

The scripts are optimized for RTX 4070 8GB:

- **FP16 precision**: `torch_dtype=torch.float16`
- **CUDA events**: Precise GPU timing
- **Memory cleanup**: `torch.cuda.empty_cache()` between runs
- **Smart loading**: `low_cpu_mem_usage=True`

- Time-to-First-Token (TTFT)
- Total Latency
- Prompt Tokens count

## Reproducing Paper Results

The experiments are designed to reproduce key findings from FastVLM CVPR 2025 using the **TextVQA** benchmark:

| Experiment Script              | Paper Figure(s) / Table(s) | Description / Purpose                                                                | Models Used                                                  | Comparison Models  | Benchmarks    |
| :----------------------------- | :------------------------- | :----------------------------------------------------------------------------------- | :----------------------------------------------------------- | :----------------- | :------------ |
| `exp_resolution_scaling.py`    | Fig. 3, Fig. 4, Table 4    | Tests Latency & VRAM across resolutions (224-1024px).                                | `llava-fastvithd_0.5b_stage3`                                | N/A                | TextVQA (Val) |
| `exp_token_budget_ablation.py` | Table 5                    | Evaluates Accuracy vs Latency trade-off for different token budgets (16-576 tokens). | `llava-fastvithd_0.5b_stage3`                                | N/A                | TextVQA (Val) |
| `exp_stage_comparison.py`      | Table 6                    | Compares Stage 2 vs Stage 3 performance (Accuracy & Latency).                        | `llava-fastvithd_0.5b_stage2`, `llava-fastvithd_0.5b_stage3` | Stage 2 vs Stage 3 | TextVQA (Val) |
| `exp_prompt_length_effect.py`  | Fig. 5                     | Measures impact of prompt length (Short/Medium/Long) on Latency & TTFT.              | `llava-fastvithd_0.5b_stage3`                                | N/A                | TextVQA (Val) |

## Citation

```bibtex
@inproceedings{fastvlm2025,
  title={FastVLM: Efficient Vision-Language Model Inference via Token Reduction},
  author={Authors et al.},
  booktitle={CVPR},
  year={2025}
}
```
