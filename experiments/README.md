# FastVLM Ablation Experiments

This folder contains standalone scripts to reproduce and visualize FastVLM ablation experiments from the CVPR 2025 paper, optimized for RTX 4070 8GB GPU.

## 🎯 Experiments Overview

| Experiment                | Script                         | Purpose                                | Outputs                     |
| ------------------------- | ------------------------------ | -------------------------------------- | --------------------------- |
| **Resolution Scaling**    | `exp_resolution_scaling.py`    | Test resolutions [256, 512, 768, 1024] | Latency vs Resolution plots |
| **Token Budget Ablation** | `exp_token_budget_ablation.py` | Test token budgets [16, 64, 144, 256]  | TTFT vs Token Budget        |
| **Stage Comparison**      | `exp_stage_comparison.py`      | Compare Stage-2 vs Stage-3 models      | BLEU similarity analysis    |
| **Prompt Length Effect**  | `exp_prompt_length_effect.py`  | Test prompts [5, 20, 60] words         | Latency vs Prompt Length    |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run All Experiments

**Linux/Mac:**

```bash
./run_all_experiments.sh
```

### 3. Custom Paths

```bash
# Linux/Mac
./run_all_experiments.sh ../checkpoints/stage2 ../checkpoints/stage3 ../images cuda

# Windows
run_all_experiments.bat ..\checkpoints\stage2 ..\checkpoints\stage3 ..\images cuda
```

## 📊 Individual Experiments

### Resolution Scaling

```bash
python exp_resolution_scaling.py \
    --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --image-folder ../images \
    --device cuda
```

**Outputs:**

- `results/resolution_scaling.csv`
- `results/plots/resolution_vs_latency.png`
- `results/plots/resolution_vs_vram.png`

### Token Budget Ablation

```bash
python exp_token_budget_ablation.py \
    --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --image-folder ../images \
    --device cuda
```

**Outputs:**

- `results/token_budget.csv`
- `results/plots/token_budget_vs_ttft_resize.png`
- `results/plots/token_budget_methods_comparison.png`

### Stage Comparison

```bash
python exp_stage_comparison.py \
    --stage2-path ../checkpoints/llava-fastvithd_0.5b_stage2 \
    --stage3-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --image-folder ../images \
    --device cuda
```

**Outputs:**

- `results/stage_compare.csv`
- `results/plots/stage_comparison_ttft.png`
- `results/plots/stage_comparison_bleu.png`

### Prompt Length Effect

```bash
python exp_prompt_length_effect.py \
    --model-path ../checkpoints/llava-fastvithd_0.5b_stage3 \
    --image-folder ../images \
    --device cuda
```

**Outputs:**

- `results/prompt_length.csv`
- `results/plots/prompt_length_vs_latency.png`
- `results/plots/prompt_length_dual_axis.png`

## 📁 File Structure

```
experiments/
├── exp_resolution_scaling.py      # Resolution scaling experiment
├── exp_token_budget_ablation.py   # Token budget experiment
├── exp_stage_comparison.py        # Stage 2 vs 3 comparison
├── exp_prompt_length_effect.py    # Prompt length analysis
├── utils_plot.py                  # Plotting utilities
├── requirements.txt               # Python dependencies
├── run_all_experiments.sh         # Linux/Mac runner
├── run_all_experiments.bat        # Windows runner
└── README.md                      # This file

results/
├── resolution_scaling.csv         # Resolution experiment data
├── token_budget.csv              # Token budget experiment data
├── stage_compare.csv             # Stage comparison data
├── prompt_length.csv             # Prompt length experiment data
├── combined.csv                  # All experiments combined
├── experiment_summary.txt        # Text summary
└── plots/                        # All generated plots
    ├── resolution_vs_latency.png
    ├── resolution_vs_vram.png
    ├── token_budget_vs_ttft_resize.png
    ├── stage_comparison_ttft.png
    └── ...
```

## ⚙️ Configuration Options

All scripts support these common arguments:

| Argument         | Description                  | Default    |
| ---------------- | ---------------------------- | ---------- |
| `--model-path`   | Path to model checkpoint     | Required   |
| `--image-folder` | Folder with test images      | `./images` |
| `--device`       | Device (cuda/cpu)            | `cuda`     |
| `--batch-size`   | Batch size (keep 1 for VRAM) | `1`        |

### Stage Comparison Specific:

- `--stage2-path`: Path to Stage 2 model
- `--stage3-path`: Path to Stage 3 model

## 🔧 Memory Optimization

The scripts are optimized for RTX 4070 8GB:

- **FP16 precision**: `torch_dtype=torch.float16`
- **CUDA events**: Precise GPU timing
- **Memory cleanup**: `torch.cuda.empty_cache()` between runs
- **Small batches**: `batch_size=1` to minimize VRAM
- **Smart loading**: `low_cpu_mem_usage=True`

## 📊 Generated Metrics

### Resolution Scaling

- Total latency (ms)
- Time to First Token (TTFT) (ms)
- VRAM usage (GB)
- Tokens generated
- Resolution vs performance plots

### Token Budget Ablation

- TTFT measurements
- BLEU score vs Stage 3 reference
- Token budget vs accuracy trade-offs
- Resize vs center-crop comparison

### Stage Comparison

- Average TTFT per stage
- BLEU similarity between Stage 2 and 3
- Performance across different prompt types
- 5 images × 3 prompts analysis

### Prompt Length Effect

- Latency impact of prompt length
- Output length correlation
- Short (5), Medium (20), Long (60) word prompts
- Multiple prompt categories tested

## 🎨 Plot Types Generated

- **Line plots**: Trend analysis (resolution vs latency)
- **Dual-axis plots**: Compare two metrics simultaneously
- **Grouped bar charts**: Stage/method comparisons
- **Scatter plots**: Correlation analysis
- **Heatmaps**: Multi-dimensional comparisons

## 📈 Expected Results

Based on FastVLM paper findings:

- **Resolution Scaling**: Higher resolution → higher latency, diminishing returns
- **Token Budget**: Lower budgets → faster TTFT, potential accuracy loss
- **Stage Comparison**: Stage 2 faster but less accurate than Stage 3
- **Prompt Length**: Longer prompts → higher latency, more detailed outputs

## 🎯 Reproducing Paper Results

The experiments are designed to reproduce key findings from FastVLM CVPR 2025:

- **Table 4**: Resolution vs accuracy trade-offs
- **Table 5**: Stage comparison metrics
- **Figure 3**: Latency vs accuracy curves
- **Figure 4**: TTFT measurements

## 📝 Customization

### Adding New Experiments

1. Create new script following the pattern
2. Import `utils_plot` for consistent visualizations
3. Save results to `results/` directory
4. Add to runner scripts

## 🏆 Citation

```bibtex
@inproceedings{fastvlm2025,
  title={FastVLM: Efficient Vision-Language Model Inference via Token Reduction},
  author={Authors et al.},
  booktitle={CVPR},
  year={2025}
}
```
