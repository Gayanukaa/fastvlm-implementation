# ⚠️ Replication Limitations – FastVLM Paper

This document summarizes the tables from the **FastVLM paper** that **cannot be experimentally replicated** in this project, along with the exact reasons and constraints.

The project focuses on replicating evaluation components that are possible with the available checkpoint and hardware. The remaining tables depend on resources or model access beyond this environment.

## Non Replicable Tables Summary

<details>
<summary><b>❌ Table 1 — Model Architecture Summary</b></summary>

**Reason Not Replicable:**  
This table contains internal model configuration details, architectural component sizes, and training design metadata. These values are not generated through inference and require direct access to the authors' internal configuration files.

**Limitation:**  
Architectural metadata used during pretraining is not publicly available and cannot be derived from inference-only checkpoints.

</details>

<details>
<summary><b>❌ Table 2 — Pretraining and Instruction-Tuning Dataset Details</b></summary>

**Reason Not Replicable:**  
The table lists dataset compositions and scale (millions to billions of samples). Many datasets are public only in partial form, privately curated, or filtered based on undisclosed criteria.

**Limitation:**  
Reconstructing this dataset pipeline requires large-scale compute, storage, and dataset licensing access not available in this environment.

</details>

<details>
<summary><b>❌ Table 6 — Full Vision-Language Benchmark Comparisons</b></summary>

**Reason Not Replicable:**  
This table compares FastVLM against numerous other VLMs (LLaVA variants, Cambrian, ConvLLaVA, etc.) across many benchmarks such as TextVQA, POPE, SEED, DocVQA, and GQA.

**Important Note:**  
Even the FastVLM authors **did not re-run all models**. Many values were **taken directly from original papers**.

**Limitation:**  
The external baseline models are not accessible as runnable checkpoints, and evaluating them would require a multi-GPU setup (Vicuna-7B scale or larger).

</details>

<details>
<summary><b>❌ Table 7 — Image Corruption Robustness</b></summary>

**Reason Not Replicable:**  
This evaluation requires multiple models and full inference pipelines to test robustness under blur, noise, compression, occlusion, and similar transformations.

**Limitation:**  
Only a single checkpoint (FastVLM Stage-3) is available locally. Robustness testing requires multiple comparable VLMs and the full corruption evaluation framework.

</details>

<details>
<summary><b>❌ Table 10 — Generalization Across Diverse Benchmarks</b></summary>

**Reason Not Replicable:**  
This table evaluates performance across a broad range of vision-language reasoning benchmarks using multiple external VLMs.

**Limitation:**  
The required models, evaluation scripts, and multimodal benchmarks are not accessible or runnable under the current hardware environment.

</details>

<details>
<summary><b>❌ Table 11 — Multimodal Reasoning and Alignment Evaluation</b></summary>

**Reason Not Replicable:**  
This table measures fine-grained alignment, reasoning, and multimodal understanding across multiple models.

**Limitation:**  
Access to multiple tuned VLMs, proprietary datasets, and large-scale model evaluation pipelines is required, which is not feasible on available hardware.

</details>

## Replicable Tables & Figures Summary

<details>
<summary><b>✔️ Table 3 — Encoder Latency & Parameter Comparison</b></summary>

**Script:** `exp_table3.py`

**What It Replicates:**  
Compares encoder architectures (ViT-L/14, ConvNeXt-L, FastViT-HD) in terms of parameter count and inference latency at their respective input resolutions.

**Metrics:**

- Encoder Size (M) — parameters in millions
- Input Resolution (px)
- Latency (ms/image)

**Output:** `results/plots/table3_encoder_comparison.png`

**Limitations:**

- Uses locally downloaded encoder weights (from `encoder_models/`) rather than the original paper's exact weights
- Latency values are hardware-dependent (measured on RTX 4070 8GB vs paper's hardware)
- Exact numerical match to paper values is not guaranteed due to hardware and framework differences

</details>

<details>
<summary><b>✔️ Table 4 — Visual Token Efficiency (Partial)</b></summary>

**Script:** `exp_table4.py`

**What It Replicates:**  
Evaluates multiple encoder/resolution combinations for visual token efficiency, measuring accuracy on VQA benchmarks.

**Encoder Configurations:**

- FastViT-HD: 256, 512, 768, 1024px
- ConvNeXt-L: 320, 512px

**Metrics:**

- Visual token count (based on downsampling factor)
- Inference latency (ms)
- TextVQA accuracy (%)
- DocVQA accuracy (%)
- GQA accuracy (%) — currently disabled due to dataset issues

**Output:** `results/plots/table4_visual_token_efficiency.png`

**Limitations:**

- Only one checkpoint is available (`llava-fastvithd_0.5b_stage3`), so accuracy values are the same across all encoder configurations
- To properly replicate Table 4, separate fine-tuned checkpoints for each encoder/resolution would be required
- GQA benchmark is disabled — lmms-lab/GQA requires joining separate image and instruction configs, which defeats streaming and requires downloading the full image dataset (~10GB)
- Accuracy metric uses relaxed matching (not official VQA evaluation protocol)
- MAX_SAMPLES is set to 10 for quick testing; full dataset evaluation requires more time
- Datasets: lmms-lab/textvqa (streaming), lmms-lab/DocVQA (streaming)

</details>

<details>
<summary><b>✔️ Table 5 — FastViT-HD Visual Token Efficiency (Mini Version)</b></summary>

**Script:** `exp_table5.py`

**What It Replicates:**  
Demonstrates FastViT-HD's visual token efficiency across different resolutions by resizing input images and measuring accuracy at each resolution separately.

**Resolutions:** 256, 512, 768, 1024px (FastViT-HD only)

**Metrics:**

- Visual token count (based on 64x downsampling factor): (resolution / 64)²
- TextVQA accuracy (%) — evaluated separately at each resolution

**Output:** `results/plots/table5_fastvithd_efficiency.png`

**Implementation Notes:**

- Images are resized to each target resolution (256×256, 512×512, etc.) before inference
- Accuracy is evaluated independently at each resolution, not shared across resolutions
- Visual tokens calculated as: 16 (256px), 64 (512px), 144 (768px), 256 (1024px)

**Limitations:**

1. **No pruning baselines:** The original Table 5 compares FastViT-HD to several token pruning/sparsification methods from other papers. Those baseline results were taken directly from the respective papers and were not re-trained by the FastVLM authors. We do not re-implement or re-train those pruning methods.

2. **Single checkpoint limitation:** We only have one fine-tuned checkpoint (`llava-fastvithd_0.5b_stage3`), which was trained at a specific resolution. In the actual paper, the authors would have either:

   - Trained separate models at each resolution, OR
   - Modified the vision encoder to accept different input resolutions dynamically

   Our replication resizes input images to each resolution, but the model's internal `image_processor` may further resize to the model's expected input size, limiting the true effect of resolution variation.

3. **Hardware and data limitations:** The original paper uses its own hardware configuration and full benchmark datasets. Our replication uses a subset of samples (`MAX_SAMPLES=10` for quick testing). Full dataset evaluation requires setting `MAX_SAMPLES=None`.

4. **Accuracy variation caveat:** While we do observe some accuracy variation across resolutions due to image resizing effects, this may not fully reflect the paper's methodology where models were potentially trained or configured for each specific resolution.

</details>

<details>
<summary><b>✔️ Figure 5 — Resolution Scaling (Vision vs LLM Prefilling Latency)</b></summary>

**Script:** `exp_figure5.py`

**What It Replicates:**  
Measures how Vision Encoder latency and LLM Prefilling latency scale with input resolution. This replicates Figure 5 from the paper which shows a grouped bar chart comparing these two components.

**Resolutions Tested:** 256, 512, 768, 1024, 1536px (matching the paper)

**Metrics:**

- Vision Latency (ms) — time for vision encoder + projector (GPU forward pass only)
- LLM Prefilling Latency (ms) — time for LLM forward pass (no token generation)

**Output:** `results/plots/figure5_resolution_scaling.png`

**Implementation Notes:**

- **GPU-only timing:** CPU preprocessing (`process_images`) and data transfer (`.to(device)`) are done OUTSIDE the timer. Only pure GPU forward pass time is measured.
- **Forced image processor resolution:** The script overrides the image processor's `crop_size` and `size` settings for each resolution to ensure the model actually processes the target resolution (not a default size).
- Vision latency is measured by calling `model.encode_images()` which runs the vision tower and projection layer
- LLM prefilling latency is measured by running a single forward pass through `model.model()` with the prepared multimodal embeddings
- Decoding/token generation time is **excluded** (unlike `model.generate()`)

**Visual Token Scaling:**

The number of visual tokens scales with `(resolution / 64)²` due to FastViT-HD's 64x downsampling:

- 256px → 16 tokens
- 512px → 64 tokens
- 768px → 144 tokens
- 1024px → 256 tokens
- 1536px → 576 tokens

**Limitations:**

- **Hardware differences affect latency ratios:**
  - **Paper (Apple M1 Max):** The "LLM Prefilling" (orange bars) appears relatively slow because the M1 has lower memory bandwidth compared to server GPUs. This makes the prefill latency look significant compared to vision latency.
  - **NVIDIA GPU (our setup):** On high-end GPUs with higher memory bandwidth, LLM prefilling is extremely fast. The orange bars (prefill) may appear much smaller relative to the blue bars (vision) compared to the paper's Figure 5.
- FastViT-HD's efficient 64x downsampling architecture may show different scaling patterns compared to other vision encoders (e.g., ViT-L/14)
- CUDA synchronization overhead may slightly affect timing measurements
- Uses a subset of TextVQA samples for efficiency (`--num-samples` flag)

</details>

<details>
<summary><b>✔️ Figure 4 — Pareto Curve (Avg-5 Score vs TTFT)</b></summary>

**Script:** `exp_stage_comparison.py --replication-target figure4`

**What It Replicates:**  
Generates the Pareto curve showing the trade-off between accuracy (Avg-5 Score) and latency (TTFT). The script automatically loops through multiple resolutions (256, 512, 768, 1024) and plots them as a line graph.

**Usage:**

```bash
# Automatic multi-resolution sweep + Pareto curve generation
python exp_stage_comparison.py --replication-target figure4 --num-samples 100

# Quick test with fewer samples
python exp_stage_comparison.py --replication-target figure4 --num-samples 10
```

**Implementation Notes:**

The script automatically:

1. Loops through resolutions: [256, 512, 768, 1024]px
2. At each resolution:
   - Forces image processor to target resolution
   - Evaluates on Avg-5 benchmarks (TextVQA, DocVQA)
   - Measures average TTFT (time-to-first-token)
3. Saves individual CSV results per resolution
4. Generates Pareto curve plot with:
   - X-axis: Latency (ms) — log scale
   - Y-axis: Avg-5 Score (%)
   - Annotations showing resolution at each point

**Benchmarks (Avg-5 subset):** TextVQA, DocVQA  
_(Full Avg-5 in paper: TextVQA, DocVQA, ChartQA, AI2D, InfoVQA)_

**Metrics:**

- Avg-N Score (%) — average accuracy across available benchmarks
- Avg TTFT (ms) — average time-to-first-token (GPU-only timing)

**Output:**

- `results/figure4_{resolution}px_results.csv` — per-resolution data
- `results/plots/figure4_pareto_curve.png` — Pareto curve visualization

**Limitations:**

- Only 2 of 5 benchmarks available (TextVQA, DocVQA) — ChartQA, AI2D, InfoVQA not implemented
- Resolution override forces image processor to target size, but model was trained at specific resolution
- Accuracy uses exact match, not official VQA/ANLS metrics
- TTFT measured on NVIDIA GPU (paper uses Apple M1 Max for Figure 4)
- FastViT-HD visual tokens scale as (resolution // 64)² affecting prefill cost

</details>

<details>
<summary><b>✔️ Table 6 — VLM Benchmark Comparison (Partial)</b></summary>

**Script:** `exp_stage_comparison.py --replication-target table6`

**What It Replicates:**  
Evaluates FastVLM on the Avg-5 benchmark suite used in Table 6 for comparing VLM performance.

**Usage:**

```bash
python exp_stage_comparison.py --replication-target table6 --resolution 1024 --num-samples 100
```

**Benchmarks (Avg-5 subset):** TextVQA, DocVQA

**Metrics:**

- Per-benchmark accuracy (%)
- Avg-N Score (%)
- Inference latency (ms)

**Output:** `results/table6_{resolution}px_results.csv`

**Limitations:**

- Only 2 of 5 benchmarks available — full Table 6 requires ChartQA, AI2D, InfoVQA
- Paper compares multiple models (LLaVA variants, Cambrian, etc.) — we only have FastVLM checkpoint
- Accuracy uses exact match as a proxy for official metrics

</details>

<details>
<summary><b>✔️ Table 11 — Text-Rich Benchmark Evaluation (Partial)</b></summary>

**Script:** `exp_stage_comparison.py --replication-target table11`

**What It Replicates:**  
Evaluates FastVLM on text-rich benchmarks at specific resolutions as shown in Table 11.

**Usage:**

```bash
python exp_stage_comparison.py --replication-target table11 --resolution 1024 --num-samples 100
python exp_stage_comparison.py --replication-target table11 --resolution 1152 --num-samples 100
```

**Benchmarks (Text-Rich subset):** TextVQA, DocVQA  
_(Full set in paper: TextVQA, DocVQA, ChartQA, InfoVQA, OCRBench)_

**Metrics:**

- Per-benchmark accuracy (%)
- Inference latency (ms)

**Output:** `results/table11_{resolution}px_results.csv`

**Limitations:**

- Only 2 of 5 text-rich benchmarks available
- Resolution override may not fully replicate paper's multi-resolution training
- Official ANLS metric for DocVQA is approximated with exact match

</details>

<details>
<summary><b>✔️ Stage Comparison (Stage-2 vs Stage-3)</b></summary>

**Script:** `exp_stage_comparison.py --replication-target stage-comparison`

**What It Replicates:**  
Compares performance between Stage-2 (pre-training) and Stage-3 (fine-tuned) checkpoints on TextVQA.

**Usage:**

```bash
python exp_stage_comparison.py --replication-target stage-comparison --stage2-path ../checkpoints/llava-fastvithd_0.5b_stage2 --stage3-path ../checkpoints/llava-fastvithd_0.5b_stage3 --num-samples 20
```

**Models Compared:**

- `llava-fastvithd_0.5b_stage2`
- `llava-fastvithd_0.5b_stage3`

**Metrics:**

- TextVQA accuracy (Exact Match %)
- Inference latency (ms)

**Output:** `results/stage_comparison_textvqa.csv`, `results/plots/stage_comparison_*.png`

**Limitations:**

- Only 0.5B model variants are compared (1.5B and 7B require more VRAM)
- Accuracy metric uses exact match, not official VQA evaluation

</details>

<details>
<summary><b>✔️ Figure — Prompt Length Effect (Latency vs Prompt Size)</b></summary>

**Script:** `exp_prompt_length_effect.py`

**What It Replicates:**  
Analyzes how different prompt lengths affect Time-to-First-Token (TTFT) and total inference latency.

**Prompt Categories:** Short, Medium, Long (with context prefixes)

**Metrics:**

- Time-to-First-Token (TTFT) in ms
- Total latency (ms)
- Prompt token count

**Output:** `results/plots/prompt_length_textvqa.png`

**Limitations:**

- Prompt length categories are synthetically extended from TextVQA questions
- Results are specific to the Qwen-2 conversation template used by FastVLM

</details>

---

### 📌 Statement for Thesis

> _"Due to restricted access to external models, datasets, and large-scale compute infrastructure, several evaluation tables from the original FastVLM paper cannot be reproduced. Instead, this work focuses on replicating the portions that can be executed with the publicly available checkpoint and hardware limitations, and reports partial benchmarks where appropriate."_
