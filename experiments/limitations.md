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
- GQA benchmark is disabled due to HuggingFace dataset loading issues
- Accuracy metric uses relaxed matching (not official VQA evaluation protocol)
- MAX_SAMPLES is set to 10 for quick testing; full dataset evaluation requires more time

</details>

<details>
<summary><b>✔️ Table 5 — FastViT-HD Visual Token Efficiency (Mini Version)</b></summary>

**Script:** `exp_table5.py` / `make_table5_fastvlm.py`

**What It Replicates:**  
Demonstrates FastViT-HD's visual token efficiency across different resolutions with accuracy measurement.

**Resolutions:** 256, 512, 768, 1024px (FastViT-HD only)

**Metrics:**

- Visual token count (based on 64x downsampling factor)
- TextVQA accuracy (%)

**Output:** `results/plots/table5_fastvithd_efficiency.png`

**Limitations:**

1. **No pruning baselines:** The original Table 5 compares FastViT-HD to several token pruning/sparsification methods from other papers. Those baseline results were taken directly from the respective papers and were not re-trained by the FastVLM authors. We do not re-implement or re-train those pruning methods.

2. **Single checkpoint limitation:** We only have one fine-tuned checkpoint (`llava-fastvithd_0.5b_stage3`). The accuracy shown is from this checkpoint. The visual token counts are calculated based on the 64x downsampling factor of FastViT-HD.

3. **Hardware and data limitations:** The original paper uses its own hardware configuration and full benchmark datasets. Our replication may use a subset of samples for quick testing. Therefore, our accuracy numbers may differ from the paper.

</details>

<details>
<summary><b>✔️ Figure — Resolution Scaling (Latency & VRAM)</b></summary>

**Script:** `exp_resolution_scaling.py`

**What It Replicates:**  
Tests how inference latency and VRAM usage scale with input resolution using TextVQA samples.

**Resolutions Tested:** 224, 336, 448, 512, 672, 768, 896, 1024px

**Metrics:**

- Inference latency (ms)
- Peak VRAM usage (MB)

**Output:** `results/plots/resolution_scaling_textvqa.png`

**Limitations:**

- Hardware-specific measurements (RTX 4070 8GB)
- VRAM measurements may vary based on CUDA version and driver
- Uses a subset of TextVQA samples for efficiency

</details>

<details>
<summary><b>✔️ Figure — Token Budget Ablation (Accuracy vs Latency)</b></summary>

**Script:** `exp_token_budget_ablation.py`

**What It Replicates:**  
Evaluates the trade-off between accuracy and latency by varying the visual token budget via resolution scaling.

**Token Budgets:** 16, 64, 144, 256, 576 tokens

**Metrics:**

- TextVQA accuracy (Exact Match %)
- Inference latency (ms)

**Output:** `results/plots/token_budget_textvqa.png`

**Limitations:**

- Single checkpoint evaluation
- Accuracy metric uses exact match comparison
- Token budgets are approximated via resolution (actual token count depends on model architecture)

</details>

<details>
<summary><b>✔️ Figure — Stage Comparison (Stage-2 vs Stage-3)</b></summary>

**Script:** `exp_stage_comparison.py`

**What It Replicates:**  
Compares performance between Stage-2 (pre-training) and Stage-3 (fine-tuned) checkpoints on TextVQA.

**Models Compared:**

- `llava-fastvithd_0.5b_stage2`
- `llava-fastvithd_0.5b_stage3`

**Metrics:**

- TextVQA accuracy (Exact Match %)
- Time-to-First-Token (TTFT)
- Total inference latency

**Output:** `results/plots/stage_comparison_textvqa.png`

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
