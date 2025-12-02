# ⚠️ Replication Limitations – FastVLM Paper

This document summarizes the tables from the **FastVLM paper** that **cannot be experimentally replicated** in this project, along with the exact reasons and constraints.

The project focuses on replicating evaluation components that are possible with the available checkpoint and hardware. The remaining tables depend on resources or model access beyond this environment.

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

## ✔️ Replicable Tables Summary

| Table                     | Status                    | Notes                                             |
| ------------------------- | ------------------------- | ------------------------------------------------- |
| Table 3                   | ✔️ Replicable             | Encoder latency & parameter comparison            |
| Table 4                   | ✔️ Partial                | Evaluation on visible benchmarks (subset)         |
| Table 5                   | ✔️ Partial (Mini Version) | Token efficiency trends only                      |
| Tables 1, 2, 6, 7, 10, 11 | ❌ Not Replicable         | Due to limited compute, datasets, or model access |

### 📌 Statement for Thesis

> _"Due to restricted access to external models, datasets, and large-scale compute infrastructure, several evaluation tables from the original FastVLM paper cannot be reproduced. Instead, this work focuses on replicating the portions that can be executed with the publicly available checkpoint and hardware limitations, and reports partial benchmarks where appropriate."_
