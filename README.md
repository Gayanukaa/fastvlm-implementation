# FastVLM: Research Implementation & Extensions

This repository contains our research implementation and extensions of **[FastVLM: Efficient Vision Encoding for Vision Language Models](https://www.arxiv.org/abs/2412.13303)** (CVPR 2025) by Apple Inc.

## Overview

**Original FastVLM** introduces FastViTHD, a hybrid vision encoder that outputs ~100 tokens per image (vs 576 for CLIP).

**Our Study:**

1. **Experiment Replications** - Conduct 4 key experiments from the paper
2. **Video Fine-tuning Adaptation** - Extended FastVLM for video-text training
3. **Windows Inference Engine** - Real-time inference with live webcam support

## Quick Start

### Installation

```bash
conda create -n fastvlm python=3.10
conda activate fastvlm
pip install -e .
```

### Download Models

```bash
bash get_models.sh  # Downloads to checkpoints/
```

## 1) Experiment Replications

We replicated 4 experiments from the FastVLM paper to validate the reported results.

### Replicated Experiments

| Experiment            | Paper Ref | Description                                  |
| --------------------- | --------- | -------------------------------------------- |
| Encoder Comparison    | Table 3   | ViT-L/14 vs ConvNeXt-L vs FastViT-HD latency |
| Token Efficiency      | Table 4   | Visual tokens across encoders & resolutions  |
| Resolution Scaling    | Table 5   | FastViT-HD @ 256-1024px                      |
| Model Size Comparison | Table 11  | FastVLM 0.5B vs 1.5B                         |

**Details:** See [`experiments/README.md`](experiments/README.md)

## 2) Video Fine-tuning Extension

We adapted FastVLM to support video-text pair training using sparse temporal sampling.

### Key Modifications

- **Video Frame Extraction**: Uniform sampling of N frames using `decord`
- **Token Expansion**: `<image>` → `<image><image><image>...` (N times)
- **Efficient Processing**: ~800 tokens for 8 frames vs 4,608 for CLIP-based models

### Usage

```bash
cd scripts
bash finetune_video.sh
```

**Note:** Full fine-tuning implementation available at [EdgeVLM-Labs/fastvlm-adaptation](https://github.com/EdgeVLM-Labs/fastvlm-adaptation) as it modifies core training code in `llava/`.

<p align="center">
<img src="docs/Finetune Graphs Collection.png"/>
<br>
<em>Training metrics from video fine-tuning showing loss convergence and performance improvements</em>
</p>

**Details:** See [`scripts/finetune.md`](scripts/finetune.md)

## 3) Windows Inference Engine

Interactive Gradio app with real-time webcam inference and optimized performance.

<p align="center">
<img src="docs/Windows Inference Engine.png"/>
</p>

### Features

- **Dual Modes**: Chat (image upload) + Live (webcam streaming)
- **Performance Optimizations**:
  - Prompt caching for repeated queries
  - Frame skipping (adjustable 1-10x)
  - TF32 acceleration on Ampere+ GPUs
- **Real-time Metrics**: TTFT, tokens/sec, avg/min/max latency, FPS

### Run

```bash
python windows_app/app.py
```

**Details:** See [`windows_app/README.md`](windows_app/README.md)

## License & Attribution

### Original Work

FastVLM is developed by Apple Inc. and released under the [Apple Sample Code License](LICENSE).

**Citation:**

```bibtex
@InProceedings{fastvlm2025,
  author = {Pavan Kumar Anasosalu Vasu, Fartash Faghri, Chun-Liang Li,
            Cem Koc, Nate True, Albert Antony, Gokul Santhanam,
            James Gabriel, Peter Grasch, Oncel Tuzel, Hadi Pouransari},
  title = {FastVLM: Efficient Vision Encoding for Vision Language Models},
  booktitle = {CVPR},
  year = {2025}
}
```

### Our Research Extensions

The modifications in this repository (experiment replications, video fine-tuning adaptation, and Windows inference engine) are provided for **research and educational purposes only**, in accordance with the Apple Sample Code License which permits use, modification, and redistribution for non-commercial research.

Please refer to the [LICENSE](LICENSE) file for full terms.
