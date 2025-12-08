# FastVLM Windows Inference App

This is a Gradio-based interactive application for running FastVLM inference on Windows (or Linux).

## Features

- **Fast TTFT**: Optimized for low latency time-to-first-token.
- **Model Switching**: Switch between Stage 2 and Stage 3 models.
- **Performance Metrics**: Real-time display of TTFT and TPS (Tokens Per Second).

## Prerequisites

- Python 3.10+
- CUDA-enabled GPU (recommended)
- Dependencies installed (see root `pyproject.toml` or run `setup.sh`)

## How to Run

1.  Ensure you are in the root directory of the repository.
2.  Run the app:
    ```bash
    python windows_app/app.py
    ```
3.  Open the displayed URL in your browser (usually `http://127.0.0.1:7860`).

## Models

The app expects the following model checkpoints in the `checkpoints/` directory:

- `checkpoints/llava-fastvithd_0.5b_stage2`
- `checkpoints/llava-fastvithd_0.5b_stage3`

If your models are in a different location, please update the `MODELS` dictionary in `windows_app/app.py`.
