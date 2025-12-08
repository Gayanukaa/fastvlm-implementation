#!/usr/bin/env python3
"""
Regenerate plots from existing CSV results using the updated utils_plot.py.
This ensures all plots are generated as PNGs (no PDFs) as requested.
"""

import os
import sys
import warnings

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils_plot import save_dual_axis_plot, save_grouped_bar, save_plot

warnings.filterwarnings("ignore")


def regenerate_resolution_scaling():
    csv_path = "results/resolution_scaling_textvqa.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Skipping.")
        return

    print(f"🔄 Regenerating Resolution Scaling plots from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("  ⚠️ CSV is empty.")
        return

    res_vals = df["resolution"].tolist()
    lat_vals = df["latency"].tolist()
    vram_vals = df["vram"].tolist()

    save_dual_axis_plot(
        res_vals,
        lat_vals,
        vram_vals,
        "Resolution (px)",
        "Latency (ms)",
        "VRAM (GB)",
        "Resolution Scaling (TextVQA)",
        "resolution_scaling_textvqa.png",
    )


def regenerate_stage_comparison():
    csv_path = "results/stage_comparison_textvqa.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Skipping.")
        return

    print(f"🔄 Regenerating Stage Comparison plots from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("  ⚠️ CSV is empty.")
        return

    # Group by stage
    stage2_res = df[df["stage"] == "Stage 2"]
    stage3_res = df[df["stage"] == "Stage 3"]

    metrics = {
        "Latency (ms)": [stage2_res["latency"].mean(), stage3_res["latency"].mean()],
        "Accuracy (Exact Match)": [
            stage2_res["accuracy"].mean(),
            stage3_res["accuracy"].mean(),
        ],
    }

    # Plot Latency
    save_grouped_bar(
        {
            "Stage 2": [metrics["Latency (ms)"][0]],
            "Stage 3": [metrics["Latency (ms)"][1]],
        },
        ["Average"],
        "Stage",
        "Latency (ms)",
        "Stage Comparison - Latency",
        "stage_comparison_latency.png",
    )

    # Plot Accuracy
    save_grouped_bar(
        {
            "Stage 2": [metrics["Accuracy (Exact Match)"][0]],
            "Stage 3": [metrics["Accuracy (Exact Match)"][1]],
        },
        ["Average"],
        "Stage",
        "Accuracy",
        "Stage Comparison - Accuracy",
        "stage_comparison_accuracy.png",
    )


def regenerate_token_budget():
    csv_path = "results/token_budget_textvqa.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Skipping.")
        return

    print(f"🔄 Regenerating Token Budget plots from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("  ⚠️ CSV is empty.")
        return

    budgets_val = df["budget"].tolist()
    lat_val = df["latency"].tolist()
    acc_val = df["accuracy"].tolist()

    save_dual_axis_plot(
        budgets_val,
        lat_val,
        acc_val,
        "Token Budget",
        "Latency (ms)",
        "Accuracy",
        "Token Budget Ablation (TextVQA)",
        "token_budget_ablation.png",
    )


def regenerate_prompt_length():
    csv_path = "results/prompt_length_textvqa.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️ {csv_path} not found. Skipping.")
        return

    print(f"🔄 Regenerating Prompt Length plots from {csv_path}...")
    df = pd.read_csv(csv_path)

    if df.empty:
        print("  ⚠️ CSV is empty.")
        return

    # Ensure correct order if possible
    category_order = ["Short", "Medium", "Long"]
    df["category"] = pd.Categorical(
        df["category"], categories=category_order, ordered=True
    )
    df = df.sort_values("category")

    tokens_val = df["avg_tokens"].tolist()
    lat_val = df["latency"].tolist()
    ttft_val = df["ttft"].tolist()

    save_dual_axis_plot(
        tokens_val,
        lat_val,
        ttft_val,
        "Prompt Tokens",
        "Total Latency (ms)",
        "TTFT (ms)",
        "Prompt Length Effect (TextVQA)",
        "prompt_length_effect.png",
    )


if __name__ == "__main__":
    print("🎨 Regenerating all plots as PNGs...")
    regenerate_resolution_scaling()
    regenerate_stage_comparison()
    regenerate_token_budget()
    regenerate_prompt_length()
    print("✅ Done!")
