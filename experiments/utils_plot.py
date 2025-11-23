"""
Utility functions for plotting FastVLM experiment results with LaTeX styling.
"""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure matplotlib for LaTeX-style plots
plt.rcParams.update(
    {
        # Use LaTeX for text rendering
        "text.usetex": False,  # Set to True if you have LaTeX installed
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 13,
        # Figure aesthetics
        "figure.figsize": (6, 4),  # Single column width for papers
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
        # Line and marker styles
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
        "axes.linewidth": 0.8,
        "grid.linewidth": 0.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        # Colors and styling
        "axes.edgecolor": "black",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

# Academic color palette - colorblind friendly
ACADEMIC_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def ensure_plot_dir(filename: str) -> str:
    """Ensure the plot directory exists and return full path."""
    plot_dir = os.path.join("results", "plots")
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, filename)


def save_plot(
    x: List[float],
    y: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    marker: str = "o-",
) -> None:
    """
    Save a LaTeX-style line plot.

    Args:
        x: X-axis values
        y: Y-axis values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename (without path)
        marker: Line style and marker
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        x,
        y,
        marker,
        color=ACADEMIC_COLORS[0],
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.2,
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)

    # Clean up the plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)

    # Force PNG extension if PDF was requested
    if full_path.endswith(".pdf"):
        full_path = full_path.replace(".pdf", ".png")

    plt.savefig(full_path, format="png", bbox_inches="tight")
    plt.close()
    print(f"📊 Plot saved: {full_path}")


def save_dual_axis_plot(
    x: List[float],
    y1: List[float],
    y2: List[float],
    xlabel: str,
    y1label: str,
    y2label: str,
    title: str,
    filename: str,
) -> None:
    """
    Save a LaTeX-style plot with dual y-axes.

    Args:
        x: X-axis values
        y1: Left y-axis values
        y2: Right y-axis values
        xlabel: X-axis label
        y1label: Left y-axis label
        y2label: Right y-axis label
        title: Plot title
        filename: Output filename
    """
    fig, ax1 = plt.subplots(figsize=(6, 4))

    color1 = ACADEMIC_COLORS[0]  # Blue
    color2 = ACADEMIC_COLORS[1]  # Orange

    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(y1label, color=color1)
    line1 = ax1.plot(
        x,
        y1,
        "o-",
        color=color1,
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=y1label,
    )
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel(y2label, color=color2)
    line2 = ax2.plot(
        x,
        y2,
        "s-",
        color=color2,
        linewidth=1.5,
        markersize=6,
        markerfacecolor="white",
        markeredgewidth=1.2,
        label=y2label,
    )
    ax2.tick_params(axis="y", labelcolor=color2)

    # Clean up spines
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)

    ax1.set_title(title, pad=10)
    fig.tight_layout()

    full_path = ensure_plot_dir(filename)

    # Force PNG extension if PDF was requested
    if full_path.endswith(".pdf"):
        full_path = full_path.replace(".pdf", ".png")

    plt.savefig(full_path, format="png", bbox_inches="tight")
    plt.close()
    print(f"📊 Dual-axis plot saved: {full_path}")


def save_grouped_bar(
    data: Dict[str, List[float]],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """
    Save a LaTeX-style grouped bar chart.

    Args:
        data: Dictionary where keys are group names and values are lists of values
        labels: X-axis labels for each group
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(7, 4.5))

    x = np.arange(len(labels))
    width = 0.25  # Narrower bars for cleaner look

    groups = list(data.keys())
    n_groups = len(groups)

    # Calculate bar positions
    bar_positions = []
    for i in range(n_groups):
        pos = x + (i - n_groups / 2 + 0.5) * width
        bar_positions.append(pos)

    # Create bars with academic colors
    for i, (group_name, values) in enumerate(data.items()):
        color = ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)]
        bars = ax.bar(
            bar_positions[i],
            values,
            width,
            label=group_name,
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5,
        )

        # Add value labels on bars (only if not too many bars)
        if len(values) <= 8:  # Avoid cluttering with too many labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:  # Only label positive values
                    ax.annotate(
                        f"{height:.1f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 2),  # 2 points vertical offset
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0 if len(max(labels, key=len)) < 8 else 45)

    # Clean legend
    ax.legend(frameon=False, loc="best")

    # Clean up spines and grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3, axis="y")

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)

    # Force PNG extension if PDF was requested
    if full_path.endswith(".pdf"):
        full_path = full_path.replace(".pdf", ".png")

    plt.savefig(full_path, format="png", bbox_inches="tight")
    plt.close()
    print(f"📊 Grouped bar chart saved: {full_path}")


def save_heatmap(
    data: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    title: str,
    filename: str,
    cmap: str = "RdYlBu_r",
) -> None:
    """
    Save a LaTeX-style heatmap visualization.

    Args:
        data: 2D numpy array
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Plot title
        filename: Output filename
        cmap: Colormap name (academic-friendly default)
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(data, cmap=cmap, aspect="auto")

    # Set ticks and labels
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Rotate the tick labels and set their alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel("Value", rotation=-90, va="bottom")

    # Add text annotations
    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            text = ax.text(
                j,
                i,
                f"{data[i, j]:.2f}",
                ha="center",
                va="center",
                color="black" if data[i, j] > data.mean() else "white",
                fontsize=9,
            )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)

    # Force PNG extension if PDF was requested
    if full_path.endswith(".pdf"):
        full_path = full_path.replace(".pdf", ".png")

    plt.savefig(full_path, format="png", bbox_inches="tight")
    plt.close()
    print(f"📊 Heatmap saved: {full_path}")


def save_scatter_plot(
    x: List[float],
    y: List[float],
    labels: List[str],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    """
    Save a LaTeX-style scatter plot with labels.

    Args:
        x: X-axis values
        y: Y-axis values
        labels: Point labels
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(6, 4.5))

    # Use academic colors for scatter points
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(len(x))]
    scatter = ax.scatter(
        x, y, s=80, alpha=0.7, c=colors, edgecolors="white", linewidth=0.5
    )

    # Add labels to points (with smart positioning)
    for i, label in enumerate(labels):
        ax.annotate(
            label,
            (x[i], y[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"
            ),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)

    # Clean up spines and grid
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)

    # Force PNG extension if PDF was requested
    if full_path.endswith(".pdf"):
        full_path = full_path.replace(".pdf", ".png")

    plt.savefig(full_path, format="png", bbox_inches="tight")
    plt.close()
    print(f"📊 Scatter plot saved: {full_path}")


def save_correlation_plot(
    x: List[float],
    y: List[float],
    xlabel: str,
    ylabel: str,
    title: str,
    filename: str,
    show_correlation: bool = True,
) -> None:
    """
    Save a correlation plot with trend line.

    Args:
        x: X-axis values
        y: Y-axis values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
        show_correlation: Whether to show correlation coefficient
    """
    fig, ax = plt.subplots(figsize=(6, 4))

    # Scatter plot
    ax.scatter(
        x,
        y,
        color=ACADEMIC_COLORS[0],
        alpha=0.7,
        s=60,
        edgecolors="white",
        linewidth=0.5,
    )

    # Add trend line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x, p(x), color=ACADEMIC_COLORS[1], linestyle="--", linewidth=1.5, alpha=0.8)

    # Calculate and display correlation if requested
    if show_correlation:
        corr = np.corrcoef(x, y)[0, 1]
        ax.text(
            0.05,
            0.95,
            f"r = {corr:.3f}",
            transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment="top",
            fontsize=10,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)

    # Clean styling
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)

    # Force PNG extension if PDF was requested
    if full_path.endswith(".pdf"):
        full_path = full_path.replace(".pdf", ".png")

    plt.savefig(full_path, format="png", bbox_inches="tight")
    plt.close()
    print(f"📊 Correlation plot saved: {full_path}")
