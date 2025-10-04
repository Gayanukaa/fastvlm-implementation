"""
Utility functions for plotting FastVLM experiment results.
"""
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from typing import List, Dict, Any

# Set style for consistent plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def ensure_plot_dir(filename: str) -> str:
    """Ensure the plot directory exists and return full path."""
    plot_dir = os.path.join('results', 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    return os.path.join(plot_dir, filename)

def save_plot(x: List[float], y: List[float], xlabel: str, ylabel: str,
              title: str, filename: str, marker: str = 'o-') -> None:
    """
    Save a simple line plot.

    Args:
        x: X-axis values
        y: Y-axis values
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename (without path)
        marker: Line style and marker
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker, linewidth=2, markersize=8)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    full_path = ensure_plot_dir(filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Plot saved: {full_path}")

def save_dual_axis_plot(x: List[float], y1: List[float], y2: List[float],
                       xlabel: str, y1label: str, y2label: str,
                       title: str, filename: str) -> None:
    """
    Save a plot with dual y-axes.

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
    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel(y1label, color=color1, fontsize=12)
    ax1.plot(x, y1, 'o-', color=color1, linewidth=2, markersize=8, label=y1label)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.set_ylabel(y2label, color=color2, fontsize=12)
    ax2.plot(x, y2, 's-', color=color2, linewidth=2, markersize=8, label=y2label)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()

    full_path = ensure_plot_dir(filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Dual-axis plot saved: {full_path}")

def save_grouped_bar(data: Dict[str, List[float]], labels: List[str],
                    xlabel: str, ylabel: str, title: str, filename: str) -> None:
    """
    Save a grouped bar chart.

    Args:
        data: Dictionary where keys are group names and values are lists of values
        labels: X-axis labels for each group
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(labels))
    width = 0.35

    groups = list(data.keys())
    n_groups = len(groups)

    # Calculate bar positions
    bar_positions = []
    for i in range(n_groups):
        pos = x + (i - n_groups/2 + 0.5) * width
        bar_positions.append(pos)

    # Create bars
    colors = plt.cm.Set3(np.linspace(0, 1, n_groups))
    for i, (group_name, values) in enumerate(data.items()):
        bars = ax.bar(bar_positions[i], values, width, label=group_name,
                     color=colors[i], alpha=0.8)

        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    full_path = ensure_plot_dir(filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Grouped bar chart saved: {full_path}")

def save_heatmap(data: np.ndarray, row_labels: List[str], col_labels: List[str],
                title: str, filename: str, cmap: str = 'viridis') -> None:
    """
    Save a heatmap visualization.

    Args:
        data: 2D numpy array
        row_labels: Labels for rows
        col_labels: Labels for columns
        title: Plot title
        filename: Output filename
        cmap: Colormap name
    """
    plt.figure(figsize=(10, 8))

    sns.heatmap(data, annot=True, fmt='.3f', cmap=cmap,
                xticklabels=col_labels, yticklabels=row_labels,
                cbar_kws={'shrink': 0.8})

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Columns', fontsize=12)
    plt.ylabel('Rows', fontsize=12)
    plt.tight_layout()

    full_path = ensure_plot_dir(filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Heatmap saved: {full_path}")

def save_scatter_plot(x: List[float], y: List[float], labels: List[str],
                     xlabel: str, ylabel: str, title: str, filename: str) -> None:
    """
    Save a scatter plot with labels.

    Args:
        x: X-axis values
        y: Y-axis values
        labels: Point labels
        xlabel: X-axis label
        ylabel: Y-axis label
        title: Plot title
        filename: Output filename
    """
    plt.figure(figsize=(10, 7))

    scatter = plt.scatter(x, y, s=100, alpha=0.7, c=range(len(x)), cmap='viridis')

    # Add labels to points
    for i, label in enumerate(labels):
        plt.annotate(label, (x[i], y[i]), xytext=(5, 5),
                    textcoords='offset points', fontsize=9)

    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.colorbar(scatter, label='Data Point Index')
    plt.tight_layout()

    full_path = ensure_plot_dir(filename)
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Scatter plot saved: {full_path}")