"""Clustered bar chart of test accuracy: noisy vs. without-noise, per model and dataset.

Reads ``outputs/metrics.json`` (written by ``train.py``) and saves
``outputs/accuracy_comparison.png``: OpenSource on the left, TwoWheeler on the right, with
one cluster per model (SVM, ANN, 1D-CNN) and two bars per cluster (noisy / without noise).

Run with:  uv run plot_accuracy.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "outputs" / "metrics.json"
OUT = ROOT / "outputs" / "accuracy_comparison.png"

# metrics.json model key -> short label on the x-axis (Logistic / Softmax is left out).
MODELS = {"SVM (RBF kernel)": "SVM", "ANN (MLP)": "ANN", "1D-CNN": "1D-CNN"}
DATASETS = {"OpenSource": "OpenSource", "TwoWheeler": "Two-Wheeler"}
NOISY_COLOR = "#0b3d7a"  # dark navy bar
CLEAN_COLOR = "#e3edf9"  # light bar with navy outline
Y_MIN = 90.0  # accuracies lie in 94-100 %; a 0-based axis would hide the differences


def main() -> None:
    metrics: dict[str, dict[str, dict[str, object]]] = json.loads(METRICS.read_text())

    fig = Figure(figsize=(12, 5.2), layout="constrained")
    axes = fig.subplots(1, 2, sharey=True)
    x = np.arange(len(MODELS))
    width = 0.36

    for ax, (prefix, title) in zip(axes, DATASETS.items(), strict=True):
        noisy = [100 * float(str(metrics[m][f"{prefix} - Noisy"]["accuracy"])) for m in MODELS]
        clean = [
            100 * float(str(metrics[m][f"{prefix} - Without noise"]["accuracy"])) for m in MODELS
        ]

        bars_noisy = ax.bar(x - width / 2, noisy, width, color=NOISY_COLOR)
        bars_clean = ax.bar(
            x + width / 2, clean, width, color=CLEAN_COLOR, edgecolor=NOISY_COLOR, linewidth=1.2
        )
        for bars in (bars_noisy, bars_clean):
            ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8, color="#1f2937")

        ax.set_title(f"{title} dataset", fontsize=12, pad=10)
        ax.set_xticks(x, list(MODELS.values()), fontsize=10)
        ax.set_xlabel("Model", fontsize=9)
        ax.set_ylim(Y_MIN, 101)
        ax.tick_params(axis="y", labelsize=8)
        ax.yaxis.grid(alpha=0.3)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_linewidth(2.5)

    axes[0].set_ylabel(f"Test accuracy (%)  - axis starts at {Y_MIN:.0f} %", fontsize=8)
    fig.legend(
        handles=[
            Patch(facecolor=NOISY_COLOR, label="With background noise"),
            Patch(facecolor=CLEAN_COLOR, edgecolor=NOISY_COLOR, label="Without background noise"),
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Test accuracy: noisy vs. without-noise recordings", fontsize=14)
    fig.savefig(OUT, dpi=200)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
