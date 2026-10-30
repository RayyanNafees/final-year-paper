"""Render every result of a ``train.py`` run as separate, publication-ready PNG images.

Reads ``<out-dir>/metrics.json`` and writes to ``<out-dir>/figures/``:

* ``confusion/<model>__<dataset>__<noise>.png``  one confusion matrix per model, dataset and
  noise condition (4 models x 2 datasets x 2 noise conditions = 16 images)
* ``accuracy_<dataset>.png``  all models on one dataset; each model has a pair of bars
  (with / without background noise)
* ``accuracy_<model>.png``    one model on both datasets, same pair of bars per dataset

Run with:  uv run plot_results.py --out-dir outputs_newdata
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib.figure import Figure
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent

# metrics.json model key -> (file key, short label)
MODELS = {
    "Logistic / Softmax Regression": ("logistic_regression", "Logistic / Softmax\nRegression"),
    "SVM (RBF kernel)": ("svm", "SVM (RBF)"),
    "ANN (MLP)": ("ann", "ANN (MLP)"),
    "1D-CNN": ("cnn1d", "1D-CNN"),
}
DATASETS = {"OpenSource": "OpenSource (standalone motor)", "TwoWheeler": "Two-Wheeler"}
NOISE = {"Noisy": ("noisy", "With background noise"), "Without noise": ("clean", "Without noise")}

NOISY_COLOR = "#0b3d7a"  # dark navy bar
CLEAN_COLOR = "#e3edf9"  # light bar with navy outline
TEXT_COLOR = "#1f2937"
Y_MIN = 90.0  # accuracies lie in 93-100 %; a 0-based axis would hide the differences

Metrics = dict[str, dict[str, dict[str, object]]]


def accuracy(metrics: Metrics, model: str, dataset: str, noise: str) -> float:
    return 100 * float(str(metrics[model][f"{dataset} - {noise}"]["accuracy"]))


def style_bar_axes(ax) -> None:  # noqa: ANN001 - matplotlib Axes
    ax.set_ylim(Y_MIN, 101.5)
    ax.set_ylabel(f"Test accuracy (%)  - axis starts at {Y_MIN:.0f} %", fontsize=9)
    ax.tick_params(axis="y", labelsize=9)
    ax.yaxis.grid(alpha=0.3)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def paired_bars(
    title: str, labels: list[str], noisy: list[float], clean: list[float], xlabel: str, path: Path
) -> None:
    """One cluster per label, two bars per cluster: with noise (dark) and without noise (light)."""
    fig = Figure(figsize=(max(5.5, 2.0 * len(labels) + 1.5), 5.2), layout="constrained")
    ax = fig.subplots()
    x = np.arange(len(labels))
    width = 0.36
    bars_noisy = ax.bar(x - width / 2, noisy, width - 0.02, color=NOISY_COLOR)
    bars_clean = ax.bar(
        x + width / 2, clean, width - 0.02, color=CLEAN_COLOR, edgecolor=NOISY_COLOR, linewidth=1.2
    )
    for bars in (bars_noisy, bars_clean):
        ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=9, color=TEXT_COLOR)
    ax.set_xticks(x, labels, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    style_bar_axes(ax)
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
    fig.suptitle(title, fontsize=13)
    fig.savefig(path, dpi=200)


def confusion_image(title: str, cm: list[list[int]], classes: list[str], path: Path) -> None:
    """Counts with the row percentage underneath (share of each actual class)."""
    counts = np.array(cm)
    row_pct = 100 * counts / counts.sum(axis=1, keepdims=True)
    annot = np.array(
        [[f"{c}\n({p:.1f}%)" for c, p in zip(cr, pr, strict=True)]
         for cr, pr in zip(counts, row_pct, strict=True)]
    )
    size = 4.2 + 0.9 * len(classes)
    fig = Figure(figsize=(size, size - 0.4), layout="constrained")
    ax = fig.subplots()
    sns.heatmap(
        counts,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        linewidths=2,
        linecolor="white",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        annot_kws={"size": 12},
    )
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("Actual class", fontsize=10)
    ax.tick_params(axis="y", rotation=0)
    ax.set_title(title, fontsize=11)
    fig.savefig(path, dpi=200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    out_dir: Path = args.out_dir.resolve()
    metrics: Metrics = json.loads((out_dir / "metrics.json").read_text())

    fig_dir = out_dir / "figures"
    cm_dir = fig_dir / "confusion"
    cm_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    for model, (model_key, model_label) in MODELS.items():
        for dataset, dataset_title in DATASETS.items():
            for noise, (noise_key, noise_title) in NOISE.items():
                m = metrics[model][f"{dataset} - {noise}"]
                acc = 100 * float(str(m["accuracy"]))
                title = (
                    f"{model_label.replace(chr(10), ' ')}\n{dataset_title}, {noise_title.lower()}"
                    f"\nAccuracy {acc:.2f}%"
                )
                path = cm_dir / f"{model_key}__{dataset.lower()}__{noise_key}.png"
                confusion_image(title, m["confusion_matrix"], list(m["classes"]), path)  # type: ignore[arg-type]
                saved.append(path)

    for dataset, dataset_title in DATASETS.items():
        path = fig_dir / f"accuracy_{dataset.lower()}.png"
        paired_bars(
            f"{dataset_title}: test accuracy with vs. without background noise",
            [label for _, label in MODELS.values()],
            [accuracy(metrics, m, dataset, "Noisy") for m in MODELS],
            [accuracy(metrics, m, dataset, "Without noise") for m in MODELS],
            "Model",
            path,
        )
        saved.append(path)

    for model, (model_key, model_label) in MODELS.items():
        path = fig_dir / f"accuracy_{model_key}.png"
        paired_bars(
            f"{model_label.replace(chr(10), ' ')}: test accuracy with vs. without noise",
            list(DATASETS.values()),
            [accuracy(metrics, model, d, "Noisy") for d in DATASETS],
            [accuracy(metrics, model, d, "Without noise") for d in DATASETS],
            "Dataset",
            path,
        )
        saved.append(path)

    print(f"Saved {len(saved)} images to {fig_dir}")


if __name__ == "__main__":
    main()
