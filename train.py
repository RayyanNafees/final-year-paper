"""Binary fault detection for BLDC motors from psychoacoustic features.

Trains four classifiers (Logistic Regression, SVM, ANN, 1D-CNN) to tell a
healthy motor from a faulty one, using only the psychoacoustic features
(Loudness, Sharpness, Fluctuation, Roughness). Loss curves, confusion
matrices and test metrics are written to ``outputs/``.

Run with:  uv run train.py
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    hinge_loss,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "OpenSource"
OUT_DIR = ROOT / "outputs"
FEATURES = ["Loudness", "Sharpness", "Fluctuation", "Roughness"]
CLASS_NAMES = ["Healthy", "Faulty"]

SKLEARN_EPOCHS = 100
TORCH_EPOCHS = 150
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass
class History:
    """Per-epoch training and validation loss."""

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)


@dataclass
class Split:
    """Scaled train / validation / test arrays."""

    x_train: FloatArray
    y_train: IntArray
    x_val: FloatArray
    y_val: IntArray
    x_test: FloatArray
    y_test: IntArray


@dataclass
class ModelResult:
    name: str
    loss_name: str
    history: History
    y_pred: IntArray


# --------------------------------------------------------------------------- data


def load_dataset(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load every psychoacoustic CSV and attach `label` (0=healthy, 1=faulty) and `noise`."""
    frames: list[pd.DataFrame] = []
    # Works for the nested repo layout and for a flat folder of the four CSVs.
    for csv in sorted(data_dir.rglob("*.csv")):
        df = pd.read_csv(csv)
        df["label"] = 1 if "broken" in csv.stem.lower() else 0
        df["noise"] = "noisy" if "noisy" in csv.stem.lower() else "pure"
        df["source"] = csv.name
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No psychoacoustic CSV files found under {data_dir}")
    return pd.concat(frames, ignore_index=True)


def make_split(df: pd.DataFrame) -> Split:
    """Stratified 70/15/15 split; the scaler is fit on the training set only."""
    x = df[FEATURES].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int64)
    # Stratify on label *and* noise condition so every split has all four recordings.
    strata = df["label"].astype(str) + "_" + df["noise"]

    x_train, x_tmp, y_train, y_tmp, _, s_tmp = train_test_split(
        x, y, strata, test_size=0.30, stratify=strata, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=s_tmp, random_state=SEED
    )

    scaler = StandardScaler().fit(x_train)
    return Split(
        x_train=scaler.transform(x_train).astype(np.float32),
        y_train=y_train,
        x_val=scaler.transform(x_val).astype(np.float32),
        y_val=y_val,
        x_test=scaler.transform(x_test).astype(np.float32),
        y_test=y_test,
    )


# ------------------------------------------------------------------ sklearn models

LossFn = Callable[[SGDClassifier, FloatArray, IntArray], float]


def _log_loss(clf: SGDClassifier, x: FloatArray, y: IntArray) -> float:
    return float(log_loss(y, clf.predict_proba(x), labels=[0, 1]))


def _hinge_loss(clf: SGDClassifier, x: FloatArray, y: IntArray) -> float:
    return float(hinge_loss(y, clf.decision_function(x), labels=[0, 1]))


def train_sgd(
    clf: SGDClassifier,
    split: Split,
    loss_fn: LossFn,
    *,
    x_train: FloatArray,
    x_val: FloatArray,
    epochs: int = SKLEARN_EPOCHS,
) -> History:
    """Train an SGD linear model one epoch at a time, recording train/val loss."""
    rng = np.random.default_rng(SEED)
    history = History()
    classes = np.array([0, 1])
    for _ in range(epochs):
        order = rng.permutation(len(x_train))
        clf.partial_fit(x_train[order], split.y_train[order], classes=classes)
        history.train.append(loss_fn(clf, x_train, split.y_train))
        history.val.append(loss_fn(clf, x_val, split.y_val))
    return history


def run_logistic_regression(split: Split) -> ModelResult:
    clf = SGDClassifier(
        loss="log_loss", alpha=1e-4, learning_rate="constant", eta0=0.01, random_state=SEED
    )
    history = train_sgd(clf, split, _log_loss, x_train=split.x_train, x_val=split.x_val)
    return ModelResult("Logistic Regression", "log loss", history, clf.predict(split.x_test))


def run_svm(split: Split) -> ModelResult:
    """RBF-kernel SVM: Nystroem kernel approximation + hinge-loss SGD (linear SVM)."""
    kernel = Nystroem(kernel="rbf", gamma=0.5, n_components=300, random_state=SEED)
    x_train = kernel.fit_transform(split.x_train).astype(np.float32)
    x_val = kernel.transform(split.x_val).astype(np.float32)
    x_test = kernel.transform(split.x_test).astype(np.float32)

    clf = SGDClassifier(
        loss="hinge", alpha=1e-4, learning_rate="constant", eta0=0.005, random_state=SEED
    )
    history = train_sgd(clf, split, _hinge_loss, x_train=x_train, x_val=x_val)
    return ModelResult("SVM (RBF kernel)", "hinge loss", history, clf.predict(x_test))


# -------------------------------------------------------------------- torch models


class ANN(nn.Module):
    """Fully connected feed-forward network (MLP)."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNN1D(nn.Module):
    """1D-CNN treating the feature vector as a 1-channel signal of length `n_features`."""

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, features) -> (batch, 1 channel, features)
        return self.classifier(self.features(x)).squeeze(-1)


@torch.no_grad()
def _eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> float:
    model.eval()
    return float(criterion(model(x), y).item())


@torch.no_grad()
def _predict(model: nn.Module, x: torch.Tensor) -> IntArray:
    model.eval()
    return (torch.sigmoid(model(x)) > 0.5).long().cpu().numpy()


def train_torch(
    model: nn.Module, split: Split, device: torch.device, epochs: int = TORCH_EPOCHS
) -> tuple[History, IntArray]:
    """Train with Adam + BCE; restore the weights with the lowest validation loss."""
    model.to(device)
    x_train = torch.from_numpy(split.x_train).to(device)
    y_train = torch.from_numpy(split.y_train).float().to(device)
    x_val = torch.from_numpy(split.x_val).to(device)
    y_val = torch.from_numpy(split.y_val).float().to(device)
    x_test = torch.from_numpy(split.x_test).to(device)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    history = History()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        history.train.append(_eval_loss(model, x_train, y_train, criterion))
        history.val.append(_eval_loss(model, x_val, y_val, criterion))
        if history.val[-1] < best_val:
            best_val = history.val[-1]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    return history, _predict(model, x_test)


def run_ann(split: Split, device: torch.device) -> ModelResult:
    torch.manual_seed(SEED)
    history, y_pred = train_torch(ANN(len(FEATURES)), split, device)
    return ModelResult("ANN (MLP)", "BCE loss", history, y_pred)


def run_cnn(split: Split, device: torch.device) -> ModelResult:
    torch.manual_seed(SEED)
    history, y_pred = train_torch(CNN1D(len(FEATURES)), split, device)
    return ModelResult("1D-CNN", "BCE loss", history, y_pred)


# ------------------------------------------------------------------ reporting


def compute_metrics(y_true: IntArray, y_pred: IntArray) -> dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def plot_losses(results: list[ModelResult], path: Path) -> None:
    fig = Figure(figsize=(12, 8), layout="constrained")
    axes = fig.subplots(2, 2).flatten()
    for ax, res in zip(axes, results, strict=True):
        epochs = np.arange(1, len(res.history.train) + 1)
        ax.plot(epochs, res.history.train, label="train", color="#2563eb", linewidth=2)
        ax.plot(epochs, res.history.val, label="validation", color="#ea580c", linewidth=2)
        ax.set_title(res.name)
        ax.set_xlabel("epoch")
        ax.set_ylabel(res.loss_name)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("Training loss per epoch", fontsize=14)
    fig.savefig(path, dpi=150)


def plot_confusion_matrices(results: list[ModelResult], y_test: IntArray, path: Path) -> None:
    fig = Figure(figsize=(11, 9), layout="constrained")
    axes = fig.subplots(2, 2).flatten()
    for ax, res in zip(axes, results, strict=True):
        cm = confusion_matrix(y_test, res.y_pred, labels=[0, 1])
        acc = accuracy_score(y_test, res.y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            ax=ax,
            annot_kws={"size": 14},
        )
        ax.set_title(f"{res.name}  (accuracy {acc:.2%})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices on the held-out test set", fontsize=14)
    fig.savefig(path, dpi=150)


def main() -> None:
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(exist_ok=True)

    df = load_dataset()
    print(f"Loaded {len(df)} samples from {df['source'].nunique()} files")
    print(df.groupby(["source", "label"]).size().to_string(), "\n")

    split = make_split(df)
    print(
        f"train={len(split.y_train)}  val={len(split.y_val)}  test={len(split.y_test)}  "
        f"device={device}\n"
    )

    results = [
        run_logistic_regression(split),
        run_svm(split),
        run_ann(split, device),
        run_cnn(split, device),
    ]

    metrics: dict[str, dict[str, object]] = {}
    for res in results:
        m = compute_metrics(split.y_test, res.y_pred)
        metrics[res.name] = m
        cm = m["confusion_matrix"]
        print(
            f"{res.name:<20} acc={m['accuracy']:.4f}  prec={m['precision']:.4f}  "
            f"rec={m['recall']:.4f}  f1={m['f1']:.4f}"
        )
        print(f"{'':<20} confusion matrix [[TN, FP], [FN, TP]] = {cm}")

    plot_losses(results, OUT_DIR / "loss_curves.png")
    plot_confusion_matrices(results, split.y_test, OUT_DIR / "confusion_matrices.png")
    (OUT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved plots and metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
