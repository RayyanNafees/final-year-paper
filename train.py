"""BLDC motor condition classification from psychoacoustic features.

Four dataset cases are trained and evaluated independently:

* OpenSource, with background noise      -> Healthy / Faulty / Heavy Load (3 classes)
* OpenSource, without background noise   -> Healthy / Faulty / Heavy Load (3 classes)
* TwoWheeler, with background noise      -> Healthy / Faulty              (2 classes)
* TwoWheeler, without background noise   -> Healthy / Faulty              (2 classes)

Every case is trained with four models: Logistic/Softmax Regression, SVM (RBF kernel),
ANN and 1D-CNN. The inputs are the psychoacoustic features (Loudness, Sharpness,
Fluctuation, Roughness), and each row's class label comes from its CSV file name.
For every model the script writes one loss-curve image and one confusion-matrix image
to ``outputs/``, each with one panel per dataset case.

Run with:  uv run train.py [--data-dir newdata --out-dir outputs_newdata]
Only the features of FEATURES that are present in the CSVs are used, so a dataset without
e.g. Fluctuation trains on the remaining three features.
"""

from __future__ import annotations

import argparse
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
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F  # noqa: N812 - conventional PyTorch alias
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FEATURES = ["Loudness", "Sharpness", "Fluctuation", "Roughness"]

# Dataset case name -> folder (relative to DATA_DIR) that holds its CSV files.
CASES: dict[str, str] = {
    "OpenSource - Noisy": "OpenSource/With Background noise",
    "OpenSource - Without noise": "OpenSource/No Background noise",
    "TwoWheeler - Noisy": "TwoWheeler/with background noise",
    "TwoWheeler - Without noise": "TwoWheeler/Without background noise",
}

# File-name token -> class name. Tokens such as "pure" or "noisy" are ignored.
LABEL_TOKENS = {
    "good": "Healthy",
    "healthy": "Healthy",
    "broken": "Faulty",
    "faulty": "Faulty",
    "heavyload": "Heavy Load",
}
CLASS_ORDER = ["Healthy", "Faulty", "Heavy Load"]

# Strictly positive, heavily right-skewed features (Fluctuation skew 5-20, spanning several
# orders of magnitude) are log10-transformed before standardisation.
LOG_FEATURES = ["Fluctuation", "Roughness"]

SVM_EPOCHS = 100
SVM_GRID = [(gamma, alpha) for gamma in (0.1, 0.5, 1.0) for alpha in (1e-4, 1e-3)]
MAX_EPOCHS = 300
PATIENCE = 30  # early stopping: epochs without validation-loss improvement
BATCH_SIZE = 64

FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]


@dataclass
class History:
    """Per-epoch training and validation loss."""

    train: list[float] = field(default_factory=list)
    val: list[float] = field(default_factory=list)
    best_epoch: int | None = None  # 1-based epoch whose weights were kept
    note: str = ""  # e.g. selected hyper-parameters


@dataclass
class Split:
    """Scaled train / validation / test arrays of one dataset case."""

    x_train: FloatArray
    y_train: IntArray
    x_val: FloatArray
    y_val: IntArray
    x_test: FloatArray
    y_test: IntArray
    class_names: list[str]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]


@dataclass
class CaseResult:
    """Outcome of training one model on one dataset case."""

    history: History
    y_test: IntArray
    y_pred: IntArray
    class_names: list[str]


# --------------------------------------------------------------------------- data


def class_from_filename(csv: Path) -> str:
    """Map e.g. ``Noisy_HeavyLoad.csv`` -> ``Heavy Load`` using the tokens of the file name."""
    for token in csv.stem.lower().split("_"):
        if token in LABEL_TOKENS:
            return LABEL_TOKENS[token]
    raise ValueError(f"Cannot infer the class of {csv.name}; known tokens: {sorted(LABEL_TOKENS)}")


def load_case(folder: Path) -> tuple[pd.DataFrame, list[str]]:
    """Stack every CSV in ``folder`` into one frame with an integer ``label`` column."""
    csvs = sorted(p for p in folder.rglob("*.csv") if not p.name.startswith("._"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found under {folder}")

    frames = []
    for csv in csvs:
        df = pd.read_csv(csv)
        df["class_name"] = class_from_filename(csv)
        df["source"] = csv.name
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)

    present = set(data["class_name"])
    class_names = [c for c in CLASS_ORDER if c in present]
    data["label"] = data["class_name"].map({c: i for i, c in enumerate(class_names)})
    return data, class_names


def make_split(
    data: pd.DataFrame, class_names: list[str], log_features: list[str] = LOG_FEATURES
) -> Split:
    """Stratified 70/15/15 split -> log10 of skewed features -> StandardScaler fit on train only."""
    features = data[[f for f in FEATURES if f in data.columns]].copy()
    for col in (c for c in log_features if c in features.columns):
        features[col] = np.log10(features[col])
    x = features.to_numpy(dtype=np.float32)
    y = data["label"].to_numpy(dtype=np.int64)

    x_train, x_tmp, y_train, y_tmp = train_test_split(
        x, y, test_size=0.30, stratify=y, random_state=SEED
    )
    x_val, x_test, y_val, y_test = train_test_split(
        x_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED
    )

    scaler = StandardScaler().fit(x_train)
    return Split(
        x_train=scaler.transform(x_train).astype(np.float32),
        y_train=y_train,
        x_val=scaler.transform(x_val).astype(np.float32),
        y_val=y_val,
        x_test=scaler.transform(x_test).astype(np.float32),
        y_test=y_test,
        class_names=class_names,
    )


# ---------------------------------------------------------------------- SVM (sklearn)


def _fit_svm(
    split: Split, gamma: float, alpha: float
) -> tuple[History, Nystroem, SGDClassifier, float]:
    """Train one SVM configuration epoch by epoch; return it with its validation accuracy."""
    kernel = Nystroem(kernel="rbf", gamma=gamma, n_components=300, random_state=SEED)
    x_train = kernel.fit_transform(split.x_train).astype(np.float32)
    x_val = kernel.transform(split.x_val).astype(np.float32)

    clf = SGDClassifier(
        loss="hinge", alpha=alpha, learning_rate="constant", eta0=0.005, random_state=SEED
    )
    labels = np.arange(split.n_classes)
    rng = np.random.default_rng(SEED)
    history = History(note=f"gamma={gamma}, alpha={alpha:g}")
    for _ in range(SVM_EPOCHS):
        order = rng.permutation(len(x_train))
        clf.partial_fit(x_train[order], split.y_train[order], classes=labels)
        history.train.append(
            float(hinge_loss(split.y_train, clf.decision_function(x_train), labels=labels))
        )
        history.val.append(
            float(hinge_loss(split.y_val, clf.decision_function(x_val), labels=labels))
        )
    val_acc = float(accuracy_score(split.y_val, clf.predict(x_val)))
    return history, kernel, clf, val_acc


def train_svm(split: Split, _device: torch.device) -> tuple[History, IntArray]:
    """RBF-kernel SVM: Nystroem kernel approximation + hinge-loss SGD, one epoch at a time.

    The RBF width `gamma` and the L2 strength `alpha` are picked from SVM_GRID by
    validation accuracy (ties -> lower validation hinge loss); the test set is never used.
    Multiclass cases are handled one-vs-rest; the loss reported for them is the
    multiclass (Crammer-Singer) hinge loss.
    """
    fits = [_fit_svm(split, gamma, alpha) for gamma, alpha in SVM_GRID]
    history, kernel, clf, _ = max(fits, key=lambda f: (f[3], -f[0].val[-1]))
    x_test = kernel.transform(split.x_test).astype(np.float32)
    return history, clf.predict(x_test).astype(np.int64)


# ------------------------------------------------------------------- torch models


def output_dim(n_classes: int) -> int:
    """One logit (sigmoid) for binary problems, one logit per class (softmax) otherwise."""
    return 1 if n_classes == 2 else n_classes


class LinearClassifier(nn.Module):
    """Logistic regression (binary) / softmax regression (multiclass): a single linear layer."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(n_features, output_dim(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class ANN(nn.Module):
    """Fully connected feed-forward network (MLP)."""

    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim(n_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNN1D(nn.Module):
    """1D-CNN treating the feature vector as a 1-channel signal of length `n_features`."""

    def __init__(self, n_features: int, n_classes: int) -> None:
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
            nn.Linear(32, output_dim(n_classes)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)  # (batch, features) -> (batch, 1 channel, features)
        return self.classifier(self.features(x))


def classification_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy for a single logit, categorical cross-entropy otherwise."""
    if logits.shape[-1] == 1:
        return F.binary_cross_entropy_with_logits(logits.squeeze(-1), y.float())
    return F.cross_entropy(logits, y)


def predict_labels(logits: torch.Tensor) -> torch.Tensor:
    if logits.shape[-1] == 1:
        return (logits.squeeze(-1) > 0).long()  # sigmoid(z) > 0.5  <=>  z > 0
    return logits.argmax(dim=-1)


@torch.no_grad()
def _eval_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    return float(classification_loss(model(x), y).item())


def train_torch(
    model: nn.Module, split: Split, device: torch.device, lr: float
) -> tuple[History, IntArray]:
    """AdamW + ReduceLROnPlateau + early stopping; keep the weights with the lowest val loss."""
    model.to(device)
    x_train = torch.from_numpy(split.x_train).to(device)
    y_train = torch.from_numpy(split.y_train).to(device)
    x_val = torch.from_numpy(split.x_val).to(device)
    y_val = torch.from_numpy(split.y_val).to(device)
    x_test = torch.from_numpy(split.x_test).to(device)

    loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    history = History()
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad(set_to_none=True)
            classification_loss(model(xb), yb).backward()
            optimizer.step()

        history.train.append(_eval_loss(model, x_train, y_train))
        history.val.append(_eval_loss(model, x_val, y_val))
        scheduler.step(history.val[-1])
        if history.val[-1] < best_val - 1e-5:
            best_val = history.val[-1]
            best_state = copy.deepcopy(model.state_dict())
            history.best_epoch = epoch
        elif history.best_epoch is not None and epoch - history.best_epoch >= PATIENCE:
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        y_pred = predict_labels(model(x_test)).cpu().numpy()
    return history, y_pred


def train_linear(split: Split, device: torch.device) -> tuple[History, IntArray]:
    torch.manual_seed(SEED)
    return train_torch(LinearClassifier(split.n_features, split.n_classes), split, device, lr=1e-2)


def train_ann(split: Split, device: torch.device) -> tuple[History, IntArray]:
    torch.manual_seed(SEED)
    return train_torch(ANN(split.n_features, split.n_classes), split, device, lr=1e-3)


def train_cnn(split: Split, device: torch.device) -> tuple[History, IntArray]:
    torch.manual_seed(SEED)
    return train_torch(CNN1D(split.n_features, split.n_classes), split, device, lr=1e-3)


# ----------------------------------------------------------------- model registry

Trainer = Callable[[Split, torch.device], tuple[History, IntArray]]


@dataclass(frozen=True)
class ModelSpec:
    key: str  # used in output file names
    title: str
    trainer: Trainer
    binary_loss: str
    multiclass_loss: str

    def loss_name(self, n_classes: int) -> str:
        return self.binary_loss if n_classes == 2 else self.multiclass_loss

    def case_title(self, n_classes: int) -> str:
        if self.key == "logistic_regression":
            return "Logistic regression" if n_classes == 2 else "Softmax regression"
        return self.title


MODELS = [
    ModelSpec(
        "logistic_regression",
        "Logistic / Softmax Regression",
        train_linear,
        "binary cross-entropy",
        "cross-entropy (softmax)",
    ),
    ModelSpec("svm", "SVM (RBF kernel)", train_svm, "hinge loss", "multiclass hinge loss"),
    ModelSpec("ann", "ANN (MLP)", train_ann, "binary cross-entropy", "cross-entropy"),
    ModelSpec("cnn1d", "1D-CNN", train_cnn, "binary cross-entropy", "cross-entropy"),
]


# ---------------------------------------------------------------------- reporting


def compute_metrics(res: CaseResult) -> dict[str, object]:
    """Accuracy plus macro-averaged precision / recall / F1 (equal weight per class)."""
    labels = list(range(len(res.class_names)))
    return {
        "classes": res.class_names,
        "accuracy": float(accuracy_score(res.y_test, res.y_pred)),
        "precision_macro": float(
            precision_score(res.y_test, res.y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(res.y_test, res.y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(res.y_test, res.y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(res.y_test, res.y_pred, labels=labels).tolist(),
        "epochs_trained": len(res.history.train),
        "best_epoch": res.history.best_epoch,
        "selected_hyperparameters": res.history.note or None,
    }


def plot_loss_curves(spec: ModelSpec, results: dict[str, CaseResult], path: Path) -> None:
    fig = Figure(figsize=(13, 9), layout="constrained")
    axes = fig.subplots(2, 2).flatten()
    for ax, (case, res) in zip(axes, results.items(), strict=True):
        n = len(res.class_names)
        epochs = np.arange(1, len(res.history.train) + 1)
        ax.plot(epochs, res.history.train, label="train", color="#2563eb", linewidth=2)
        ax.plot(epochs, res.history.val, label="validation", color="#ea580c", linewidth=2)
        if res.history.best_epoch is not None:
            ax.axvline(
                res.history.best_epoch,
                color="#6b7280",
                linestyle="--",
                linewidth=1,
                label=f"best epoch ({res.history.best_epoch})",
            )
        detail = f", {res.history.note}" if res.history.note else ""
        ax.set_title(f"{case}\n{spec.case_title(n)}, {n} classes{detail}")
        ax.set_xlabel("epoch")
        ax.set_ylabel(spec.loss_name(n))
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(f"{spec.title}: training & validation loss per epoch", fontsize=15)
    fig.savefig(path, dpi=150)


def plot_confusion_matrices(spec: ModelSpec, results: dict[str, CaseResult], path: Path) -> None:
    fig = Figure(figsize=(13, 11), layout="constrained")
    axes = fig.subplots(2, 2).flatten()
    for ax, (case, res) in zip(axes, results.items(), strict=True):
        labels = list(range(len(res.class_names)))
        cm = confusion_matrix(res.y_test, res.y_pred, labels=labels)
        acc = accuracy_score(res.y_test, res.y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=res.class_names,
            yticklabels=res.class_names,
            ax=ax,
            annot_kws={"size": 13},
        )
        ax.set_title(f"{case}\naccuracy {acc:.2%}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(f"{spec.title}: confusion matrices on each held-out test set", fontsize=15)
    fig.savefig(path, dpi=150)


# --------------------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    data_dir: Path = args.data_dir.resolve()
    out_dir: Path = args.out_dir.resolve()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir.mkdir(exist_ok=True)
    print(f"device={device}, data={data_dir}\n")

    splits: dict[str, Split] = {}
    for case, rel in CASES.items():
        data, class_names = load_case(data_dir / rel)
        splits[case] = make_split(data, class_names)
        counts = data.groupby(["source", "class_name"]).size().to_dict()
        used = [f for f in FEATURES if f in data.columns]
        print(f"{case}: {len(data)} rows, classes={class_names}, features={used}")
        for (src, cls), n in counts.items():
            print(f"    {src:<22} -> {cls:<10} {n} rows")
        s = splits[case]
        print(f"    split train={len(s.y_train)} val={len(s.y_val)} test={len(s.y_test)}")
    print()

    metrics: dict[str, dict[str, dict[str, object]]] = {}
    summary: dict[str, dict[str, float]] = {}
    for spec in MODELS:
        results: dict[str, CaseResult] = {}
        for case, split in splits.items():
            history, y_pred = spec.trainer(split, device)
            results[case] = CaseResult(history, split.y_test, y_pred, split.class_names)
            print(f"trained {spec.title:<30} on {case}")

        metrics[spec.title] = {case: compute_metrics(res) for case, res in results.items()}
        summary[spec.title] = {
            case: float(accuracy_score(res.y_test, res.y_pred)) for case, res in results.items()
        }
        plot_loss_curves(spec, results, out_dir / f"{spec.key}_loss_curves.png")
        plot_confusion_matrices(spec, results, out_dir / f"{spec.key}_confusion_matrices.png")

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    print("\nTest accuracy (rows = model, columns = dataset case)")
    print(pd.DataFrame(summary).T.to_string(float_format=lambda v: f"{v:.4f}"))
    for spec in MODELS:
        print(f"\n{spec.title}")
        for case, m in metrics[spec.title].items():
            print(f"  {case:<28} classes={m['classes']}  confusion={m['confusion_matrix']}")
    print(f"\nSaved {2 * len(MODELS)} images and metrics.json to {out_dir}")


if __name__ == "__main__":
    main()
