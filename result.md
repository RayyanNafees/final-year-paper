# Results on `newdata` (3 features: Loudness, Sharpness, Roughness)

The `newdata` CSVs contain the same 1000 windows per recording as `data/`, but without the
Fluctuation feature. Each model was trained and tested separately on every case, using the same
pipeline as before:

- **Split:** 70 / 15 / 15 stratified train / validation / test split, seed 42.
- **Preprocessing:** log10 of Roughness, then z-score scaling fitted on the training part only.
- **Test set size:** 450 windows for OpenSource (150 per class) and 300 for TwoWheeler (150 per class).

Reproduce with `uv run train.py --data-dir newdata --out-dir outputs_newdata`.

## 1. Metrics (test set, macro-averaged)

| Model | Case | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic / Softmax Reg. | OpenSource – without noise | 100.00% | 100.00% | 100.00% | 100.00% |
| | OpenSource – with noise | 95.33% | 95.37% | 95.33% | 95.33% |
| | TwoWheeler – without noise | 99.33% | 99.33% | 99.33% | 99.33% |
| | TwoWheeler – with noise | 93.33% | 93.83% | 93.33% | 93.31% |
| SVM (RBF) | OpenSource – without noise | 100.00% | 100.00% | 100.00% | 100.00% |
| | OpenSource – with noise | 96.00% | 96.19% | 96.00% | 95.99% |
| | TwoWheeler – without noise | 99.00% | 99.02% | 99.00% | 99.00% |
| | TwoWheeler – with noise | 98.67% | 98.67% | 98.67% | 98.67% |
| ANN (MLP) | OpenSource – without noise | 100.00% | 100.00% | 100.00% | 100.00% |
| | OpenSource – with noise | 96.00% | 96.08% | 96.00% | 96.00% |
| | TwoWheeler – without noise | 99.67% | 99.67% | 99.67% | 99.67% |
| | TwoWheeler – with noise | 99.67% | 99.67% | 99.67% | 99.67% |
| 1D-CNN | OpenSource – without noise | 100.00% | 100.00% | 100.00% | 100.00% |
| | OpenSource – with noise | **96.67%** | 96.78% | 96.67% | 96.66% |
| | TwoWheeler – without noise | 99.67% | 99.67% | 99.67% | 99.67% |
| | TwoWheeler – with noise | **99.67%** | 99.67% | 99.67% | 99.67% |

### Comparison with the previous 4-feature data (test accuracy)

| Model | OS clean (4f → 3f) | OS noisy (4f → 3f) | TW clean (4f → 3f) | TW noisy (4f → 3f) |
|---|---|---|---|---|
| Logistic / Softmax | 100 → 100 | 95.56 → 95.33 | 99.67 → 99.33 | 94.00 → 93.33 |
| SVM (RBF) | 100 → 100 | 96.00 → 96.00 | 100 → 99.00 | 99.00 → 98.67 |
| ANN (MLP) | 100 → 100 | 95.78 → 96.00 | 100 → 99.67 | 100 → 99.67 |
| 1D-CNN | 100 → 100 | 97.11 → 96.67 | 100 → 99.67 | 100 → 99.67 |

Dropping Fluctuation changes accuracy by at most 1 point, which is 0 to 3 test windows per case.

## 2. Confusion matrices (rows = actual, columns = predicted)

OpenSource class order: Healthy, Faulty, Heavy Load. TwoWheeler class order: Healthy, Faulty.

| Model | OS – without noise | OS – with noise | TW – without noise | TW – with noise |
|---|---|---|---|---|
| Logistic / Softmax | [[150,0,0],[0,150,0],[0,0,150]] | [[137,0,13],[0,150,0],[8,0,142]] | [[149,1],[1,149]] | [[148,2],[18,132]] |
| SVM (RBF) | [[150,0,0],[0,150,0],[0,0,150]] | [[135,0,15],[0,150,0],[3,0,147]] | [[150,0],[3,147]] | [[148,2],[2,148]] |
| ANN (MLP) | [[150,0,0],[0,150,0],[0,0,150]] | [[137,0,13],[0,150,0],[5,0,145]] | [[150,0],[1,149]] | [[150,0],[1,149]] |
| 1D-CNN | [[150,0,0],[0,150,0],[0,0,150]] | [[138,0,12],[0,150,0],[3,0,147]] | [[150,0],[1,149]] | [[150,0],[1,149]] |

Rendered images, one per model, dataset and noise condition, are in `outputs_newdata/figures/confusion/`
(`<model>__<dataset>__<noisy|clean>.png`). The accuracy bar charts are in `outputs_newdata/figures/`.
Regenerate them with `uv run plot_results.py --out-dir outputs_newdata`.

## 3. Observations

- **Without noise:** every model scores 100% on OpenSource. On TwoWheeler, accuracy is 99.0–99.67%, with 1 to 3
  faulty windows predicted as healthy. With Fluctuation included, the non-linear models had reached 100% here.
- **With noise, OpenSource:** accuracy is 95.3–96.7%. Every error is a Healthy ↔ Heavy Load mix-up, and Faulty is
  never confused with any other class.
- **With noise, TwoWheeler:** the ANN and 1D-CNN hold at 99.67%, and the SVM reaches 98.67%. Logistic regression drops
  to 93.33% because a linear boundary misses 18 faulty windows.
- **Best model:** the 1D-CNN gives the best or joint-best accuracy in every case.

## 4. Models used

All models take the 3 scaled features as input. Binary cases use one sigmoid output with BCE loss, and the
3-class cases use a softmax output with cross-entropy loss.

| Model | Architecture | Trainable params (2 / 3 classes) | Training |
|---|---|---|---|
| Logistic / Softmax Regression | Single linear layer, 3 → 1 (sigmoid) or 3 → 3 (softmax) | 4 / 12 | AdamW, lr 0.01 |
| SVM (RBF kernel) | Nystroem RBF approximation with 300 components, then a linear hinge-loss SVM trained by SGD (one-vs-rest for 3 classes) | – | 100 epochs, eta 0.005. gamma ∈ {0.1, 0.5, 1.0} and alpha ∈ {1e-4, 1e-3} chosen on validation accuracy. Selected values: gamma 0.1, alpha 1e-4 (OpenSource) and gamma 1.0, alpha 1e-4 (TwoWheeler) |
| ANN (MLP) | 3 → 32 → 16 → output, ReLU, dropout 0.1 | 673 / 707 | AdamW, lr 0.001 |
| 1D-CNN | Features treated as a 1-channel sequence of length 3. Conv1d(16, k=3) + BatchNorm + ReLU, then Conv1d(32, k=3) + BatchNorm + ReLU, then Dense 32 + dropout 0.2, then output | 4865 / 4931 | AdamW, lr 0.001 |

The settings shared by the PyTorch models are:

- weight decay 1e-4
- batch size 64
- ReduceLROnPlateau, which halves the learning rate after 10 epochs without improvement
- early stopping with patience 30, for at most 300 epochs, keeping the weights with the lowest validation loss
- seed 42
