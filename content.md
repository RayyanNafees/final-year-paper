# Effect of Background Noise on Model Accuracy

![Test accuracy: noisy vs. without-noise recordings](outputs/accuracy_comparison.png)

Each model was trained and tested separately on every dataset case (70 / 15 / 15 split). The
accuracies below come from each case's held-out test set: 450 samples for OpenSource
(Healthy / Faulty / Heavy Load) and 300 samples for Two-Wheeler (Healthy / Faulty).

## OpenSource dataset (3 classes: Healthy, Faulty, Heavy Load)

### SVM (RBF kernel)
- **Without noise: 100%.** The clean recordings separate perfectly, mainly on Sharpness and Loudness.
- **With noise: 96.00%,** a drop of 4.0 points. All 18 errors are Healthy 🔄 Heavy Load mix-ups (14 Healthy
  predicted as Heavy Load, 4 the other way round). Faulty is still detected perfectly.
- This is the second-best model on noisy data, but its errors lean towards flagging healthy motors as heavily
  loaded.

### ANN (MLP)
- **Without noise: 100%,** with zero misclassifications.
- **With noise: 95.78%.** Noise causes the largest drop of the three models here (📉 4.22 points). The 19 errors
  are spread evenly between Healthy ➡️ Heavy Load (10) and Heavy Load ➡️ Healthy (9).
- Background noise raises the loudness floor, so the Healthy and Heavy Load clusters overlap. A small MLP can't
  fully separate that overlap.

### 1D-CNN
- **Without noise: 100%.**
- **With noise: 97.11%,** the most robust model on noisy OpenSource data. Its drop is only 2.89 points, with 13
  errors (8 Healthy ➡️ Heavy Load, 5 Heavy Load ➡️ Healthy).
- A likely reason is that its convolution filters learn combinations of neighbouring features
  (Loudness + Sharpness, Sharpness + Fluctuation, and so on), which helps separate the overlapping noisy classes.

## Two-Wheeler dataset (2 classes: Healthy, Faulty)

### SVM (RBF kernel)
- **Without noise: 100%.**
- **With noise: 99.00%,** a small drop of 1.0 point (3 of 300 samples: 2 Faulty missed, 1 Healthy flagged).
- It is the only one of the three models affected by noise on this dataset. Some faulty samples fall close to
  the healthy cluster when noise is present.

### ANN (MLP)
- **Without noise: 100%. With noise: 100%.** Noise has **no effect** on accuracy ✅
- Healthy and faulty two-wheeler sounds differ strongly in Loudness and Sharpness, and the non-linear MLP still
  separates them perfectly after noise is added.

### 1D-CNN
- **Without noise: 100%. With noise: 100%.** Noise has **no effect** ✅
- Like the ANN, it is completely robust to background noise on this dataset, and it reaches its best weights in
  fewer epochs (24 noisy / 78 clean).

## Overall comparison
- **Noise hurts the 3-class OpenSource task much more** (📉 2.9 to 4.2 points) than the 2-class Two-Wheeler task
  (📉 0 to 1 point). Noise only blurs the *Healthy vs. Heavy Load* boundary; *Faulty* is never confused with
  anything in either dataset.
- 🏆 **Most robust model: 1D-CNN.** It has the highest noisy-data accuracy in both datasets (97.11% and 100%).
- **Without background noise, all three models reach 100%** on both datasets.
