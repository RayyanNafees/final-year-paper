# 4. Preliminary Work

Before the full experiments on the PMSM-driven four-wheeler, the acoustic part of the proposed method was tested on
a smaller and more accessible machine: a brushless DC (BLDC) motor. The aim was to check whether the four
psychoacoustic features of Section 3.2 alone are enough to tell a healthy motor from a faulty one, and how strongly
background noise affects this. The preliminary study followed the same chain as the main methodology: recording,
windowing, feature extraction, dataset preparation, model training, and evaluation with accuracy and confusion
matrices. Vibration was not used at this stage, and tonality was not yet included as a feature.

## 4.1 Data acquisition

The BLDC motor was recorded in two setups:

- **Setup A (standalone motor):** the motor was run on its own, outside any vehicle, in three operating conditions:
  healthy, faulty, and heavy load.
- **Setup B (two-wheeler):** the motor was recorded while fitted in a two-wheeler, in two operating conditions:
  healthy and faulty.

Each condition was recorded twice: once without background noise and once with traffic noise present. This gives
four recording cases (Table 2). The sound was recorded with mobile phone microphones, the low-cost acoustic option
listed in Table 1, at a sampling rate of 48 kHz. The microphone was kept at the same position for all recordings
within a setup, so that the recordings could be compared with each other.

Table 2. Recording cases of the preliminary study

| Case | Setup | Background | Classes | Windows per class | Total windows |
|---|---|---|---|---|---|
| A1 | Standalone motor | Without noise | Healthy, Faulty, Heavy load | 1000 | 3000 |
| A2 | Standalone motor | With traffic noise | Healthy, Faulty, Heavy load | 1000 | 3000 |
| B1 | Two-wheeler | Without noise | Healthy, Faulty | 1000 | 2000 |
| B2 | Two-wheeler | With traffic noise | Healthy, Faulty | 1000 | 2000 |

## 4.2 Feature extraction

The audio recordings were imported into MATLAB, and each recording was divided into 1000 windows. Four
psychoacoustic features were calculated for every window with the Audio Toolbox functions `acousticLoudness`,
`acousticSharpness`, `acousticFluctuation` and `acousticRoughness`:

- **Loudness** (sone): how strong the sound seems.
- **Sharpness** (acum): how much high-frequency content the sound has.
- **Fluctuation strength** (vacil): slow amplitude modulation of the sound, around 4 Hz.
- **Roughness** (asper): fast amplitude modulation of the sound, around 70 Hz.

Each window became one row of a CSV file, and the condition label was taken from the name of the recording. Because
smartphone microphones are not calibrated sound level meters, the feature values are treated as relative. They are
compared only within one setup, never between setups.

Table 3 shows the class means and standard deviations. In both setups the faulty motor was rougher than the healthy
one. In Setup A the faulty motor was also sharper, and without background noise the heavy-load condition was the
loudest and the least sharp. In Setup B the faulty motor was quieter and less sharp than the healthy one on average,
and its loudness varied much more (standard deviation 11.7 to 15.0, against 1.7 to 2.4 for the healthy motor).

Table 3. Mean (standard deviation) of the psychoacoustic features per class

| Case | Class | Loudness | Sharpness | Roughness |
|---|---|---|---|---|
| A1 (without noise) | Healthy | 37.8 (0.8) | 1.64 (0.03) | 0.062 (0.015) |
| | Faulty | 38.7 (0.6) | 2.11 (0.02) | 0.107 (0.034) |
| | Heavy load | 45.4 (3.2) | 1.36 (0.05) | 0.049 (0.009) |
| A2 (with noise) | Healthy | 45.3 (2.9) | 1.41 (0.06) | 0.072 (0.018) |
| | Faulty | 52.3 (2.7) | 1.66 (0.07) | 0.097 (0.021) |
| | Heavy load | 49.3 (3.1) | 1.27 (0.04) | 0.063 (0.013) |
| B1 (without noise) | Healthy | 47.7 (1.7) | 2.27 (0.08) | 0.074 (0.021) |
| | Faulty | 30.6 (15.0) | 1.60 (0.26) | 0.103 (0.039) |
| B2 (with noise) | Healthy | 53.8 (2.4) | 2.16 (0.06) | 0.084 (0.027) |
| | Faulty | 36.3 (11.7) | 1.62 (0.25) | 0.101 (0.031) |

## 4.3 Data preparation and normalisation

Each of the four cases was handled as a separate dataset. Every class contributed exactly 1000 windows, so all four
datasets were balanced and no resampling or class weighting was needed. Each dataset was split, stratified by class,
into 70% for training, 15% for validation and 15% for testing, as planned in Section 3.4. The test part was used only
once, for the final evaluation. All choices about scaling, stopping and model settings were made on the training and
validation parts.

The raw features had very different scales and shapes. Loudness lay roughly between 17 and 60, while fluctuation
strength ranged from about 1e-9 to 1e-2. Fluctuation strength was also extremely right-skewed (skewness 5.6 to 20.3),
and roughness was moderately right-skewed (skewness 0.8 to 1.5). Two numerical steps were therefore applied:

1. **Logarithmic transform.** Fluctuation strength and roughness are always positive, so they were replaced by their
   base-10 logarithm. This compresses values that span several orders of magnitude and stops a few large values from
   dominating the feature.
2. **Z-score standardisation.** Every feature was shifted to zero mean and scaled to unit variance. The mean and
   standard deviation were computed from the training part only and then applied unchanged to the validation and
   test parts, so that no information from the test data leaked into training.

To check that these steps were worthwhile, the ANN was trained on the same splits with three versions of the data
(Table 4). Without any scaling, the validation loss was highest and the test accuracy was lowest in the two-wheeler
cases. Standardisation reduced the validation loss in every case. Adding the logarithmic transform lowered it further
in the noisy cases, and kept it at or near zero in the cases without noise. The test accuracy of the two scaled
versions differed by at most three of 450 test windows. The logarithmic transform was kept because it gave the lowest
validation loss.

Table 4. Effect of normalisation on the ANN (best validation loss / test accuracy)

| Normalisation | A1 (without noise) | A2 (with noise) | B1 (without noise) | B2 (with noise) |
|---|---|---|---|---|
| None (raw features) | 0.0004 / 100% | 0.103 / 95.6% | 0.027 / 99.0% | 0.142 / 96.3% |
| Standardisation only | 0.0000 / 100% | 0.071 / 96.4% | 0.0001 / 100% | 0.032 / 100% |
| Log transform + standardisation | 0.0000 / 100% | 0.068 / 95.8% | 0.0000 / 100% | 0.023 / 100% |

## 4.4 Model training

The three models of Section 3.5 were trained separately on each of the four cases, giving twelve trained models. They
were implemented in Python with scikit-learn (SVM) and PyTorch (ANN and 1D-CNN). The two-class cases (Setup B) used a
single sigmoid output with binary cross-entropy loss. The three-class cases (Setup A) used a softmax output with
categorical cross-entropy loss.

- **SVM:** an SVM with a radial basis function (RBF) kernel [13]. The kernel was approximated with 300 Nystroem
  components, and a linear SVM (hinge loss) was trained on top by stochastic gradient descent for 100 epochs. This made
  it possible to record the loss after every epoch. The kernel width (gamma, from 0.1, 0.5 and 1.0) and the
  regularisation strength (alpha, from 0.0001 and 0.001) were chosen by validation accuracy. The selected values were
  gamma = 0.1 and alpha = 0.0001 in three cases, and gamma = 0.5 in case B2. The three-class cases used a one-vs-rest
  scheme.
- **ANN:** a fully connected network with two hidden layers of 32 and 16 neurons, ReLU activations and a dropout of
  0.1 (about 700 trainable parameters).
- **1D-CNN:** the four features of a window were treated as a short one-channel sequence [14]. Two 1D convolution
  layers (16 and 32 filters, kernel size 3), each followed by batch normalisation and ReLU, fed a dense layer of 32
  neurons with a dropout of 0.2 (about 6000 trainable parameters).

The ANN and 1D-CNN were trained with the AdamW optimiser:

- **Optimiser settings:** learning rate 0.001, weight decay 0.0001 and mini-batches of 64 windows.
- **Learning-rate schedule:** the learning rate was halved whenever the validation loss had not improved for 10
  epochs.
- **Early stopping:** training stopped when the validation loss had not improved for 30 epochs (at most 300 epochs),
  and the weights from the epoch with the lowest validation loss were kept.

All random seeds were fixed, so every run can be repeated exactly. The training and validation losses stayed close
together in every case (Fig. 5), which shows that the models did not overfit. The complete training of all models
took a few minutes on a laptop CPU.

## 4.5 Results

Table 5 lists the test accuracy of every model in every case. Because the classes are balanced, the macro-averaged
precision, recall and F1-score differ from the accuracy by less than 0.002 in all cases. Fig. 1 compares the
accuracy with and without background noise, and Figs. 2 to 4 show the confusion matrices.

Table 5. Test accuracy of the three models

| Model | A1: Standalone, without noise | A2: Standalone, with noise | B1: Two-wheeler, without noise | B2: Two-wheeler, with noise |
|---|---|---|---|---|
| SVM | 100% | 96.00% | 100% | 99.00% |
| ANN | 100% | 95.78% | 100% | 100% |
| 1D-CNN | 100% | **97.11%** | 100% | **100%** |

![Fig. 1](outputs/accuracy_comparison.png)

Fig. 1. Test accuracy with and without background noise for the standalone motor (left) and the two-wheeler (right).

![Fig. 2](outputs/svm_confusion_matrices.png)

Fig. 2. Confusion matrices of the SVM for the four cases.

![Fig. 3](outputs/ann_confusion_matrices.png)

Fig. 3. Confusion matrices of the ANN for the four cases.

![Fig. 4](outputs/cnn1d_confusion_matrices.png)

Fig. 4. Confusion matrices of the 1D-CNN for the four cases.

![Fig. 5](outputs/cnn1d_loss_curves.png)

Fig. 5. Training and validation loss of the 1D-CNN for the four cases. The dashed line marks the epoch whose weights
were kept.

**Without background noise (cases A1 and B1),** all three models classified every test window correctly in both
setups. With clean recordings, the four psychoacoustic features separated the conditions completely.

**With background noise in the standalone setup (case A2),** accuracy fell to between 95.78% and 97.11%. All errors
were confusions between the healthy and the heavy-load condition:

- SVM: 18 errors out of 300 healthy and heavy-load test windows.
- ANN: 19 errors.
- 1D-CNN: 13 errors.

None of the models ever confused the faulty motor with another condition. The traffic noise raised the loudness of all
recordings, and the loudness gap between the healthy and heavy-load conditions shrank from 7.6 to 4.0, which is
comparable to the spread within each class. The two classes therefore overlapped in feature space.

**With background noise in the two-wheeler (case B2),** the ANN and the 1D-CNN still reached 100%. The SVM made three
errors out of 300: two faulty windows were classified as healthy, and one healthy window as faulty.

## 4.6 Discussion

The preliminary study supports three conclusions for the main work:

1. **Psychoacoustic features are informative for motor health.** Healthy and faulty motors were separated almost
   perfectly in both setups, even with traffic noise present and with low-cost phone recordings.
2. **Background noise mainly affects the operating-condition (load) information, not the fault information.** The drop
   in accuracy under noise came entirely from healthy and heavy-load windows being mixed up. Fault detection itself
   was not affected in the standalone setup.
3. **The 1D-CNN was the most robust model.** It had the highest accuracy in the noisy standalone case and 100% in all
   other cases. It also converged fastest, reaching its best validation loss within 24 to 78 epochs. The ANN and the
   SVM followed closely, with the SVM the only model to make errors in the noisy two-wheeler case.

The preliminary study also has limitations, which the main experiments are designed to address:

- **Few recordings per class.** Each class came from one recording per noise condition, and neighbouring windows of
  the same recording are similar to each other. An additional check on the standalone healthy and faulty data, in
  which the first 70% of the windows were used for training and the last 30% for testing, gave the same accuracy.
  Even so, recordings from more motors and more sessions are needed before the results can be generalised.
- **Hold-out validation only.** The models were validated on a single held-out split. Repeated or k-fold
  cross-validation will be used in the main study to report the variation in accuracy.
- **Separate models per noise condition.** Each model was trained and tested within one noise condition. Training on
  mixed conditions, and testing on conditions not seen during training, is needed to judge robustness in real use.
- **Sound only.** Only the acoustic modality was used. The main study adds tonality to the acoustic features, adds the
  vibration features of Section 3.3, and moves from a BLDC motor to the PMSM traction motor of a four-wheeler.
