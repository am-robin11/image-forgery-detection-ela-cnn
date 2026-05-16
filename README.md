# Image Forgery Detection with ELA + Lightweight CNN

Final project for **CSE 422 — Neural Networks Lab** at UITS.

Detects image tampering using **Error Level Analysis (ELA)** as preprocessing and a compact **CNN** for binary classification, with **Grad-CAM** heatmaps for explainability. Trained and evaluated on the [CASIA v2.0](https://www.kaggle.com/datasets/sophatvathana/casia-dataset) dataset.

---

## How It Works

Standard forgery detection struggles because manipulated regions can look visually identical to authentic content. The key insight here is to work in the *compression artifact domain* rather than the pixel domain:

1. **ELA Conversion** — Each image is re-saved at a fixed JPEG quality (90), then the pixel-wise difference between the original and the re-saved image is amplified. Tampered regions, having been compressed differently, show up as brighter areas in the ELA image.
2. **CNN Classification** — A lightweight 3-block CNN is trained on these ELA images to classify each image as authentic or tampered.
3. **Grad-CAM Explainability** — Gradient-weighted Class Activation Maps highlight which spatial regions drove the model's prediction, making the decision interpretable.
4. **FFT Analysis** — Frequency-domain comparisons provide an additional lens for detecting compression inconsistencies at splicing boundaries.

```
Raw Image
    │
    ▼
ELA Conversion (re-compress → diff → amplify → resize 128×128)
    │
    ▼
Lightweight CNN ──► Authentic / Tampered
    │
    ▼
Grad-CAM Heatmap (overlaid on original image)
```

---

## Model Architecture

A compact Sequential CNN (~4.3M parameters, 16 MB) — efficient enough to train on a free Colab GPU in ~1 hour.

| Layer | Output Shape | Params |
|---|---|---|
| Conv2D 32 + BN + MaxPool | 64 × 64 × 32 | 1,024 |
| Conv2D 64 + BN + MaxPool | 32 × 32 × 64 | 18,752 |
| Conv2D 128 + BN + MaxPool | 16 × 16 × 128 | 74,368 |
| Flatten | 32,768 | — |
| Dense 128 + Dropout 0.5 | 128 | 4,194,432 |
| Dense 1 (sigmoid) | 1 | 129 |

- **Loss:** Binary cross-entropy
- **Optimizer:** Adam
- **Regularization:** BatchNorm + Dropout (0.5)
- **Callbacks:** EarlyStopping + ModelCheckpoint

---

## Results

Evaluated on a held-out test set of **951 samples** from CASIA v2.0.

| Metric | Score |
|---|---|
| Test Accuracy | **93.48%** |
| Precision (Tampered) | 83.11% |
| Recall (Tampered) | 87.92% |
| F1-Score (Tampered) | 85% |
| **ROC-AUC** | **98.09%** |

**Classification Report:**

|  | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Authentic | 0.97 | 0.95 | 0.96 | 744 |
| Tampered | 0.83 | 0.88 | 0.85 | 207 |
| **Weighted Avg** | **0.94** | **0.93** | **0.94** | **951** |

**Confusion Matrix (Test Set):**

|  | Pred: Authentic | Pred: Tampered |
|---|---|---|
| **Actual: Authentic** | 707 | 37 |
| **Actual: Tampered** | 25 | 182 |

---

## Dataset

[CASIA v2.0 Image Tampering Detection Dataset](https://www.kaggle.com/datasets/sophatvathana/casia-dataset)

| Split | Samples |
|---|---|
| Training | 7,600 |
| Validation | 950 |
| Test | 951 |
| **Total** | **9,501** |

Tampered images include splicing, copy-move, and post-processing (blur, noise, recompression). The dataset is organized as:

```
Dataset/
├── Au/   ← 7,491 authentic images
└── Tp/   ← 5,123 tampered images
```

---

## Usage

Open `image_forgery_detection.ipynb` in **Google Colab** (recommended). Mount your Drive and update the dataset path in Cell 2:

```python
DRIVE_PROJECT_PATH = "/content/drive/MyDrive/<your-dataset-folder>"
```

Then run all cells. The notebook handles ELA conversion, training, evaluation, Grad-CAM, and FFT analysis end-to-end.

---

## Environment

| Component | Spec |
|---|---|
| Platform | Google Colab (Free Tier) |
| GPU | Tesla T4 (15 GB VRAM) |
| Framework | TensorFlow 2.x / Keras |
| Python | 3.10 |
| Key Libraries | NumPy, Pillow, OpenCV, scikit-learn, Matplotlib |

---

## Files

| File | Description |
|---|---|
| `image_forgery_detection.ipynb` | Full pipeline — ELA preprocessing, CNN training, evaluation, Grad-CAM, FFT |
| `report.pdf` | Full project report (CSE 422, UITS) |
