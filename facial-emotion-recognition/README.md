# Real-Time Facial Emotion Recognition Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15%2B-orange.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-3.0%2B-red.svg)](https://keras.io/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning and Computer Vision system for **Facial Expression Recognition (FER)**. This project implements a rigorous experimental pipeline benchmarking custom Convolutional Neural Networks against pre-trained Transfer Learning models (MobileNetV2), featuring automated data ingestion, class-imbalance weighting, in-graph vectorized augmentation, extensive diagnostic evaluation (confusion matrices, calibration curves, error analysis), and real-time webcam inference with confidence calibration and Top-3 predictions.

---

## Table of Contents

- [Overview & Research Scope](#overview--research-scope)
- [Pipeline Architecture](#pipeline-architecture)
- [Dataset (FER-2013)](#dataset-fer-2013)
- [Model Architectures & Experiment Matrix](#model-architectures--experiment-matrix)
  - [Custom CNN Baseline](#1-custom-cnn-baseline)
  - [Transfer Learning & Fine-Tuning (MobileNetV2)](#2-transfer-learning--fine-tuning-mobilenetv2)
  - [Controlled Experiment Matrix (E1–E4)](#3-controlled-experiment-matrix-e1e4)
- [Data Augmentation & Preprocessing](#data-augmentation--preprocessing)
- [Evaluation Suite & Diagnostics](#evaluation-suite--diagnostics)
- [Real-Time Computer Vision Pipeline](#real-time-computer-vision-pipeline)
- [Repository Structure](#repository-structure)
- [Installation & Quick Start](#installation--quick-start)
  - [1. Environment Setup](#1-environment-setup)
  - [2. Data Download & Preparation](#2-data-download--preparation)
  - [3. Training Models](#3-training-models)
  - [4. Model Evaluation & Comparison](#4-model-evaluation--comparison)
  - [5. Real-Time Application](#5-real-time-application)
- [Ethical Considerations & Limitations](#ethical-considerations--limitations)
- [Tech Stack](#tech-stack)

---

## Overview & Research Scope

Facial expression recognition is a challenging multi-class image classification task characterized by subtle facial muscle movements, high inter-class ambiguity (e.g., *Fear* vs. *Surprise*, *Sad* vs. *Neutral*), lighting variations, and acute class imbalance.

This project addresses these challenges through a principled experimental methodology:
1. **Custom Deep Architecture**: Designing, training, and regularizing a 3-block convolutional neural network built specifically for $48 \times 48$ grayscale face crops.
2. **Transfer Learning Adaptations**: Repurposing ImageNet-trained lightweight backbones (MobileNetV2) by upsampling and channel-broadcasting grayscale inputs to evaluate cross-domain representation transfer.
3. **In-Graph Vectorized Augmentation**: Integrating data augmentation directly as non-trainable Keras layers inside the computational graph, preventing train-test data leakage and eliminating training-serving skew.
4. **Class-Imbalance Compensation**: Mitigating skew using inverse-frequency class weights so minority classes (such as *Disgust*) are not marginalized.
5. **Beyond-Accuracy Evaluation**: Measuring performance via Top-3 accuracy, macro/weighted F1-scores, expected calibration error (ECE), per-class confusion matrices, and confident failure mining.
6. **Real-Time Deployment**: Pairing real-time Haar Cascade facial detection with calibrated neural inference in a user-facing interactive application.

---

## Pipeline Architecture

```text
                 FER-2013 Dataset (35,887 images)
                                │
               Automated Split & Checksum Validation
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
   Train (28,709)       Validation (3,589)         Test (3,589)
        │                       │                       │
 Class Weighting                │                  Held-Out Only
        │                       │                       │
 Vectorized Augmentation        │                       │
 (Flip, Rot, Zoom, Shift, Br.)  │                       │
        │                       │                       │
 ┌──────┴───────────────────────┴───────────────────────┐
 │               Model Architecture Candidates          │
 │  • E1: Custom CNN (Baseline)                         │
 │  • E2: Custom CNN + Data Augmentation                │
 │  • E3: MobileNetV2 (Frozen Feature Extractor)        │
 │  • E4: MobileNetV2 (Fine-Tuning Last 30 Layers)      │
 └──────────────────────────────┬───────────────────────┘
                                │
                   Training & Optimization
               (Adam, Categorical Cross-Entropy,
             EarlyStopping, ReduceLROnPlateau)
                                │
                                ▼
                   Diagnostic Evaluation Suite
         (Accuracy, Macro/Weighted F1, ECE, Top-3,
         Confusion Matrix, Loss Curves, Error Analysis)
                                │
                                ▼
                     Real-Time Inference Engine
         (OpenCV Haar Cascade Face Detector + Webcam / UI)
                                │
                 ┌──────────────┴──────────────┐
                 ▼                             ▼
        Predicted Expression           Top-3 Probabilities
         e.g., Happy (87.4%)           Happy:    87.4%
                                       Neutral:   6.8%
                                       Surprise:  3.1%
```

---

## Dataset (FER-2013)

The project uses the benchmark **FER-2013 (Facial Expression Recognition 2013)** dataset comprising **35,887** grayscale facial crops categorized into seven discrete expression categories:

| Class ID | Emotion Label | Train Samples | Validation Samples | Test Samples | Total | Inverse Class Weight ($w_c$) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **0** | **Angry**    | 3,995 | 467 | 491 | 4,953 | 1.03 |
| **1** | **Disgust**  | 436   | 56  | 55  | 547   | **9.41** |
| **2** | **Fear**     | 4,097 | 496 | 528 | 5,121 | 1.00 |
| **3** | **Happy**    | 7,215 | 895 | 879 | 8,989 | **0.57** |
| **4** | **Sad**      | 4,830 | 653 | 594 | 6,077 | 0.85 |
| **5** | **Surprise** | 3,171 | 415 | 416 | 4,002 | 1.29 |
| **6** | **Neutral**  | 4,965 | 607 | 626 | 6,198 | 0.83 |
| **Total** | — | **28,709** | **3,589** | **3,589** | **35,887** | — |

> [!IMPORTANT]
> The original dataset partition is strictly maintained:
> - `Training` $\rightarrow$ `data/processed/train` (28,709 images)
> - `PublicTest` $\rightarrow$ `data/processed/validation` (3,589 images)
> - `PrivateTest` $\rightarrow$ `data/processed/test` (3,589 images)
> 
> The test set (`PrivateTest`) is held out exclusively for final reporting and is never used during parameter optimization or early stopping.

### Class Imbalance Handling
To prevent models from biasing towards high-frequency categories (*Happy*) while ignoring minority categories (*Disgust*), training utilizes balanced inverse-frequency weights:
$$w_c = \frac{N_{\text{total}}}{K \cdot N_c}$$
where $K = 7$ (number of classes), $N_{\text{total}} = 28,709$, and $N_c$ is the frequency of class $c$.

---

## Model Architectures & Experiment Matrix

### 1. Custom CNN Baseline
Constructed with the Keras Functional API, the custom network processes $48 \times 48 \times 1$ inputs through three feature-extraction blocks followed by dense classification:

```text
Input (48×48×1)
  │
  ├── [Block 1] Conv2D(32, 3×3, same) ➔ ReLU ➔ BatchNormalization ➔ MaxPool(2×2) ➔ Dropout(0.25)
  │
  ├── [Block 2] Conv2D(64, 3×3, same) ➔ ReLU ➔ BatchNormalization ➔ MaxPool(2×2) ➔ Dropout(0.25)
  │
  ├── [Block 3] Conv2D(128, 3×3, same) ➔ ReLU ➔ MaxPool(2×2)
  │
  └── [Dense Head] Flatten ➔ Dense(128, ReLU) ➔ Dropout(0.50) ➔ Dense(7, Softmax)
```

- **Feature extraction hierarchy**: Block 1 captures low-level primitives (edges, gradients); Block 2 combines them into facial components (mouth corners, eyebrows, eye contours); Block 3 synthesizes holistic expression patterns.
- **Stabilization**: `BatchNormalization` smooths the loss landscape, while progressive dropout ($0.25 \rightarrow 0.50$) curbs co-adaptation of features.

### 2. Transfer Learning & Fine-Tuning (MobileNetV2)
MobileNetV2 brings depthwise separable convolutions pre-trained on ImageNet.
- **Input Adaptations**: Grayscale inputs are broadcast to 3 channels ($48 \times 48 \times 3$) and bilinearly upsampled to $96 \times 96 \times 3$ to align with the convolutional receptive field of MobileNetV2.
- **Normalization**: Rescaled from $[0, 255]$ to $[-1, 1]$ via $x \mapsto \frac{x}{127.5} - 1$.
- **Two-Stage Training**:
  - **E3 (Feature Extractor)**: MobileNetV2 backbone is completely frozen; only the custom classification head (`GlobalAveragePooling2D` $\rightarrow$ `Dense(128)` $\rightarrow$ `Dropout(0.5)` $\rightarrow$ `Softmax(7)`) is trained at learning rate $\eta = 10^{-3}$.
  - **E4 (Fine-Tuning)**: The top 30 layers of MobileNetV2 are unfrozen and trained end-to-end at a reduced learning rate $\eta = 10^{-5}$ with Adam.

### 3. Controlled Experiment Matrix (E1–E4)

| Exp ID | Model Architecture | Augmentation | Transfer Learning | Fine-Tuning | Purpose |
|:---:|:---|:---:|:---:|:---:|:---|
| **E1** | Custom CNN | No | No | No | Baseline establishing pure convolutional performance without regularization |
| **E2** | Custom CNN | **Yes** | No | No | Quantifies variance reduction and generalization gains from data augmentation |
| **E3** | MobileNetV2 | **Yes** | **Yes** (Frozen) | No | Evaluates transferability of general visual features to expression classification |
| **E4** | MobileNetV2 | **Yes** | **Yes** | **Yes** (Top 30) | Adapts pre-trained representations specifically to facial micro-movements |

---

## Data Augmentation & Preprocessing

Rather than expanding the dataset on disk with static transforms, augmentation is incorporated as **on-the-fly vectorized Keras layers**:

```python
layers.RandomFlip("horizontal")
layers.RandomRotation(0.20, fill_mode="reflect")        # ~ ±11.5 degrees
layers.RandomZoom(0.10, fill_mode="reflect")            # ±10% scale
layers.RandomTranslation(0.10, 0.10, fill_mode="reflect") # ±10% shift
layers.RandomBrightness(0.20, value_range=(0, 255))     # Lighting variations
```

### Key Engineering Advantages
1. **Zero Train-Serving Skew**: Because preprocessing and normalization layers are saved inside the `.keras` model artifact, webcam and production inference pipelines directly consume raw face crops without needing manual rescaling code.
2. **Deterministic Evaluation**: Augmentation layers automatically bypass during `training=False` (evaluation and inference).
3. **Hardware Acceleration**: Transform operations execute in parallel on batches without stalling the input pipeline.

---

## Evaluation Suite & Diagnostics

Evaluation extends well beyond aggregate accuracy:

1. **Classification Report**: Macro & weighted Precision, Recall, and F1-score across all 7 classes.
2. **Top-3 Accuracy**: Measures whether the true emotion exists within the model's top 3 highest-confidence predictions (crucial for ambiguous expressions).
3. **Confusion Matrices**: Both raw integer count matrices and row-normalized percentages to highlight systematic confusion pairs (e.g., *Sad* misclassified as *Neutral*).
4. **Calibration & Reliability Diagrams**: Expected Calibration Error (ECE) binned across confidence intervals to verify if a reported 80% confidence corresponds to an 80% empirical accuracy.
5. **Error Analysis Sheets**: Automatically extracting and plotting the top 25 highest-confidence errors to diagnose mislabeled dataset artifacts versus genuine model failures.

```
results/
├── E1_train_summary.json / E1_history.csv
├── E2_train_summary.json / E2_history.csv
├── comparison_table.md
├── accuracy_comparison.png
├── confusion_matrices.png
├── confusion_matrix_<EXP>.png
├── accuracy_curve_<EXP>.png
├── loss_curve_<EXP>.png
├── calibration_<EXP>.png
└── error_analysis/
    └── <EXP>_errors.png
```

---

## Real-Time Computer Vision Pipeline

```text
Webcam Stream (30 FPS)
         │
         ▼
OpenCV Haar Cascade Face Detector (`haarcascade_frontalface_default.xml`)
         │
         ▼
Face Region of Interest (ROI) Bounding Box Extraction
         │
         ▼
Square Aspect Crop & Resizing
         │
         ▼
Trained Model Inference (Softmax Output Vector $\hat{y} \in \mathbb{R}^7$)
         │
         ▼
Bounding Box Annotation + Dynamic Confidence Bars + Top-3 Emotion Ranking
```

The application outputs both the dominant predicted emotion and the top-3 probabilities, giving transparent visibility into prediction confidence and ambiguity.

---

## Repository Structure

```text
facial-emotion-recognition/
├── requirements.txt
├── run_all_experiments.ps1             # Orchestration script for full benchmark
│
├── data/
│   ├── raw/                            # Raw fer2013.csv archive
│   └── processed/                      # ImageFolder hierarchy (train/val/test)
│       ├── train/                      # 28,709 images in 7 emotion folders
│       ├── validation/                 # 3,589 images in 7 emotion folders
│       ├── test/                       # 3,589 images in 7 emotion folders
│       └── manifest.csv                # Dataset split manifest with checksums
│
├── models/                             # Saved .keras model checkpoints
│   ├── E1.best.keras
│   └── E2.best.keras
│
├── notebooks/                          # Interactive exploration & analysis
│
├── results/                            # Training logs, curves & evaluation reports
│   ├── class_distribution.png
│   ├── E1_history.csv
│   ├── E1_train_summary.json
│   └── ...
│
├── src/
│   ├── __init__.py
│   ├── config.py                       # Single source of truth for hyperparameters
│   ├── data_prep.py                    # Dataset download, validation & partitioning
│   ├── data.py                         # tf.data batched input pipelines
│   ├── augmentation.py                 # Vectorized Keras augmentation layers
│   ├── cnn_model.py                    # Custom CNN baseline architecture
│   ├── transfer_model.py               # MobileNetV2 transfer & fine-tuning models
│   ├── train.py                        # Standalone experiment training CLI
│   ├── evaluate.py                     # Full diagnostic evaluation harness
│   └── report.py                       # Comparative report & summary generator
│
└── tests/                              # Unit & integration tests
```

---

## Installation & Quick Start

### 1. Environment Setup

Clone the repository and create an isolated virtual environment:

```bash
# Navigate to project directory
cd facial-emotion-recognition

# Create and activate virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# (Optional: macOS / Linux)
# source .venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Download & Preparation

The dataset module downloads the official `fer2013.csv` from HuggingFace, validates row integrity, and unpacks it into the standard directory structure:

```bash
python -m src.data_prep
```

This generates `data/processed/train`, `data/processed/validation`, and `data/processed/test` partitioned into the 7 emotion subdirectories, along with `results/class_distribution.png`.

### 3. Training Models

Train individual experiments using the command-line interface:

```bash
# Train Experiment 1 (Custom CNN baseline)
python -m src.train --experiment E1

# Train Experiment 2 (Custom CNN with Vectorized Augmentation)
python -m src.train --experiment E2

# Train Experiment 3 (MobileNetV2 Transfer Learning - Frozen Backbone)
python -m src.train --experiment E3

# Train Experiment 4 (MobileNetV2 Fine-Tuning)
python -m src.train --experiment E4 --epochs 20

# Run a quick 2-epoch smoke test to verify pipeline connectivity
python -m src.train --experiment E1 --fast
```

#### Automated Sequential Execution
To train all four experiments sequentially (optimized for CPU/single-GPU memory efficiency):

```powershell
powershell -ExecutionPolicy Bypass -File run_all_experiments.ps1
```

### 4. Model Evaluation & Comparison

Run the comprehensive evaluation suite against the held-out test split:

```bash
# Evaluate a single trained model
python -m src.evaluate --experiment E1

# Evaluate all trained experiments
python -m src.evaluate --all

# Generate the cross-experiment markdown comparison table and summary figures
python -m src.report
```

### 5. Real-Time Application

Launch the real-time webcam detector or interactive web interface:

```bash
# Launch interactive Streamlit application
streamlit run app.py
```

---

## Ethical Considerations & Limitations

1. **Facial Expression vs. Internal Affect**: This model classifies *observable facial expressions* based on geometric and texture patterns in 2D images. It **does not** detect or determine genuine internal emotional states, psychological intent, or cognitive affect.
2. **Dataset Bias & Environmental Variance**: The FER-2013 dataset contains predominantly centered, cropped faces under synthetic or posed conditions. Predictions under severe off-angle rotations, partial occlusions (e.g., masks, hands), or poor illumination may degrade.
3. **Responsible Use**: Facial analysis models should never be deployed for punitive scoring, emotional surveillance, hiring decisions, or high-stakes behavioral analysis.

---

## Tech Stack

- **Core Framework**: [Python 3.10+](https://www.python.org/)
- **Deep Learning**: [TensorFlow](https://www.tensorflow.org/) & [Keras 3](https://keras.io/)
- **Computer Vision**: [OpenCV](https://opencv.org/) (Haar Cascade detection)
- **Data Engineering**: [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [Pillow](https://python-pillow.org/)
- **Evaluation & Diagnostics**: [scikit-learn](https://scikit-learn.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)
- **Interactive Web App**: [Streamlit](https://streamlit.io/)
