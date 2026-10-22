# Real-Time Facial Emotion Recognition Using Deep Learning

## Project Overview

This project builds a **Real-Time Facial Emotion Recognition system using CNNs**, with a simple but complete Deep Learning pipeline.

The project will cover:

- Dataset collection
- Data preprocessing
- Data augmentation
- Custom CNN development
- Transfer learning
- Fine-tuning
- Model comparison
- Model evaluation
- Confusion matrix
- Training/validation curves
- Face detection
- Real-time webcam prediction
- Confidence scores
- Top-3 predictions
- Streamlit deployment

The goal is to keep the core implementation manageable while making the project substantial enough for a Deep Learning course project.

---

# 1. Dataset

## FER-2013

The primary dataset will be **FER-2013 (Facial Expression Recognition 2013)**.

It contains approximately **35,887 grayscale 48×48 facial images** divided into seven emotion classes.

### Emotion Classes

| Label | Emotion |
|---:|---|
| 0 | Angry |
| 1 | Disgust |
| 2 | Fear |
| 3 | Happy |
| 4 | Sad |
| 5 | Surprise |
| 6 | Neutral |

We do not need to collect thousands of photographs ourselves.

The original dataset should be preserved with its train/validation/test separation rather than randomly mixing all images, so that the final evaluation remains meaningful.

## Suggested Dataset Structure

```text
dataset/
│
├── train/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
│
├── validation/
│   ├── angry/
│   ├── disgust/
│   ├── fear/
│   ├── happy/
│   ├── sad/
│   ├── surprise/
│   └── neutral/
│
└── test/
    ├── angry/
    ├── disgust/
    ├── fear/
    ├── happy/
    ├── sad/
    ├── surprise/
    └── neutral/
```

---

# 2. What We Are Training

The project is a **7-class image classification problem**.

### Input

Each FER-2013 image is:

```text
48 × 48 × 1
```

The `1` represents the grayscale channel.

### Output

The model predicts probabilities for:

```text
Angry
Disgust
Fear
Happy
Sad
Surprise
Neutral
```

The final neural network layer therefore contains **7 neurons**.

For example, a prediction could look like:

```text
Angry      0.03
Disgust    0.01
Fear       0.05
Happy      0.82
Sad        0.04
Surprise   0.02
Neutral    0.03
```

The predicted class is:

```text
Happy
```

with a confidence of approximately:

```text
82%
```

The final layer will use **Softmax** to convert the model's raw outputs into class probabilities.

---

# 3. Complete Project Pipeline

```text
                    FER-2013
                        ↓
                 Data Cleaning
                        ↓
                 Preprocessing
                        ↓
               Data Augmentation
                        ↓
          ┌─────────────┴─────────────┐
          ↓                           ↓
     Custom CNN                Transfer Learning
          ↓                           ↓
          └─────────────┬─────────────┘
                        ↓
                 Model Evaluation
                        ↓
                Best Model Selection
                        ↓
                  Face Detection
                        ↓
                  Face Cropping
                        ↓
               Emotion Prediction
                        ↓
              Real-Time Webcam
```

---

# 4. Stage 1 — Data Preprocessing

Before training, all images need to be standardized.

## 4.1 Image Size

FER-2013 images are already 48×48, so the initial model will use:

```text
48 × 48
```

## 4.2 Normalization

Original pixel values range from:

```text
0–255
```

They will be normalized to:

```text
0–1
```

using:

\[
x_{normalized} = \frac{x}{255}
\]

This makes optimization more stable.

## 4.3 Tensor Representation

Each grayscale image will be represented as:

```text
48 × 48 × 1
```

where:

- 48 = image height
- 48 = image width
- 1 = grayscale channel

---

# 5. Stage 2 — Data Augmentation

Facial images can vary because of:

- Lighting
- Face orientation
- Position
- Scale
- Expression intensity
- Camera conditions

Therefore, augmentation will be applied to the **training set**.

It should not be applied to the test set because the test set should represent unseen data.

## Augmentation Methods

### Random Rotation

Small rotations such as:

```text
±10–15°
```

### Horizontal Flip

Mirror the face horizontally.

### Zoom

Slightly zoom into the face.

### Translation

Move the face slightly horizontally or vertically.

### Brightness Variation

Simulate different lighting conditions.

Conceptually:

```text
Original face
      ↓
 ┌────┼────┬────┐
 ↓    ↓    ↓    ↓
Flip Rotate Zoom Brightness
 ↓    ↓    ↓    ↓
       Training Examples
```

The purpose is to increase variation in the training data and reduce overfitting.

---

# 6. Stage 3 — Custom CNN

The first model will be a CNN designed specifically for this project.

This establishes a baseline and demonstrates understanding of CNN architecture rather than relying only on a pretrained model.

## Proposed Architecture

```text
Input
48×48×1
   ↓
Conv2D — 32 filters, 3×3
   ↓
ReLU
   ↓
Batch Normalization
   ↓
MaxPooling
   ↓
Dropout
   ↓
Conv2D — 64 filters, 3×3
   ↓
ReLU
   ↓
Batch Normalization
   ↓
MaxPooling
   ↓
Dropout
   ↓
Conv2D — 128 filters, 3×3
   ↓
ReLU
   ↓
MaxPooling
   ↓
Flatten
   ↓
Dense — 128
   ↓
Dropout
   ↓
Dense — 7
   ↓
Softmax
```

## 6.1 Convolution

Convolutional layers extract visual features.

The model can progressively learn:

```text
Edges
  ↓
Simple Shapes
  ↓
Eyes / Mouth / Facial Patterns
  ↓
Higher-Level Expression Features
```

## 6.2 ReLU

The Rectified Linear Unit introduces non-linearity:

\[
ReLU(x)=max(0,x)
\]

## 6.3 Max Pooling

Max pooling reduces the spatial dimensions of feature maps while retaining important features.

## 6.4 Batch Normalization

Batch normalization helps stabilize and speed up training.

## 6.5 Dropout

Dropout randomly disables a proportion of neurons during training.

Its purpose is to reduce overfitting.

## 6.6 Dense Layer

The dense layer combines the extracted features before classification.

## 6.7 Softmax

The final Softmax layer converts the seven outputs into probabilities.

---

# 7. Loss Function

Because this is a seven-class classification problem, the model will use **categorical cross-entropy**.

\[
L=-\sum_{i=1}^{7}y_i\log(\hat{y_i})
\]

where:

- \(y_i\) = actual class
- \(\hat{y_i}\) = predicted probability for class \(i\)

The objective of training is to minimize this loss.

---

# 8. Optimizer

The initial optimizer will be:

**Adam**

A starting learning rate can be:

```text
learning_rate = 0.001
```

Adam is a practical baseline for this type of image classification problem.

Other optimizers such as SGD or AdamW can be tested later if required, but they are not necessary for the first implementation.

---

# 9. Training Configuration

An initial configuration can be:

```text
Epochs: 30–50
Batch size: 64
Optimizer: Adam
Learning rate: 0.001
Loss: Categorical Cross-Entropy
```

The exact values can be adjusted based on actual training behavior.

We will monitor:

- Training accuracy
- Validation accuracy
- Training loss
- Validation loss

## Early Stopping

Early stopping will be used to stop training when validation performance stops improving.

This helps reduce unnecessary training and can reduce overfitting.

---

# 10. Stage 4 — Transfer Learning

The second major approach will use a pretrained CNN.

Possible models include:

- MobileNetV2
- EfficientNetB0

For this project, **MobileNetV2** is a practical starting point because it is relatively lightweight and suitable for deployment.

## Transfer Learning Pipeline

```text
Pretrained CNN
      ↓
Remove Original Classifier
      ↓
Global Average Pooling
      ↓
Dense Layer
      ↓
Dropout
      ↓
7-Class Softmax
```

The pretrained model contains general visual features learned from a large image dataset.

Initially, most pretrained layers can be frozen:

```text
Pretrained Layers → Frozen
Classifier        → Trainable
```

---

# 11. Fine-Tuning

After training the new classifier, some of the later pretrained layers can be unfrozen.

The model is then trained again with a much smaller learning rate.

```text
Pretrained Model
       ↓
Train New Classifier
       ↓
Unfreeze Last Few Layers
       ↓
Fine-Tune
       ↓
Final Model
```

Fine-tuning allows the pretrained features to adapt more specifically to facial-expression classification.

---

# 12. Model Comparison

The project will compare multiple approaches.

## Experiment 1 — Basic CNN

```text
Augmentation: No
Transfer Learning: No
Fine-Tuning: No
```

## Experiment 2 — CNN + Augmentation

```text
Augmentation: Yes
Transfer Learning: No
Fine-Tuning: No
```

## Experiment 3 — Transfer Learning

```text
Augmentation: Yes
Transfer Learning: Yes
Fine-Tuning: No
```

## Experiment 4 — Transfer Learning + Fine-Tuning

```text
Augmentation: Yes
Transfer Learning: Yes
Fine-Tuning: Yes
```

## Comparison Table

The actual values will be filled after experiments.

| Experiment | Augmentation | Transfer Learning | Fine-Tuning | Accuracy | F1 |
|---|---|---|---|---:|---:|
| E1 | No | No | No | TBD | TBD |
| E2 | Yes | No | No | TBD | TBD |
| E3 | Yes | Yes | No | TBD | TBD |
| E4 | Yes | Yes | Yes | TBD | TBD |

The project should report the actual measured results rather than predetermined accuracy values.

---

# 13. Model Evaluation

Evaluation should go beyond accuracy.

## 13.1 Accuracy

\[
Accuracy=
\frac{Correct\ Predictions}{Total\ Predictions}
\]

## 13.2 Precision

\[
Precision=
\frac{TP}{TP+FP}
\]

## 13.3 Recall

\[
Recall=
\frac{TP}{TP+FN}
\]

## 13.4 F1 Score

\[
F1=
2\frac{Precision\times Recall}
{Precision+Recall}
\]

These metrics should be reported both overall and, where appropriate, for each emotion class.

---

# 14. Confusion Matrix

A confusion matrix will show which emotions are being confused with one another.

Example structure:

```text
                 Predicted
             A  D  F  H  S  Su N
Actual A     ■  □  □  □  □  □  □
       D     □  ■  □  □  □  □  □
       F     □  □  ■  □  □  □  □
       H     □  □  □  ■  □  □  □
       S     □  □  □  □  ■  □  □
       Su    □  □  □  □  □  ■  □
       N     □  □  □  □  □  □  ■
```

This can reveal patterns such as one expression being frequently classified as another.

That information is more useful than accuracy alone because it helps explain the model's errors.

---

# 15. Training Curves

The project will generate at least two important graphs.

## 15.1 Accuracy Curve

Plot:

- Training accuracy
- Validation accuracy

against:

- Epoch

This can help identify:

- Learning progress
- Convergence
- Overfitting
- Underfitting

## 15.2 Loss Curve

Plot:

- Training loss
- Validation loss

against:

- Epoch

These curves provide evidence about how the model learned during training.

---

# 16. Stage 5 — Face Detection

The trained emotion classifier expects a face image.

Therefore, the webcam application needs a face detector before the classifier.

Initial implementation:

**OpenCV Haar Cascade**

Pipeline:

```text
Webcam Frame
      ↓
Face Detector
      ↓
Face Bounding Box
      ↓
Crop Face
      ↓
Resize
      ↓
Preprocess
      ↓
CNN
      ↓
Emotion
```

Other face detectors can be considered later:

- MediaPipe
- MTCNN
- RetinaFace

However, OpenCV Haar Cascade is sufficient for the first version and keeps the implementation simple.

---

# 17. Stage 6 — Real-Time Webcam Application

The final system will use a webcam to detect faces and classify their visible expression.

Example UI:

```text
┌─────────────────────────────────────────┐
│        REAL-TIME EMOTION AI             │
├─────────────────────────────────────────┤
│                                         │
│        ┌───────────────┐                │
│        │               │                │
│        │     FACE      │                │
│        │               │                │
│        └───────────────┘                │
│                                         │
│        Emotion: HAPPY                   │
│        Confidence: 87.4%                │
│                                         │
└─────────────────────────────────────────┘
```

The system should draw a bounding box around detected faces and display the predicted expression and confidence.

---

# 18. Confidence Scores

Instead of displaying only:

```text
Happy
```

the application should display the model's confidence.

For example:

```text
Emotion: Happy
Confidence: 87.4%
```

The confidence is based on the Softmax output.

It should be described as the model's predicted probability/confidence for the class, not as a guarantee that the person is actually experiencing that emotion.

---

# 19. Top-3 Predictions

The system can display the three highest-probability classes.

Example:

```text
1. Happy       87.4%
2. Neutral      6.8%
3. Surprise     3.1%
```

This gives a more informative representation of the model's uncertainty than showing only one class.

---

# 20. Final System Architecture

The completed project will have three major layers.

## Layer 1 — Deep Learning

```text
FER-2013
    ↓
Preprocessing
    ↓
Data Augmentation
    ↓
CNN
    ↓
Training
```

## Layer 2 — Experimentation and Evaluation

```text
CNN
 ↓
CNN + Augmentation
 ↓
Transfer Learning
 ↓
Fine-Tuning
 ↓
Accuracy
Precision
Recall
F1
Confusion Matrix
Training Curves
Error Analysis
```

## Layer 3 — Application

```text
Webcam
  ↓
Face Detection
  ↓
Face Crop
  ↓
Trained Model
  ↓
Emotion
  ↓
Confidence Score
  ↓
Top-3 Predictions
```

---

# 21. Technology Stack

## Programming

- Python

## Deep Learning

- TensorFlow
- Keras

## Image Processing

- OpenCV
- NumPy

## Data Analysis

- pandas
- Matplotlib
- scikit-learn

## Application

- Streamlit

## Overall Stack

```text
Python
│
├── TensorFlow / Keras
│     ├── Custom CNN
│     └── Transfer Learning
│
├── OpenCV
│     └── Face Detection
│
├── NumPy / pandas
│     └── Data Processing
│
├── scikit-learn
│     └── Evaluation
│
└── Streamlit
      └── Web Application
```

---

# 22. Proposed GitHub Repository Structure

```text
facial-emotion-recognition/
│
├── data/
│
├── notebooks/
│   ├── 01_data_analysis.ipynb
│   ├── 02_custom_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   └── 04_evaluation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── augmentation.py
│   ├── cnn_model.py
│   ├── transfer_model.py
│   ├── train.py
│   ├── evaluate.py
│   └── face_detector.py
│
├── models/
│   ├── custom_cnn.keras
│   └── best_model.keras
│
├── results/
│   ├── confusion_matrix.png
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   └── classification_report.csv
│
├── app.py
├── requirements.txt
└── README.md
```

---

# 23. Important Project Consideration

The system should be described as **facial-expression classification**, rather than as a system that can reliably determine a person's internal emotional state.

The model is learning visual patterns associated with the labeled categories in FER-2013.

Therefore, the project should use terminology such as:

> "The model predicts the facial expression category represented in the input image."

rather than claiming:

> "The model knows how the person actually feels."

This is an important limitation to mention in the report.

---

# 24. Expected Final Demonstration

The final demo should allow a user to:

1. Open the Streamlit application.
2. Enable the webcam.
3. Show their face.
4. Detect the face automatically.
5. Draw a bounding box around the face.
6. Predict one of seven expression classes.
7. Display the confidence score.
8. Display the top-3 predictions.
9. Optionally upload a facial image instead of using the webcam.

Example:

```text
Input
  ↓
Live Webcam
  ↓
Face Detected
  ↓
Face Cropped
  ↓
Preprocessed
  ↓
Best Trained Model
  ↓
┌─────────────────────┐
│ Happy      87.4%    │
│ Neutral     6.8%    │
│ Surprise    3.1%    │
└─────────────────────┘
```

---

# 25. Main Research/Technical Questions We Can Answer

The project can investigate:

1. How well does a custom CNN classify facial expressions?
2. Does data augmentation improve generalization?
3. Does transfer learning improve performance?
4. Does fine-tuning improve performance further?
5. Which emotion classes are most difficult to classify?
6. Which emotions are commonly confused?
7. How does the model behave on real-time webcam images?
8. How does the model's confidence vary across predictions?
9. What types of images produce incorrect predictions?
10. What trade-offs exist between model accuracy and deployment complexity?

These questions give the project a stronger experimental component than simply training a classifier.

---

# 26. Final Project Title Options

### Simple

**Facial Emotion Recognition Using Convolutional Neural Networks**

### Better

**Deep Learning-Based Facial Expression Recognition Using CNN and Transfer Learning**

### Application-oriented

**Real-Time Facial Expression Recognition Using Deep Learning and Computer Vision**

### Most complete

**Real-Time Facial Expression Recognition Using CNN, Transfer Learning, and Computer Vision**

---

# 27. Summary

The project will combine:

```text
FER-2013
    +
Data Preprocessing
    +
Data Augmentation
    +
Custom CNN
    +
Transfer Learning
    +
Fine-Tuning
    +
Model Comparison
    +
Accuracy / Precision / Recall / F1
    +
Confusion Matrix
    +
Training Curves
    +
Face Detection
    +
Real-Time Webcam
    +
Confidence Scores
    +
Top-3 Predictions
    +
Streamlit
```

The core Deep Learning component remains manageable, while the complete system provides enough experimentation, evaluation, visualization, and deployment work to make it a substantial college project.
