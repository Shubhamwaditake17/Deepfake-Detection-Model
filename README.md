# Deepfake Detection Model

## Overview

This project implements a Deepfake Image Detection System using TensorFlow and MobileNetV2 Transfer Learning. The model is trained to classify facial images as either Real or Fake by learning visual patterns commonly found in manipulated images.

The project focuses on the machine learning pipeline, including data preprocessing, augmentation, transfer learning, model training, evaluation, and prediction.

---

## Problem Statement

Deepfake technology can generate highly realistic fake images that are difficult to distinguish from authentic content. Such manipulated media can be used for misinformation, fraud, and identity misuse.

The objective of this project is to build a deep learning model capable of automatically detecting deepfake images with high accuracy.

---

## Features

- Binary image classification (Real vs Fake)
- Transfer Learning using MobileNetV2
- Image preprocessing and normalization
- Data augmentation
- Model training and validation
- Accuracy and loss visualization
- Single image prediction support

---

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- OpenCV
- Matplotlib
- Scikit-Learn

---

## Dataset Structure

```text
Dataset/
│
├── Train/
│   ├── Real/
│   └── Fake/
│
├── Validation/
│   ├── Real/
│   └── Fake/
│
└── Test/
    ├── Real/
    └── Fake/
```

### Train Set

Used for learning image patterns and updating model weights.

### Validation Set

Used during training to monitor performance and detect overfitting.

### Test Set

Used to evaluate the final model on unseen data.

---

## Data Preprocessing

The following preprocessing techniques are applied:

### Normalization

```python
rescale=1./255
```

Converts pixel values from:

```text
0-255
```

to

```text
0-1
```

which improves training stability.

### Data Augmentation

```python
rotation_range=10
zoom_range=0.2
horizontal_flip=True
```

Benefits:

- Improves generalization
- Reduces overfitting
- Simulates real-world image variations

---

## Model Architecture

### Base Model

```python
MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)
```

MobileNetV2 is pretrained on ImageNet and acts as a feature extractor.

### Custom Layers

```python
GlobalAveragePooling2D()
Dense(128, activation='relu')
Dropout(0.5)
Dense(1, activation='sigmoid')
```

#### GlobalAveragePooling2D

Converts feature maps into a compact feature vector.

#### Dense Layer

Learns deepfake-specific patterns.

#### Dropout

Reduces overfitting.

#### Sigmoid Layer

Outputs probability for binary classification.

---

## Transfer Learning

Most MobileNetV2 layers are frozen during training.

```python
layer.trainable = False
```

This allows the model to reuse previously learned image features while fine-tuning only the final layers for deepfake detection.

---

## Model Compilation

```python
model.compile(
    optimizer=Adam(learning_rate=0.00005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)
```

### Optimizer

Adam

### Loss Function

Binary Cross Entropy

### Evaluation Metric

Accuracy

---

## Training

```python
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    steps_per_epoch=500,
    validation_steps=100
)
```

Training workflow:

1. Load image batches
2. Extract features using MobileNetV2
3. Generate predictions
4. Compute loss
5. Update trainable weights
6. Validate model performance
7. Repeat for all epochs

---

## Prediction Pipeline

```text
Input Image
      ↓
Resize to 128×128
      ↓
Normalize
      ↓
Feature Extraction
      ↓
Classification
      ↓
Real / Fake
```

Example:

```python
pred = model.predict(img)[0][0]

if pred > 0.5:
    print("Real")
else:
    print("Fake")
```

---

## Results

The model learns to identify:

- Facial inconsistencies
- Image manipulation artifacts
- Texture irregularities
- Synthetic image patterns

Training and validation metrics are monitored to ensure proper learning and reduce overfitting.

---

## Applications

- Digital Forensics
- Social Media Verification
- Fake News Detection
- Content Moderation
- Identity Protection Systems
- AI-Generated Media Analysis

---

## Future Improvements

- Video Deepfake Detection
- Real-Time Detection
- Explainable AI Visualization
- Ensemble Models
- Larger Dataset Training
- Deployment as an API

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shubhamwaditake17/Deepfake-Detection-Model.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Open and run:

```text
Deepfake_Detection_Model.ipynb
```

or

```bash
jupyter notebook
```

and execute all cells.

---

## Author

Shubham Waditake

Third Year Computer Science Engineering

Areas of Interest:
- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
