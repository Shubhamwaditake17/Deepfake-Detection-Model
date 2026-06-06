# Deepfake Detection Model

## Overview

This project presents a Deepfake Image Detection System built using TensorFlow and MobileNetV2 Transfer Learning. The objective is to classify facial images as either Real or Fake by leveraging a pretrained deep learning model and fine-tuning it on a deepfake dataset.

Deepfake technology has become increasingly sophisticated, making it difficult to distinguish manipulated content from authentic media. This project addresses that challenge by utilizing convolutional neural networks (CNNs) and transfer learning techniques to automatically identify manipulated facial images.

The model is trained on labeled real and fake face images, performs image preprocessing and augmentation, and predicts whether a given image is genuine or artificially generated.

---

## Problem Statement

Deepfake images generated using modern AI techniques can be used for misinformation, identity theft, fraud, and digital manipulation. Manual detection is often unreliable and time-consuming.

The goal of this project is to build a machine learning system capable of automatically detecting deepfake images with high accuracy using computer vision and deep learning techniques.

---

## Features

- Deepfake image classification
- Transfer Learning using MobileNetV2
- Image preprocessing and normalization
- Data augmentation for improved generalization
- Binary classification (Real vs Fake)
- Training and validation performance visualization
- Single image prediction pipeline
- Streamlit/Gradio integration support
- Scalable architecture for future deployment

---

## Technology Stack

### Programming Language

- Python

### Libraries and Frameworks

- TensorFlow
- Keras
- NumPy
- Matplotlib
- OpenCV
- Scikit-Learn
- Streamlit
- Gradio

### Deep Learning Model

- MobileNetV2 (Pretrained on ImageNet)

---

## Project Structure

```text
Deepfake-Detection-Model/
│
├── app.py
├── gradio_app.py
├── Deepfake_Detection_Model.ipynb
├── test.png
├── requirements.txt
├── README.md
│
├── Dataset/
│   ├── Train/
│   │   ├── Real/
│   │   └── Fake/
│   │
│   ├── Validation/
│   │   ├── Real/
│   │   └── Fake/
│   │
│   └── Test/
│       ├── Real/
│       └── Fake/
│
└── deepfake_model.h5
```

---

## Dataset Organization

The dataset is divided into three subsets:

### Training Set

Used for learning patterns and updating model weights.

```text
Train/
├── Real
└── Fake
```

### Validation Set

Used during training to evaluate model performance and detect overfitting.

```text
Validation/
├── Real
└── Fake
```

### Test Set

Used after training to evaluate the final model on unseen images.

```text
Test/
├── Real
└── Fake
```

---

## Data Preprocessing

Before training, images undergo several preprocessing operations.

### Normalization

Pixel values originally range between:

```text
0 - 255
```

They are normalized using:

```python
rescale=1./255
```

Resulting range:

```text
0 - 1
```

This improves training stability and convergence speed.

### Data Augmentation

The following transformations are applied to training images:

```python
rotation_range=10
zoom_range=0.2
horizontal_flip=True
```

Benefits:

- Improves generalization
- Reduces overfitting
- Creates additional training variations

---

## Model Architecture

### Base Model

The project uses MobileNetV2 pretrained on ImageNet.

```python
MobileNetV2(
    weights='imagenet',
    include_top=False,
    input_shape=(128,128,3)
)
```

Reasons for choosing MobileNetV2:

- Lightweight architecture
- Fast training
- High accuracy
- Effective feature extraction
- Suitable for deployment

---

### Custom Classification Head

```python
GlobalAveragePooling2D()
Dense(128, activation='relu')
Dropout(0.5)
Dense(1, activation='sigmoid')
```

#### GlobalAveragePooling2D

Reduces feature maps into a compact feature vector.

#### Dense Layer

Learns deepfake-specific features extracted from MobileNetV2.

#### Dropout

Reduces overfitting by randomly deactivating neurons during training.

#### Sigmoid Layer

Outputs a probability value:

```text
0 → Fake
1 → Real
```

---

## Transfer Learning Strategy

Most layers of MobileNetV2 are frozen:

```python
layer.trainable = False
```

This preserves previously learned visual features such as:

- Edges
- Facial structures
- Textures
- Patterns

Only the final layers are fine-tuned for deepfake detection.

Benefits:

- Faster training
- Reduced computational cost
- Better performance on small datasets

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

Adam optimizer is used for efficient gradient updates.

### Loss Function

Binary Cross Entropy is used because the problem involves two classes:

- Real
- Fake

### Evaluation Metric

Accuracy is used to measure classification performance.

---

## Training Process

```python
model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    steps_per_epoch=500,
    validation_steps=100
)
```

### Training Workflow

1. Load image batches
2. Extract features using MobileNetV2
3. Generate predictions
4. Calculate loss
5. Update trainable weights
6. Evaluate validation performance
7. Repeat for multiple epochs

---

## Prediction Pipeline

The prediction workflow is:

```text
Input Image
      ↓
Resize (128x128)
      ↓
Normalize
      ↓
Convert to Tensor
      ↓
Model Prediction
      ↓
Real or Fake Classification
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

The model learns visual inconsistencies commonly present in deepfake images, including:

- Facial distortions
- Blending artifacts
- Texture irregularities
- Synthetic image patterns

Training and validation accuracy are monitored throughout the training process to ensure proper learning and minimize overfitting.

---

## Applications

This project can be applied in:

- Social Media Content Verification
- Digital Forensics
- News Verification Systems
- Cybersecurity Platforms
- Identity Protection Systems
- Fake Media Detection Tools
- AI Content Moderation Systems

---

## Future Improvements

Potential enhancements include:

- Video Deepfake Detection
- Face Detection using MTCNN
- Real-time Webcam Detection
- Explainable AI Visualization
- Ensemble Learning Approaches
- Deployment using Docker
- Cloud-based Inference APIs
- Mobile Application Integration

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Shubhamwaditake17/Deepfake-Detection-Model.git
```

Move into the project directory:

```bash
cd Deepfake-Detection-Model
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Jupyter Notebook

```bash
jupyter notebook
```

Open:

```text
Deepfake_Detection_Model.ipynb
```

### Streamlit Application

```bash
streamlit run app.py
```

### Gradio Application

```bash
python gradio_app.py
```

---

## Author

**Shubham Waditake**

Third Year Computer Science Engineering Student

Areas of Interest:

- Artificial Intelligence
- Machine Learning
- Deep Learning
- Computer Vision
- Data Structures and Algorithms

---
