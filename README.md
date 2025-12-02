# 🎭 Emotion Detection from Facial Images using ResNet18

This project implements a deep learning model for classifying human emotions from facial images using the FER2013 dataset. Our goal was to detect five distinct emotions — **angry, happy, neutral, sad, and surprise** — with a focus on performance, generalization, and handling class imbalance.

## 🔧 Key Features

- **Backbone**: Pre-trained **ResNet18** architecture with a custom classification head
- **Loss Function**: **Focal Loss** for better handling of class imbalance
- **Data Augmentation**: Includes random rotations, flips, color jitter, perspective distortion, and erasing
- **Optimization**: Utilizes **AdamW** optimizer with **OneCycleLR** scheduler for stable and efficient training
- **Evaluation**: Provides loss/accuracy curves, per-class accuracy, and confusion matrix
- **Visualization**: Sample predictions (correct and incorrect) are visualized and saved as PNG

## 📊 Results

- Validation Accuracy - 76.22%
- Training Accuracy - 89.7%
- Per-class accuracy and confusion matrix visualized for insights

## 📁 Dataset

We used the [FER2013 dataset](https://www.kaggle.com/datasets/msambare/fer2013), a well-known benchmark for facial emotion recognition. The dataset was filtered down to 5 emotion classes due to class imbalance in the original 7-class distribution.


## 📈 Training and Evaluation

- **Epochs**: 20
- **Batch size**: 64
- **Scheduler**: OneCycleLR
- **Device**: CUDA (if available)

## 🧠 Future Improvements

- Enhance regularization to reduce overfitting
- Explore data resampling techniques like SMOTE
- Incorporate additional emotion classes with balanced data
