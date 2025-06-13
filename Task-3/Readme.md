# Task 3 - CNN Image Classification on MNIST

## 📌 Overview
This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

## 🧠 Model Architecture
- **Input Shape**: 28x28x1 grayscale images
- **Layers**:
  - Conv2D (32 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Conv2D (64 filters, 3x3 kernel) + ReLU
  - MaxPooling2D (2x2)
  - Flatten
  - Dense (64 units, ReLU)
  - Dense (10 units, Softmax)

## 📊 Dataset
- **MNIST**: 70,000 grayscale images of handwritten digits (0–9)
  - Training: 60,000 images
  - Testing: 10,000 images
- Normalized pixel values to [0, 1]
- Reshaped for CNN input compatibility

## 🧪 Performance
- **Output Layer**: 10 neurons (for digits 0–9)
- Model evaluates classification accuracy using:
  - Confusion Matrix
  - Classification Report
- Trained model can be saved using Keras's `.save()` method

## 📁 Files Included
- `Task_3.ipynb` – Jupyter Notebook with the full training and evaluation pipeline
- `Confusion_Matrix.png` – Visual representation of model predictions
- `Classification_Report.png` – Precision, recall, and F1-score metrics
- `cnn_image_classifier.h5` *(if saved)* – Trained model file
- `README.md`

## 🚀 How to Run
Use Google Colab or Jupyter Notebook:
```bash
jupyter notebook Task_3.ipynb

🛠️ Requirements
Python 3.x

TensorFlow

NumPy

Matplotlib

Seaborn

scikit-learn

🙌 Acknowledgements
Dataset: MNIST

Frameworks: TensorFlow/Keras