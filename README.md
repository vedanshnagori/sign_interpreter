
# Sign Language Interpreter using CNN & TensorFlow Lite

A deep learning-based Sign Language Interpreter that recognizes hand gestures representing the English alphabet (A–Z). This project combines **CNN**, **transfer learning**, and **TensorFlow Lite** for accurate and lightweight sign recognition — ready for real-time deployment.

![Architecture](./Architecture.png)

---

## Project Overview

| File/Folder                 | Description                                              |
|----------------------------|----------------------------------------------------------|
| `Explomodeltraining.ipynb` | Jupyter Notebook for model training and evaluation       |
| `asl_cnn_model4_int8.tflite` | Optimized model for mobile deployment using TFLite       |
| `Architecture.png`         | Model architecture diagram                              |
| `Project_report.pdf`       | Detailed report including methodology and results        |
| `Presentation Deck.pdf`    | Presentation slides for the project                     |

---

## Objectives

- Gathered a labeled dataset of American Sign Language hand gestures representing the 26 English alphabets for model training and testing.
- Resized all images to a fixed size for consistency, converted them to grayscale to reduce complexity, and normalized pixel values (0–1) to help the model train faster and more efficiently.
- To make the model run faster and use less memory on mobile devices, it was converted to TensorFlow Lite format. Then, quantization was applied to shrink the model by using smaller, more efficient numbers—without losing much accuracy
- Set up the optimized model to work smoothly in real-time environments, like mobile or embedded systems. Tested it with live inputs to ensure it could make quick and accurate predictions, ready for use in real-world applications.

---

## Dataset

- Dataset: American Sign Language (ASL) Alphabet Dataset
- Total Classes: **26 (A–Z)**
- Input Image Size: **64x64 grayscale**
- Preprocessing: Normalization, Resizing, Data Augmentation
- Tech Stack
  - `Python`
  - `TensorFlow & Keras`
  - `OpenCV`
  - `NumPy / Pandas`
  - `Matplotlib`
  - `TensorFlow Lite`

---

## Model Architecture

A custom CNN with the following:
- 3 Convolutional layers + MaxPooling
- Dropout layers to prevent overfitting
- Dense layers with `softmax` activation
- Final model exported as `.tflite` with quantization

📈 **Accuracy Achieved**: ~96%

> See `Architecture.png` for the full structure.

---

## Model Optimization (TFLite)

Model converted for mobile deployment:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

