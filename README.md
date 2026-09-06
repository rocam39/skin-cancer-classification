# Skin Cancer Classification Using ResNet-50

A deep learning-based image classification system for classifying skin lesions into seven diagnostic categories using a pretrained ResNet-50 convolutional neural network.

## Overview

This project uses transfer learning with ResNet-50 pretrained on ImageNet to classify dermatoscopic images from the HAM10000 dataset.

The model performs seven-class classification and uses techniques such as:

- Transfer learning
- Dropout regularization
- L2 regularization
- Class-weighted loss
- Learning-rate scheduling
- Early stopping
- Gradient clipping

A separate prediction script allows a trained model to classify a new skin-lesion image.

## Classes

The model predicts seven categories:

| Code | Diagnosis |
|---|---|
| `nv` | Melanocytic nevi |
| `mel` | Melanoma |
| `df` | Dermatofibroma |
| `akiec` | Actinic keratoses / Bowen's disease |
| `bcc` | Basal cell carcinoma |
| `bkl` | Benign keratosis-like lesions |
| `vasc` | Vascular lesions |

## Model Architecture

The project uses a pretrained ResNet-50 model.

The original fully connected classification layer is replaced with:

```text
ResNet-50
    ↓
Dropout (30%)
    ↓
Fully Connected Layer
    ↓
7 Classes
```

The ResNet-50 backbone uses ImageNet pretrained weights and is fine-tuned for skin-lesion classification.

## Dataset

The project uses the **HAM10000 (Human Against Machine with 10000 training images)** dataset.

The dataset contains dermatoscopic images representing seven different categories of skin lesions.

The dataset itself is **not included in this repository** because of its size.

## Training

The dataset is divided into:

- 70% training
- 15% validation
- 15% testing

The training pipeline includes:

- Image resizing to `224 × 224`
- Random horizontal flipping
- Random rotation
- Random resized cropping
- ImageNet normalization
- Class-weighted cross-entropy loss
- Adam optimizer
- Learning-rate reduction on validation-loss plateau
- Early stopping
- Gradient clipping

The best model is selected using validation performance. The test set is reserved for final evaluation.

## Prediction

`predict.py` loads the trained ResNet-50 model and label encoder, preprocesses a new image, and returns:

- Predicted class
- Class index
- Prediction confidence
- Diagnostic category
- Benign/malignant classification

Run the prediction script with:

```bash
python predict.py
```

The script will ask for the path of the image to classify.

### Model Weights

The trained `.pth` model weights are **not included in this repository due to their file size**.

The model can be reproduced by running:

```bash
python train.py
```

After training, the model weights will be saved to:

```text
models/best_skin_cancer_resnet50_model.pth
```

The label encoder is also generated during training.

## Installation

Clone the repository and install the required dependencies:

```bash
git clone <your-repository-url>
cd skin-cancer-classification
pip install -r requirements.txt
```

## Project Structure

```text
skin-cancer-classification/
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
│
└── models/
    └── (generated model files)
```

The HAM10000 dataset should be downloaded separately and stored locally. It is not included in the repository because of its size.

## Results

The training script generates:

- Training and validation accuracy curves
- Confusion matrix
- Classification report
- Final test-set evaluation

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not be used as a substitute for professional medical evaluation.

## Technologies

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Pillow
