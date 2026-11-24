# Intelligent System for Alzheimer's Detection using DCGAN and CNN

A web application that classifies brain MRI scans into four stages of Alzheimer's disease. A DCGAN generates extra training images to fix class imbalance and overfitting, a CNN does the classification, and an autoencoder rejects images that are not MRI scans.

> **Disclaimer:** This is a research and learning project. It is not a medical device and must not be used for diagnosis.
>
> **Kaggle Repo:** https://www.kaggle.com/code/detty10/intelligent-system-for-alzheimer-disease-detection

## Overview

The original dataset is small (about 6,400 MRI images across 4 classes) and heavily imbalanced, so a CNN trained on it alone overfits. This project solves that in three steps:

1. **DCGAN data generation.** One generator per class creates synthetic MRI images to enlarge the training set.
2. **CNN classification.** A CNN is trained on the original plus generated images and exported to TensorFlow Lite for serving.
3. **Out-of-distribution check.** An autoencoder trained on MRI images measures reconstruction error. If a user uploads something that is not an MRI scan, the app says so instead of returning a wrong diagnosis.

## How it works

```
Uploaded image
      |
      v
Preprocess (grayscale, resize with padding to 128x128, scale to [0, 1])
      |
      v
Autoencoder: reconstruction error > 0.005 ?
      |-- yes --> "This is not an MRI image of Alzheimer's disease."
      |
      no
      v
CNN (TensorFlow Lite) --> predicted class
```

### Classes

The dataset has four classes: `Non_Demented`, `Very_Mild_Demented`, `Mild_Demented`, `Moderate_Demented`.

### DCGAN

- Input: latent vector of size 400. Output: 128x128 grayscale image.
- Generator: dense layer reshaped to 8x8x512, then transposed convolutions with batch normalization and LeakyReLU, with a residual block, ending in a `tanh` output.
- The discriminator is trained with a gradient penalty (weight 10).
- A separate generator is trained for each class. Only 80% of the original images of each class are used to train the generators, and the other 20% are kept as the validation set.
- The generators produced thousands of extra images per class (for example 9,949 Moderate, 9,284 Mild and 8,208 Very Mild images in the notebook run).

### CNN classifier

- Input: 128x128x1 grayscale image.
- Five convolution blocks (16, 32, 64, 128, 256 filters, two 3x3 convolutions each) with LeakyReLU, batch normalization, max pooling, dropout and L2 regularization, followed by global average pooling and dense layers with a 4-way softmax.
- Adam optimizer, categorical cross-entropy, early stopping and learning-rate reduction on plateau.
- Training augmentation: rotation up to 20 degrees and horizontal flip.

### Results (from `dcgan.ipynb`)

Both models were evaluated on the same held-out set of 1,281 original images that were never used to train the DCGAN.

| Training data | Validation accuracy |
|---|---|
| Original images only | 59% |
| Original + DCGAN-generated images | 98% |

With generated images, precision and recall are about 0.97 to 1.00 for every class. The validation set is small for the Moderate class (13 images), so treat that class's numbers with care.

### Autoencoder (non-MRI rejection)

A convolutional autoencoder (3 conv + pooling layers in the encoder, mirrored decoder) is trained on MRI images. At inference the mean squared reconstruction error is computed, and images above the threshold of `0.005` are rejected as non-MRI.

## Repository structure

| File | Purpose |
|---|---|
| `app.py` | Flask app: web page and `/api/predict` endpoint |
| `utils.py` | Preprocessing, autoencoder check, TFLite inference |
| `templates/index.html` | Upload page |
| `model_(98-20)_19-12.tflite` | CNN classifier used by the app |
| `autoencoder (1).h5` | Autoencoder used by the app |
| `model.tflite`, `autoencoder_model.h5` | Other exported model versions |
| `dcgan.ipynb` | DCGAN training, image generation, CNN training and evaluation, autoencoder training |
| `cnn.ipynb` | Variant of the training notebook (larger generated set) |
| `requirements.txt`, `runtime.txt` | Dependencies and Python version (3.10.9) |
| `Procfile`, `Aptfile` | Deployment settings (gunicorn, system libraries) |
| `.github/workflows/` | GitHub Actions workflows that deploy to Azure Web App |

## Getting started

Requirements: Python 3.10.

```bash
git clone https://github.com/mahmoud-montaser14/INTELLIGENT-SYSTEM-FOR-ALZHEIMER-S-DETECTION-USING-DCGAN-AND-CNN.git
cd INTELLIGENT-SYSTEM-FOR-ALZHEIMER-S-DETECTION-USING-DCGAN-AND-CNN
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, upload an MRI image and press **Predict**.

For production the project uses `gunicorn -b :$PORT app:app` (see `Procfile`).

## Usage

- Accepted formats: `png`, `jpg`, `jpeg`.
- Maximum upload size: 150 KB.

### API

```bash
curl -X POST -F "image=@scan.jpg" http://127.0.0.1:5000/api/predict
```

The response is plain text with the predicted class. If the image is rejected by the autoencoder, the response is the message `This is not an MRI image of Alzheimer's disease.` Invalid files, missing files and oversized files return a 400 error message.

## Tech stack

Python, TensorFlow / Keras, TensorFlow Lite, OpenCV, NumPy, Flask, gunicorn, GitHub Actions, Azure Web App.

## Author

Mahmoud Montaser
