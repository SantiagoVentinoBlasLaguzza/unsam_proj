# Alzheimer fMRI Analysis with Variational Autoencoders (VAE)

## Overview
This repository contains the implementation of a **Beta Variational Autoencoder (Beta-VAE)** to analyze fMRI tensor data from Alzheimer's patients. The project focuses on training a latent space representation of the data and classifying subjects into different categories (Alzheimer's Disease - AD, Cognitively Normal - CN, and Others) based on extracted latent features.

## Features
- **Beta-VAE for latent space learning**
- **Preprocessing pipeline** to load, normalize, and split fMRI data
- **Latent space visualization** using PCA, t-SNE, and UMAP
- **Classification of latent embeddings** using Logistic Regression, SVM, Random Forest, and other models
- **Hyperparameter tuning** with GridSearchCV for Random Forest
- **Plotting tools** for ROC curves, PR curves, and confusion matrices

## Repository Structure
```
📂 Project Root
├── config.yaml              # Configuration file with dataset paths and hyperparameters
├── train.py                 # Training script for BetaVAE
├── vae.py                   # Implementation of BetaVAE architecture
├── preprocess.py            # Preprocessing script for dataset preparation
├── classifier.py            # Classifier script for analyzing latent space embeddings
├── latent.py                # Script to extract and visualize latent embeddings
├── utils.py                 # Utility functions (normalization, data handling, etc.)
├── DataBaseSubjects.csv      # CSV file with metadata of fMRI subjects
└── TensorData/              # Directory containing fMRI tensors
```

## Installation
Clone the repository and install the dependencies:
```bash
$ git clone https://github.com/your_username/alzheimer-vae.git
$ cd alzheimer-vae
$ pip install -r requirements.txt
```

## Usage

### 1. Preprocessing
Run the preprocessing script to split the dataset into training, validation, and test sets:
```bash
$ python preprocess.py
```

### 2. Training the Beta-VAE
Train the model with the following command:
```bash
$ python train.py --epochs 1000 --latent_dim 250 --beta 0.5
```

### 3. Extracting Latent Embeddings
Once the model is trained, extract latent representations for visualization:
```bash
$ python latent.py --project_dir /path/to/project --latent_dim 250 --beta 0.1
```

### 4. Classifying Subjects
Train classifiers on the extracted latent features:
```bash
$ python classifier.py --project_dir /path/to/project --train_paths train_set_paths.pkl --val_paths val_set_paths.pkl --test_paths test_set_paths.pkl
```
To perform hyperparameter tuning on Random Forest, add `--tune_rf`.

## Results & Visualization
The repository includes visualization tools for latent space analysis:
- **2D Projection of Latent Space:** PCA, t-SNE, UMAP
- **Classification Performance:** ROC curves, PR curves, confusion matrices

## Configuration
Modify `config.yaml` to adjust parameters such as dataset paths, batch sizes, and training ratios:
```yaml
project_dir: "/path/to/project"
csv_filename: "DataBaseSubjects.csv"
batch_size: 32
train_ratio: 0.7
val_ratio: 0.15
test_ratio: 0.15
```

## Acknowledgments
This project was developed as part of research in computational neuroscience, focusing on Alzheimer's disease detection using deep learning techniques.

## License
MIT License



# unsam_proj
![image](https://github.com/user-attachments/assets/c4375a36-24e1-442f-83c4-ed698f6b025e)
