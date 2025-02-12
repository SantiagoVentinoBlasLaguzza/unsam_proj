#!/usr/bin/env python3
"""
classification.py

This script:
    1. Loads a trained BetaVAE model,
    2. Extracts latent representations (mu) for AD and CN subjects only,
    3. Trains multiple classification models (Logistic Regression, SVM, Random Forest, etc.),
    4. Evaluates models on a held-out test set,
    5. Plots ROC curves and confusion matrices for selected models,
    6. (Optional) Performs a hyperparameter search (GridSearchCV) on Random Forest.

Usage:
    python classification.py \
        --project_dir /path/to/your/project \
        --beta 0.5 \
        --latent_dim 250 \
        --train_paths /path/to/train_set_paths.pkl \
        --val_paths /path/to/val_set_paths.pkl \
        --test_paths /path/to/test_set_paths.pkl

Note: The script expects three pickle files containing lists of file paths:
      1) train_set_paths.pkl
      2) val_set_paths.pkl
      3) test_set_paths.pkl
      Each .pt file is assumed to be named as "AD_tensor_1234.pt" or "CN_tensor_5678.pt" etc.
"""

import os
import sys
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple
from collections import Counter

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
)
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------
# Define BetaVAE (must match your trained architecture)
# -----------------------------
class BetaVAE(nn.Module):
    def __init__(self, latent_dim: int = 250, hidden_dim: int = 1024*5, beta: float = 0.1, dropout_rate: float = 0.01):
        super(BetaVAE, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.beta = beta

        # Encoder
        self.enc_conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.enc_bn1 = nn.BatchNorm2d(32)
        self.enc_conv2 = nn.Conv2d(32, 64, 4, 2, 1)
        self.enc_bn2 = nn.BatchNorm2d(64)
        self.enc_conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        self.enc_bn3 = nn.BatchNorm2d(128)
        self.enc_conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        self.enc_bn4 = nn.BatchNorm2d(256)

        # After 4 convs (stride=2), for ~112x112 -> ~7x7
        self.flatten_size = 256 * 7 * 7

        # FC layers for mu and logvar
        self.fc1 = nn.Linear(self.flatten_size, self.hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder (not strictly needed for classification, but the model must match)
        self.fc_decode = nn.Linear(latent_dim, self.hidden_dim)
        self.fc_bn2 = nn.BatchNorm1d(self.hidden_dim)
        self.fc_decode2 = nn.Linear(self.hidden_dim, self.flatten_size)
        self.dec_conv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec_bn1 = nn.BatchNorm2d(128)
        self.dec_conv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec_bn2 = nn.BatchNorm2d(64)
        self.dec_conv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dec_bn3 = nn.BatchNorm2d(32)
        self.dec_conv4 = nn.ConvTranspose2d(32, 3, 4, 2, 1)
        self.dropout = nn.Dropout(dropout_rate)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tanh(self.enc_bn1(self.enc_conv1(x)))
        x = torch.tanh(self.enc_bn2(self.enc_conv2(x)))
        x = torch.tanh(self.enc_bn3(self.enc_conv3(x)))
        x = torch.tanh(self.enc_bn4(self.enc_conv4(x)))
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc_bn1(self.fc1(x)))
        x = self.dropout(x)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor):
        # Not strictly needed here, but included for completeness
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return mu, logvar, z


# -----------------------------
# Utility Functions
# -----------------------------
def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-Max normalize each channel to [0, 1]. Adjust if you used a different normalization method.
    """
    c, h, w = tensor.shape
    out = tensor.clone()
    for i in range(c):
        channel = out[i]
        ch_min = channel.min()
        ch_max = channel.max()
        denom = (ch_max - ch_min) + eps
        out[i] = (channel - ch_min) / denom
    return out


def extract_latents_and_labels(file_paths: List[str], model: BetaVAE, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate over a list of .pt file paths, load each subject, skip 'Others',
    compute mu from the encoder, and collect (latents, label).
    Label: AD=1, CN=0.
    """
    latents = []
    labels = []
    model.eval()

    for fp in file_paths:
        filename = os.path.basename(fp)
        group = filename.split('_')[0]  # e.g., "AD", "CN"
        if group not in ['AD', 'CN']:
            # Skip "Others" category if present
            continue

        label = 1 if group == 'AD' else 0
        data = torch.load(fp, map_location='cpu')
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        # Normalize if you used that in your pipeline
        data = normalize_tensor(data)

        # shape is (3, H, W), add batch dimension -> (1, 3, H, W)
        x = data.unsqueeze(0).to(device, dtype=torch.float)
        mu, logvar = model.encode(x)
        # Using mu as latent representation
        mu_np = mu.cpu().detach().numpy().squeeze()  # shape (latent_dim,)
        latents.append(mu_np)
        labels.append(label)

    return np.array(latents), np.array(labels)


def plot_roc_curve(y_true, y_proba, classifier_name: str):
    """
    Plot the ROC curve given true labels and predicted probabilities for the positive class.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classifier_name: str):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# -----------------------------
# Main Script
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load BetaVAE model
    model_path = os.path.join(args.project_dir, "best_beta_vae_model.pth")
    vae = BetaVAE(latent_dim=args.latent_dim, beta=args.beta).to(device)
    try:
        vae.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded trained BetaVAE from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Load file paths for train, val, test
    try:
        with open(args.train_paths, 'rb') as f:
            train_set_paths = pickle.load(f)
        with open(args.val_paths, 'rb') as f:
            val_set_paths = pickle.load(f)
        with open(args.test_paths, 'rb') as f:
            test_set_paths = pickle.load(f)
    except Exception as e:
        logging.error(f"Error loading path lists: {e}")
        sys.exit(1)

    # Combine train + val for training classifiers
    train_val_set_paths = train_set_paths + val_set_paths

    # 1) Extract latents for train+val
    X_train, y_train = extract_latents_and_labels(train_val_set_paths, vae, device)
    # 2) Extract latents for test
    X_test, y_test = extract_latents_and_labels(test_set_paths, vae, device)

    logging.info(f"Train+Val data shape: {X_train.shape}, Train+Val labels shape: {y_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")

    logging.info(f"Train+Val label distribution: {Counter(y_train)}")
    logging.info(f"Test label distribution: {Counter(y_test)}")

    if X_train.size == 0 or X_test.size == 0:
        logging.error("No valid AD/CN data found. Exiting.")
        sys.exit(1)

    # Scale data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Define classifiers
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM (Linear Kernel)": SVC(kernel='linear', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "k-NN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100),
        "Neural Network": MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000),
    }

    performance_metrics = {}

    # Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        performance_metrics[name] = {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1,
            "ROC AUC": roc_auc,
        }

        print(f"Classifier: {name}")
        print(classification_report(y_test, y_pred, target_names=["CN", "AD"]))
        print("-" * 50)

        # Plot ROC curve for demonstration (only if we have proba)
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, classifier_name=name)

        # Plot confusion matrix
        plot_confusion_matrix(y_test, y_pred, classifier_name=name)

    # Print performance metrics in a summary
    print("\n=== Performance Summary ===")
    for name, metrics in performance_metrics.items():
        print(f"Classifier: {name}")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.3f}" if value is not None else f"  {metric}: None")
        print("-" * 40)

    # Optional: Hyperparameter tuning on Random Forest
    if args.tune_rf:
        logging.info("Performing GridSearchCV on Random Forest...")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
        }
        grid_search = GridSearchCV(
            estimator=RandomForestClassifier(),
            param_grid=param_grid,
            cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        print(f"Best RandomForest params: {grid_search.best_params_}")

        best_rf = grid_search.best_estimator_
        y_pred = best_rf.predict(X_test)
        print("Evaluation of best RF model:")
        print(classification_report(y_test, y_pred, target_names=["CN", "AD"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate classifiers on latent VAE representations.")
    parser.add_argument("--project_dir", type=str, default=".", help="Directory containing the best_beta_vae_model.pth")
    parser.add_argument("--train_paths", type=str, required=True, help="Pickle file with train set paths")
    parser.add_argument("--val_paths", type=str, required=True, help="Pickle file with validation set paths")
    parser.add_argument("--test_paths", type=str, required=True, help="Pickle file with test set paths")
    parser.add_argument("--latent_dim", type=int, default=250, help="Latent dimension used in the trained BetaVAE")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta value used in the trained BetaVAE")
    parser.add_argument("--tune_rf", action='store_true', help="Perform GridSearchCV on Random Forest")
    args = parser.parse_args()

    main(args)
