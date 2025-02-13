#!/usr/bin/env python3
"""
classification.py

This script:
    1. Loads a trained BetaVAE model,
    2. Extracts latent representations (mu) for AD and CN subjects only,
    3. Trains multiple classification models (Logistic Regression, SVM, Random Forest, etc.),
    4. Evaluates models on a held-out test set,
    5. Plots individual and combined ROC curves, PR curves, confusion matrices,
       and a comparative bar plot for the performance metrics,
    6. (Optional) Performs a hyperparameter search (GridSearchCV) on Random Forest.

Usage:
    python classification.py \
        --project_dir /path/to/your/project \
        --beta 0.5 \
        --latent_dim 250 \
        --train_paths /path/to/train_set_paths.pkl \
        --val_paths /path/to/val_set_paths.pkl \
        --test_paths /path/to/test_set_paths.pkl \
        [--tune_rf]

Note: The script expects three pickle files containing lists of file paths:
      1) train_set_paths.pkl
      2) val_set_paths.pkl
      3) test_set_paths.pkl

      Each .pt file is assumed to be named as "AD_tensor_1234.pt" or "CN_tensor_5678.pt".
      Any files not matching "AD" or "CN" as a prefix will be skipped.
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
    roc_auc_score, confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)
from sklearn.model_selection import GridSearchCV

# Matplotlib / Seaborn for plots
import matplotlib.pyplot as plt
import seaborn as sns

from models.vae import BetaVAE

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# -----------------------------
# Utility Functions
# -----------------------------
def normalize_tensor(tensor: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Min-Max normalize each channel to [0, 1]. 
    Adjust if you used a different normalization in your pipeline.
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


def extract_latents_and_labels(
    file_paths: List[str], model: BetaVAE, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Iterate over a list of .pt file paths, load each subject, skip any not 'AD' or 'CN',
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
            # Skip any file that isn't strictly "AD" or "CN" at the prefix
            continue

        label = 1 if group == 'AD' else 0
        data = torch.load(fp, map_location='cpu')
        # If data is loaded as np.ndarray, convert to torch.Tensor
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        # Normalize (if needed in your pipeline)
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
    Plot the ROC curve given true labels and predicted probabilities for the positive class (AD=1).
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title(f'ROC Curve - {classifier_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix(y_true, y_pred, classifier_name: str):
    """
    Plot a confusion matrix using seaborn heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["CN", "AD"], yticklabels=["CN", "AD"])
    plt.title(f'Confusion Matrix - {classifier_name}')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()


def plot_pr_curve(y_true: np.ndarray, y_proba: np.ndarray, classifier_name: str):
    """
    Plot the Precision-Recall (PR) curve given true labels and predicted probabilities.
    Useful in situations with class imbalance.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, color='green', lw=2, label='PR curve')
    plt.title(f'Precision-Recall Curve - {classifier_name}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_combined_roc_curves(classifier_probas: dict, y_true: np.ndarray):
    """
    Plot ROC curves for all classifiers in a single figure.
    classifier_probas: dict { "Classifier Name": y_proba (array) }
    y_true: ground truth labels
    """
    plt.figure(figsize=(7, 6))
    for clf_name, proba in classifier_probas.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{clf_name} (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title("Combined ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_combined_pr_curves(classifier_probas: dict, y_true: np.ndarray):
    """
    Plot PR curves for all classifiers in a single figure.
    classifier_probas: dict { "Classifier Name": y_proba (array) }
    y_true: ground truth labels
    """
    plt.figure(figsize=(7, 6))
    for clf_name, proba in classifier_probas.items():
        precision, recall, _ = precision_recall_curve(y_true, proba)
        plt.plot(recall, precision, lw=2, label=f'{clf_name}')
    plt.title("Combined Precision-Recall Curves")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()


def plot_performance_summary(performance_metrics: dict):
    """
    Creates a comparative bar plot for Accuracy, Precision, Recall, F1-score, and AUC 
    for each classifier in performance_metrics.

    performance_metrics structure:
        {
          "ClassifierName": {
              "Accuracy": float,
              "Precision": float,
              "Recall": float,
              "F1-Score": float,
              "ROC AUC": float
          },
          ...
        }
    """
    # Convert performance metrics into a suitable structure for plotting
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC AUC"]
    classifiers = list(performance_metrics.keys())

    # Prepare a 2D list: row -> metric, column -> classifier
    data = []
    for metric in metrics:
        row = []
        for clf_name in classifiers:
            value = performance_metrics[clf_name].get(metric, None)
            row.append(value if value is not None else 0.0)
        data.append(row)

    # data is now shape (num_metrics x num_classifiers)

    x = np.arange(len(classifiers))  # classifier indices
    bar_width = 0.15
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot each metric as a separate group
    for i, metric in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * bar_width
        ax.bar(
            x + offset, data[i], bar_width, label=metric
        )

    ax.set_xticks(x)
    ax.set_xticklabels(classifiers, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Classifier Performance Comparison")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(metrics))
    plt.tight_layout()
    plt.show()


# -----------------------------
# Main Script
# -----------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1) Load BetaVAE model
    model_path = os.path.join(args.project_dir, "best_beta_vae_model.pth")
    vae = BetaVAE(latent_dim=args.latent_dim, beta=args.beta).to(device)
    try:
        vae.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"Loaded trained BetaVAE from {model_path}")
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        sys.exit(1)

    # 2) Load file paths for train, val, test
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

    # 3) Extract latents for train+val
    X_train, y_train = extract_latents_and_labels(train_val_set_paths, vae, device)
    # 4) Extract latents for test
    X_test, y_test = extract_latents_and_labels(test_set_paths, vae, device)

    logging.info(f"Train+Val data shape: {X_train.shape}, Train+Val labels shape: {y_train.shape}")
    logging.info(f"Test data shape: {X_test.shape}, Test labels shape: {y_test.shape}")
    logging.info(f"Train+Val label distribution: {Counter(y_train)}")
    logging.info(f"Test label distribution: {Counter(y_test)}")

    if X_train.size == 0 or X_test.size == 0:
        logging.error("No valid AD/CN data found. Exiting.")
        sys.exit(1)

    # 5) Scale data
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
    classifier_probas = {}  # For combined ROC/PR

    # 6) Train and evaluate each classifier
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # If predict_proba exists, store probabilities for the positive class (AD=1)
        if hasattr(clf, "predict_proba"):
            y_proba = clf.predict_proba(X_test)[:, 1]
        else:
            # For classifiers that don't have predict_proba, 
            # we can approximate using decision_function, but here SVC(probability=True) is set, so it's covered.
            y_proba = None

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

        # Print classification report
        print(f"Classifier: {name}")
        print(classification_report(y_test, y_pred, target_names=["CN", "AD"]))
        print("-" * 50)

        # Individual ROC (if we have probabilities)
        if y_proba is not None:
            plot_roc_curve(y_test, y_proba, classifier_name=name)
            plot_pr_curve(y_test, y_proba, classifier_name=name)
            classifier_probas[name] = y_proba

        # Confusion Matrix
        plot_confusion_matrix(y_test, y_pred, classifier_name=name)

    # Print performance metrics in summary form
    print("\n=== Performance Summary ===")
    for name, metrics in performance_metrics.items():
        print(f"Classifier: {name}")
        for metric, value in metrics.items():
            if value is not None:
                print(f"  {metric}: {value:.3f}")
            else:
                print(f"  {metric}: None")
        print("-" * 40)

    # 7) Plot combined ROC and PR curves (only if at least 2 classifiers have predict_proba)
    if len(classifier_probas) > 1:
        plot_combined_roc_curves(classifier_probas, y_test)
        plot_combined_pr_curves(classifier_probas, y_test)

    # 8) Plot a comparative bar chart of performance metrics
    plot_performance_summary(performance_metrics)

    # 9) Optional: Hyperparameter tuning on Random Forest
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

