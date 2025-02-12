#!/usr/bin/env python3
"""
latent_space.py

This script:
    1. Loads the trained BetaVAE model,
    2. Loads the train, validation, and test datasets,
    3. Encodes each sample to obtain its latent embedding,
    4. Visualizes a random subject's original 3-channel tensor and its reconstruction (from each partition),
    5. Projects latent space via PCA, t-SNE, and UMAP,
    6. Plots the 2D projections with color-coded groups (AD, CN, Others)
       and marker shapes for train, validation, test data.

Usage:
    python latent_space.py --project_dir /path/to/project --latent_dim 250 --beta 0.5
"""

import os
import sys
import logging
import argparse
import random
from typing import Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ----------------------
# Optional: If BetaVAE is in another file, adjust the import accordingly.
# Here, we define BetaVAE inline for completeness.
# ----------------------
class BetaVAE(nn.Module):
    def __init__(self, latent_dim=250, hidden_dim=1024, beta=0.1, dropout_rate=0.01):
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

        # After 4 downsamples, for 112x112-ish inputs -> ~7x7
        self.flatten_size = 256 * 7 * 7

        self.fc1 = nn.Linear(self.flatten_size, self.hidden_dim)
        self.fc_bn1 = nn.BatchNorm1d(self.hidden_dim)
        self.fc_mu = nn.Linear(self.hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dim, latent_dim)

        # Decoder
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

    def decode(self, z: torch.Tensor, target_size: Tuple[int, int] = (116, 116)) -> torch.Tensor:
        x = torch.tanh(self.fc_bn2(self.fc_decode(z)))
        x = torch.tanh(self.fc_decode2(x))
        x = x.view(z.size(0), 256, 7, 7)
        x = torch.tanh(self.dec_bn1(self.dec_conv1(x)))
        x = torch.tanh(self.dec_bn2(self.dec_conv2(x)))
        x = torch.tanh(self.dec_bn3(self.dec_conv3(x)))
        x = torch.sigmoid(self.dec_conv4(x))
        x = F.interpolate(x, size=target_size, mode='bicubic', align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        target_size = (x.size(-2), x.size(-1))  # match input spatial dims
        recon_x = self.decode(z, target_size=target_size)
        return recon_x, mu, logvar, z


def zero_diagonals(tensor: torch.Tensor) -> torch.Tensor:
    """
    Utility to zero out diagonals in each channel if needed.
    """
    if tensor.size(1) == tensor.size(2):
        idx = torch.arange(tensor.size(1), device=tensor.device)
        tensor[:, idx, idx] = 0
    return tensor


def load_partition_data(project_dir: str, partition_name: str) -> torch.Tensor:
    """
    Loads a .pt file containing the partition data (train, val, or test).
    """
    file_path = os.path.join(project_dir, f"{partition_name}_data.pt")
    logging.info(f"Loading {partition_name} data from {file_path} ...")
    try:
        data = torch.load(file_path, map_location='cpu')
    except Exception as e:
        logging.error(f"Error loading {partition_name} data: {e}")
        sys.exit(1)
    return data


def random_subject_visualization(
    model: nn.Module,
    data: torch.Tensor,
    partition_name: str,
    device: torch.device
) -> None:
    """
    Pick a random subject from the given partition (train, val, or test),
    plot the 3-channel input and its reconstruction.
    """
    if data is None or data.size(0) == 0:
        logging.warning(f"No data found in {partition_name} partition.")
        return

    model.eval()
    idx = random.randint(0, data.size(0) - 1)
    sample = data[idx:idx+1].to(device, dtype=torch.float)  # shape (1, 3, H, W)
    with torch.no_grad():
        recon, _, _, _ = model(sample)

    # Move to CPU and numpy for plotting
    sample_np = sample.cpu().numpy()[0]   # shape (3, H, W)
    recon_np = recon.cpu().numpy()[0]     # shape (3, H, W)

    # Plot
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for j in range(3):
        axes[0, j].imshow(sample_np[j], cmap='viridis')
        axes[0, j].set_title(f'{partition_name} Original - Matrix {j+1}')
        axes[0, j].axis('off')

        axes[1, j].imshow(recon_np[j], cmap='viridis')
        axes[1, j].set_title(f'{partition_name} Reconstructed - Matrix {j+1}')
        axes[1, j].axis('off')
    plt.tight_layout()
    plt.show()


def get_latent_embeddings(
    model: nn.Module,
    data: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Pass the entire dataset through the encoder, returning mu (or z).
    Return shape: (N, latent_dim).
    """
    model.eval()
    batch_size = 32
    embeddings = []
    for start in range(0, data.size(0), batch_size):
        end = start + batch_size
        batch = data[start:end].to(device, dtype=torch.float)
        with torch.no_grad():
            mu, logvar = model.encode(batch)
            # Option 1: Use mu as embedding
            # Option 2: Reparameterize to get z
            # z = model.reparameterize(mu, logvar)
        embeddings.append(mu.cpu())  # shape (batch_size, latent_dim)
    return torch.cat(embeddings, dim=0)


def plot_2d_projection(
    embedding_2d: np.ndarray,
    groups: List[str],
    partitions: List[str],
    title: str
):
    """
    Plot the 2D embedding (PCA, t-SNE, or UMAP) with color by group and marker by partition.
    embedding_2d: (N, 2) array
    groups: list of group labels ('AD', 'CN', 'Other') per point
    partitions: list of partition labels ('train', 'val', 'test') per point
    """
    plt.figure(figsize=(8, 6))

    # Define color + marker mapping
    color_map = {'AD': 'red', 'CN': 'blue', 'Other': 'green'}
    marker_map = {'train': 'o', 'val': 's', 'test': '^'}

    for group_label, partition_label, (x, y) in zip(groups, partitions, embedding_2d):
        c = color_map.get(group_label, 'gray')
        m = marker_map.get(partition_label, 'x')
        plt.scatter(x, y, c=c, marker=m, alpha=0.7, edgecolors='k', linewidths=0.5)

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(True)
    plt.show()


def main(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    #project_dir = args.project_dir
    project_dir = "/content/drive/My Drive/UNSAM_alzheimer/"
    model_path = os.path.join(project_dir, "best_beta_vae_model.pth")

    # 1) Load the BetaVAE model
    model = BetaVAE(latent_dim=args.latent_dim, beta=args.beta)
    logging.info(f"Loading model from {model_path}")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        logging.info("Model loaded successfully.")
    except Exception as e:
        logging.error(f"Error loading BetaVAE model: {e}")
        sys.exit(1)

    # 2) Load train, val, test data
    train_data = load_partition_data(project_dir, "train")
    val_data = load_partition_data(project_dir, "val")
    test_data = load_partition_data(project_dir, "test")

    # Optionally, zero out diagonals if you did that in preprocessing
    # train_data = zero_diagonals(train_data)
    # val_data   = zero_diagonals(val_data)
    # test_data  = zero_diagonals(test_data)

    # 3) Visual Comparison for one subject in each partition
    random_subject_visualization(model, train_data, "Train", device)
    random_subject_visualization(model, val_data, "Validation", device)
    random_subject_visualization(model, test_data, "Test", device)

    # 4) Obtain latent embeddings (mu or z) for all data
    #    Suppose we also have group labels for each row, e.g. from a CSV or saved list.
    #    Below we create dummy group labels for illustration.
    #    Real code: load your actual group labels here.
    train_size = train_data.size(0)
    val_size = val_data.size(0)
    test_size = test_data.size(0)

    # Example: label vectors
    # Suppose AD=0, CN=1, Other=2 in your data. Convert to strings for plotting:
    # Or read from a stored array if you have them. Must match data ordering.
    # For demonstration, we'll assign random groups:
    group_labels_train = [random.choice(['AD', 'CN', 'Other']) for _ in range(train_size)]
    group_labels_val = [random.choice(['AD', 'CN', 'Other']) for _ in range(val_size)]
    group_labels_test = [random.choice(['AD', 'CN', 'Other']) for _ in range(test_size)]

    partition_labels_train = ["train"] * train_size
    partition_labels_val = ["val"] * val_size
    partition_labels_test = ["test"] * test_size

    # Concatenate all data to embed them together
    all_data = torch.cat([train_data, val_data, test_data], dim=0)
    all_partition_labels = partition_labels_train + partition_labels_val + partition_labels_test
    all_group_labels = group_labels_train + group_labels_val + group_labels_test

    # Encode to get embeddings
    all_embeddings = get_latent_embeddings(model, all_data, device)
    # all_embeddings shape: (N, latent_dim)

    # 5) Dimensionality reductions: PCA, t-SNE, UMAP
    all_embeddings_np = all_embeddings.numpy()

    # --- PCA ---
    pca = PCA(n_components=2, random_state=42)
    pca_2d = pca.fit_transform(all_embeddings_np)
    plot_2d_projection(pca_2d, all_group_labels, all_partition_labels, title="PCA Projection")

    # --- t-SNE ---
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    tsne_2d = tsne.fit_transform(all_embeddings_np)
    plot_2d_projection(tsne_2d, all_group_labels, all_partition_labels, title="t-SNE Projection")

    # --- UMAP ---
    reducer = umap.UMAP(n_components=2, random_state=42)
    umap_2d = reducer.fit_transform(all_embeddings_np)
    plot_2d_projection(umap_2d, all_group_labels, all_partition_labels, title="UMAP Projection")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and visualize BetaVAE latent space.")
    parser.add_argument("--project_dir", type=str, default="./", help="Project directory path")
    parser.add_argument("--latent_dim", type=int, default=250, help="Latent dimension size (must match trained model)")
    parser.add_argument("--beta", type=float, default=0.1, help="Beta coefficient for VAE loss (must match trained model)")
    args = parser.parse_args()

    main(args)
