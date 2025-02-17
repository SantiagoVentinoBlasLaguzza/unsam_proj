#!/usr/bin/env python3
"""
Training script for BetaVAE on Alzheimer’s fMRI tensor data.

This script:
    - Loads training, validation (and optionally test) datasets (preprocessed tensor files),
    - Defines a BetaVAE model with a dynamic decoder output size,
    - Trains the model with early stopping and learning rate scheduling,
    - Logs and plots training progress (similar to training.ipynb),
    - Prints or logs intermediate results during training,
    - Displays dataset sizes and suggestions to train faster.

Usage:
    python train.py --epochs 1000 --latent_dim 250 --beta 0.5
"""

import os
import sys
import random
import argparse
import logging
import math
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import seaborn as sns

from models.vae import BetaVAE
from utils.utils import zero_diagonals

# If using in Google Colab, try to mount drive.
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Optional: to show a progress bar in the terminal
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

# ---------------------
# Global Configurations
# ---------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# For reproducibility (optional):
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Optional: Enable cuDNN benchmarking for possibly faster training
# (best if input sizes don't change frequently)
torch.backends.cudnn.benchmark = True

# ---------------------
# Loss Function
# ---------------------
def loss_function(
    recon_x: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float
) -> Tuple[torch.Tensor, float, float]:
    """
    VAE loss = reconstruction loss + beta * KL divergence.
    """
    if recon_x.shape != x.shape:
        raise RuntimeError(f"Size mismatch: recon_x {recon_x.shape} vs x {x.shape}")

    recon_loss = F.mse_loss(recon_x, x, reduction='mean')
    # Alternatively: F.binary_cross_entropy(recon_x, x, reduction='mean') if your data is in [0,1].
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    total_loss = recon_loss + beta * kld
    return total_loss, recon_loss.item(), kld.item()

class CustomTensorDataset(Dataset):
    """
    Example custom dataset if you'd prefer loading individual .pt files per subject.
    """
    def __init__(self, file_paths: List[str], log_transform_channel: int = 2) -> None:
        self.file_paths = file_paths
        self.log_transform_channel = log_transform_channel

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        fp = self.file_paths[idx]
        data = torch.load(fp, map_location='cpu')
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        # e.g., zero out diagonals
        data = zero_diagonals(data)
        # e.g., log transform channel
        if data.dim() == 3 and data.shape[0] > self.log_transform_channel:
            data[self.log_transform_channel] = torch.log(data[self.log_transform_channel] + 1e-8)
        return data

# ---------------------------------------------------------
# Training and Evaluation Utilities
# ---------------------------------------------------------
def train_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    beta: float,
    print_frequency: int = 50
) -> Tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    data_size = 0

    if TQDM_AVAILABLE:
        loader = tqdm(data_loader, desc="Training", leave=False)
    else:
        loader = data_loader

    for batch_idx, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        x = x.to(device, dtype=torch.float)
        data_size += x.size(0)

        optimizer.zero_grad()
        recon_x, mu, logvar, _ = model(x)
        loss, recon_loss, kld_loss = loss_function(recon_x, x, mu, logvar, beta)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_recon += recon_loss * x.size(0)
        total_kld += kld_loss * x.size(0)

        # Optionally print intermediate results
        if print_frequency > 0 and (batch_idx + 1) % print_frequency == 0:
            logging.info(f"[Batch {batch_idx+1}] Loss: {loss.item():.6f} "
                         f"Recon: {recon_loss:.6f}, KLD: {kld_loss:.6f}")

    avg_loss = total_loss / data_size
    avg_recon = total_recon / data_size
    avg_kld = total_kld / data_size
    return avg_loss, avg_recon, avg_kld


def evaluate_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    beta: float
) -> Tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kld = 0.0
    data_size = 0

    with torch.no_grad():
        for batch in data_loader:
            x = batch[0] if isinstance(batch, (list, tuple)) else batch
            x = x.to(device, dtype=torch.float)
            data_size += x.size(0)

            recon_x, mu, logvar, _ = model(x)
            loss, recon_loss, kld_loss = loss_function(recon_x, x, mu, logvar, beta)

            total_loss += loss.item() * x.size(0)
            total_recon += recon_loss * x.size(0)
            total_kld += kld_loss * x.size(0)

    avg_loss = total_loss / data_size
    avg_recon = total_recon / data_size
    avg_kld = total_kld / data_size
    return avg_loss, avg_recon, avg_kld


def plot_losses(train_losses: List[float], val_losses: List[float]) -> None:
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

# ---------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------
def main(args):
    logging.info("=====================================")
    logging.info("Starting BetaVAE Training Script")
    logging.info("=====================================")

    # Some suggestions to train faster:
    # 1) Try a larger batch size if you have enough GPU RAM.
    # 2) Use mixed precision (torch.cuda.amp) if your data can handle float16.
    # 3) Keep torch.backends.cudnn.benchmark = True if input size doesn't change.
    # 4) Possibly reduce the model size.

    if IN_COLAB:
        if args.force_remount:
            drive.mount('/content/drive', force_remount=True)
        else:
            drive.mount('/content/drive')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    project_dir = args.project_dir
    model_save_path = os.path.join(project_dir, 'best_beta_vae_model.pth')

    # Load training & validation data
    try:
        train_data = torch.load(os.path.join(project_dir, 'train_data.pt'), map_location='cpu')
        val_data = torch.load(os.path.join(project_dir, 'val_data.pt'), map_location='cpu')
        logging.info("Successfully loaded training and validation data.")
    except Exception as e:
        logging.error(f"Error loading dataset tensors: {e}")
        sys.exit(1)

    # Attempt loading test data if available (optional)
    test_data_path = os.path.join(project_dir, 'test_data.pt')
    test_data = None
    if os.path.exists(test_data_path):
        try:
            test_data = torch.load(test_data_path, map_location='cpu')
            logging.info("Successfully loaded test data.")
        except Exception as e:
            logging.warning(f"Could not load test_data.pt: {e}")

    # Create Datasets
    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)
    if test_data is not None:
        test_dataset = TensorDataset(test_data)
    else:
        test_dataset = None

    # Print dataset sizes
    logging.info(f"Train subjects: {len(train_dataset)}")
    logging.info(f"Validation subjects: {len(val_dataset)}")
    if test_dataset:
        logging.info(f"Test subjects: {len(test_dataset)}")
    else:
        logging.info("No test set found.")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # (Optional) create test_loader if test dataset is available
    if test_dataset:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True
        )
    else:
        test_loader = None

    # Initialize model
    model = BetaVAE(
        latent_dim=args.latent_dim,
        hidden_dim=1024*5,  # or pass custom
        beta=args.beta,
        dropout_rate=0.01,
        input_channels=3  # Adjust if your data has different channels
    ).to(device)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )

    best_val_loss = float('inf')
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    logging.info("Starting training loop...")
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss, train_recon, train_kld = train_epoch(
            model, train_loader, optimizer, device, args.beta,
            print_frequency=50  # print results every 50 batches
        )
        # Validate
        val_loss, val_recon, val_kld = evaluate_epoch(model, val_loader, device, args.beta)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.6f} (Recon {train_recon:.6f}, KLD {train_kld:.6f}) "
            f"| Val Loss: {val_loss:.6f} (Recon {val_recon:.6f}, KLD {val_kld:.6f})"
        )

        # Step scheduler with val_loss
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), model_save_path)
            logging.info(f"Validation loss improved. Model saved to {model_save_path}")
        else:
            epochs_no_improve += 1
            logging.info(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= args.early_stop:
                logging.info("Early stopping triggered.")
                break

    # Plot losses
    logging.info("Training complete. Plotting loss curves...")
    plot_losses(train_losses, val_losses)

    # Optional: Evaluate on test if available
    if test_loader:
        logging.info("Evaluating on test set...")
        test_loss, test_recon, test_kld = evaluate_epoch(model, test_loader, device, args.beta)
        logging.info(
            f"Test Loss: {test_loss:.6f} (Recon {test_recon:.6f}, KLD {test_kld:.6f})"
        )

    logging.info("=====================================")
    logging.info("BetaVAE Training Script Finished")
    logging.info("=====================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train BetaVAE for Alzheimer tensor data")
    parser.add_argument("--project_dir", type=str, default="/content/drive/My Drive/UNSAM_alzheimer",
                        help="Project directory path")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32*2, help="Batch size for training")
    parser.add_argument("--latent_dim", type=int, default=250, help="Latent dimension size")
    parser.add_argument("--beta", type=float, default=3.5, help="Beta coefficient for KL divergence")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-7, help="Weight decay for optimizer")
    parser.add_argument("--early_stop", type=int, default=50, help="Early stopping patience")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads for DataLoader")
    parser.add_argument("--force_remount", action='store_true', help="Force remounting Google Drive (Colab only)")
    args = parser.parse_args()

    main(args)
