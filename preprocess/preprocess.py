#!/usr/bin/env python3
"""
Preprocess module for tensor generation and dataset splitting for Alzheimer’s fMRI analysis.
This script loads subject metadata and associated tensor files,
applies normalization and other preprocessing functions, splits the dataset,
and saves train/val/test sets and related metadata.

Authors: Santiago V. Blas L.
Date: [2025-02-05]

Usage:
    python preprocess.py
"""

import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
import pickle
import logging
import argparse
from typing import Tuple, List, Optional, Dict, Any

# For Google Colab drive mounting
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Configure logging: INFO level with timestamp.
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def mount_drive_if_needed() -> None:
    """
    Mount Google Drive if running in Google Colab.
    """
    if IN_COLAB:
        drive_root = '/content/drive/My Drive'
        if not os.path.exists(drive_root):
            drive.mount('/content/drive')
        else:
            logging.info("Google Drive already mounted.")
    else:
        logging.info("Not running in Colab; skipping drive mount.")


def get_project_directory() -> str:
    """
    Determine the project directory based on the environment (Colab vs local).
    
    Returns:
        project_dir (str): Path to the project directory.
    """
    if IN_COLAB:
        # Adjust as needed for your Colab folder structure
        project_dir = '/content/drive/My Drive/UNSAM_alzheimer'
    else:
        project_dir = '/home/santiago/Desktop/alzheimer'
    logging.info(f"Project directory set to: {project_dir}")
    return project_dir


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML configuration file.
    
    Args:
        config_path (str): Path to the YAML file.
        
    Returns:
        config (dict): Configuration dictionary.
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        sys.exit(1)


def set_random_seed(seed: int) -> None:
    """
    Set the seed for reproducibility in numpy and torch.
    
    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f"Random seed set to: {seed}")


def load_subjects_dataframe(csv_path: str) -> pd.DataFrame:
    """
    Load subject metadata from a CSV file.
    
    Args:
        csv_path (str): Path to the CSV file.
        
    Returns:
        subjects_df (pd.DataFrame): Loaded dataframe.
    """
    try:
        subjects_df = pd.read_csv(csv_path)
        logging.info(f"Loaded {len(subjects_df)} subjects from {csv_path}")
        return subjects_df
    except Exception as e:
        logging.error(f"Error loading CSV file at {csv_path}: {e}")
        sys.exit(1)


def log_median_age(subjects_df: pd.DataFrame) -> None:
    """
    Compute and log the median age from the subjects dataframe.
    
    Args:
        subjects_df (pd.DataFrame): DataFrame containing subject metadata.
    """
    if 'Age' in subjects_df.columns:
        median_age = subjects_df['Age'].median()
        logging.info(f"Median Age: {median_age}")
    else:
        logging.warning("Column 'Age' not found in subjects dataframe.")


def split_data(subject_list: List[Any],
               train_ratio: float = 0.7,
               val_ratio: float = 0.15,
               test_ratio: float = 0.15) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Split a list of subjects (or file paths) into train, validation, and test sets.
    
    Args:
        subject_list (List[Any]): List of subject identifiers or file paths.
        train_ratio (float): Proportion for training set.
        val_ratio (float): Proportion for validation set.
        test_ratio (float): Proportion for test set.
        
    Returns:
        Tuple[List[Any], List[Any], List[Any]]: train_set, val_set, test_set.
    """
    subjects = subject_list.copy()
    np.random.shuffle(subjects)
    n_total = len(subjects)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_set = subjects[:n_train]
    val_set = subjects[n_train:n_train + n_val]
    test_set = subjects[n_train + n_val:]
    return train_set, val_set, test_set


def group_subjects_by_attributes(subjects_df: pd.DataFrame,
                                 group_cols: List[str],
                                 id_col: str) -> Dict[str, List[Any]]:
    """
    Group subjects by given columns and return a dictionary mapping group keys to subject IDs.
    
    Args:
        subjects_df (pd.DataFrame): DataFrame with subject metadata.
        group_cols (List[str]): Columns to group by (e.g. ['ResearchGroup', 'Sex']).
        id_col (str): Column containing the subject identifier.
        
    Returns:
        groups (Dict[str, List[Any]]): Dictionary mapping group names to list of subject IDs.
    """
    try:
        grouped = subjects_df.groupby(group_cols)[id_col].apply(list).to_dict()
        logging.info(f"Grouped subjects into {len(grouped)} groups based on {group_cols}")
        return grouped
    except Exception as e:
        logging.error(f"Error grouping subjects: {e}")
        return {}


def build_tensor_paths(tensor_dir: str,
                       grouped_subjects: Dict[tuple, List[Any]]) -> Dict[str, List[str]]:
    """
    Build file paths for tensor files for each group.
    
    Args:
        tensor_dir (str): Directory where tensor files are stored.
        grouped_subjects (Dict[tuple, List[Any]]): Grouped subject IDs with keys as tuples.
        
    Returns:
        tensor_groups (Dict[str, List[str]]): Dictionary with keys as joined group names and values as file paths.
    """
    tensor_groups = {}
    for group_tuple, subject_ids in grouped_subjects.items():
        group_key = "_".join(str(g) for g in group_tuple)
        tensor_groups[group_key] = [os.path.join(tensor_dir, f"{group_tuple[0]}_tensor_{sid}.pt")
                                    for sid in subject_ids]
    logging.info(f"Built tensor file paths for {len(tensor_groups)} groups.")
    return tensor_groups


def save_pickle(data: Any, file_path: str) -> None:
    """
    Save data using pickle.
    
    Args:
        data (Any): Data to pickle.
        file_path (str): File path for the pickle file.
    """
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved pickle file at {file_path}")
    except Exception as e:
        logging.error(f"Error saving pickle file {file_path}: {e}")


def normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Normalize each channel of the tensor to [0, 1].
    
    Args:
        tensor (torch.Tensor): Input tensor of shape (C, H, W).
        
    Returns:
        torch.Tensor: Normalized tensor.
    """
    # Compute min and max per channel; view tensor as (C, -1)
    tensor_flat = tensor.view(tensor.size(0), -1)
    min_vals = tensor_flat.min(dim=1, keepdim=True)[0].view(tensor.size(0), 1, 1)
    max_vals = tensor_flat.max(dim=1, keepdim=True)[0].view(tensor.size(0), 1, 1)
    ranges = max_vals - min_vals + 1e-8
    normalized = (tensor - min_vals) / ranges
    return normalized


def zero_diagonals(tensor: torch.Tensor) -> torch.Tensor:
    """
    Set the diagonal elements of each channel's square matrix to zero.
    Assumes tensor shape is (C, H, W) with H == W.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        
    Returns:
        torch.Tensor: Tensor with zeroed diagonals.
    """
    if tensor.size(1) != tensor.size(2):
        logging.warning("zero_diagonals: Expected square matrices (H==W); skipping diagonal zeroing.")
        return tensor
    indices = torch.arange(tensor.size(1))
    tensor[:, indices, indices] = 0
    return tensor


def log_transform(tensor: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    """
    Safely apply the natural logarithm transform to the tensor.
    
    Args:
        tensor (torch.Tensor): Input tensor.
        epsilon (float): Small constant to avoid log(0).
        
    Returns:
        torch.Tensor: Log-transformed tensor.
    """
    return torch.log(tensor + epsilon)


def load_tensor(fp: str) -> Optional[torch.Tensor]:
    """
    Load a tensor from a .pt file. Converts numpy arrays to torch.Tensors if needed.
    Applies preprocessing functions (zeroing diagonals, etc.).
    
    Args:
        fp (str): File path.
        
    Returns:
        Optional[torch.Tensor]: Preprocessed tensor, or None if error occurs.
    """
    if not os.path.exists(fp):
        logging.warning(f"Missing tensor file: {fp}")
        return None

    try:
        data = torch.load(fp, map_location='cpu')
        if isinstance(data, np.ndarray):
            data = torch.tensor(data)
        if not isinstance(data, torch.Tensor):
            logging.warning(f"Unexpected data format in {fp}. Expected torch.Tensor, got {type(data)}")
            return None
        # Apply preprocessing: zero diagonals
        data = zero_diagonals(data)
        # Optionally, apply log_transform on a specific channel if needed:
        # if data.shape[0] > 2:
        #     data[2] = log_transform(data[2])
        return data
    except Exception as e:
        logging.error(f"Error loading tensor {fp}: {e}")
        return None


def load_tensors(file_paths: List[str], batch_size: int = 16) -> Optional[torch.Tensor]:
    """
    Load tensors from a list of file paths in batches. Skips missing or corrupted files.
    
    Args:
        file_paths (List[str]): List of file paths.
        batch_size (int): Batch size for processing.
        
    Returns:
        Optional[torch.Tensor]: Stacked tensor of shape (N, ...), or None if no valid tensor found.
    """
    tensors = []
    for i in range(0, len(file_paths), batch_size):
        batch_paths = file_paths[i:i + batch_size]
        for fp in batch_paths:
            tensor = load_tensor(fp)
            if tensor is not None:
                tensors.append(tensor)
    if not tensors:
        logging.error("No valid tensors were loaded. Check dataset files.")
        return None
    try:
        stacked = torch.stack(tensors)
        return stacked
    except Exception as e:
        logging.error(f"Error stacking tensors: {e}")
        return None


def compute_mean_std_per_channel(dataset: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the mean and standard deviation per channel for a 4D dataset.
    Expects dataset shape: (Batch, Channels, Height, Width)
    
    Args:
        dataset (torch.Tensor): Input dataset tensor.
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (mean, std) per channel.
    """
    # Compute mean and std across Batch, Height, and Width dimensions.
    mean = dataset.mean(dim=(0, 2, 3))
    std = dataset.std(dim=(0, 2, 3))
    # Avoid division by zero
    std[std == 0] = 1e-9
    return mean, std


def normalize_dataset(dataset: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """
    Normalize a dataset using provided per-channel mean and std.
    
    Args:
        dataset (torch.Tensor): Dataset tensor with shape (Batch, Channels, Height, Width).
        mean (torch.Tensor): Mean per channel.
        std (torch.Tensor): Standard deviation per channel.
        
    Returns:
        torch.Tensor: Normalized dataset.
    """
    epsilon = 1e-9
    normalized = (dataset - mean[None, :, None, None]) / (std[None, :, None, None] + epsilon)
    return normalized


def save_dataset(data: Optional[torch.Tensor], filename: str, project_dir: str) -> None:
    """
    Save a tensor dataset to disk.
    
    Args:
        data (Optional[torch.Tensor]): Tensor data to save.
        filename (str): Filename for the saved dataset.
        project_dir (str): Base directory to save the file.
    """
    if data is None:
        logging.warning(f"Dataset {filename} is None; skipping save.")
        return
    save_path = os.path.join(project_dir, filename)
    try:
        torch.save(data, save_path)
        logging.info(f"Saved dataset to {save_path}")
    except Exception as e:
        logging.error(f"Error saving dataset to {save_path}: {e}")


def main():
    # Mount drive if needed
    mount_drive_if_needed()
    
    # Determine project directory and file paths
    project_dir = get_project_directory()
    csv_path = os.path.join(project_dir, 'DataBaseSubjects.csv')
    tensor_data_dir = os.path.join(project_dir, 'TensorData')
    config_path = os.path.join(project_dir, 'preprocess', 'config.yaml')
    
    # Load configuration
    config = load_config(config_path)
    seed = config.get('seed', 42)
    set_random_seed(seed)
    
    # Load subject metadata
    subjects_df = load_subjects_dataframe(csv_path)
    log_median_age(subjects_df)
    
    # Group subjects by ResearchGroup and Sex
    grouped_subjects = group_subjects_by_attributes(subjects_df, ['ResearchGroup', 'Sex'], 'SubjectID')
    
    # Build tensor file paths dictionary
    tensor_groups = build_tensor_paths(tensor_data_dir, grouped_subjects)
    
    # Split data into train, validation, and test sets across groups
    train_set, val_set, test_set = [], [], []
    for group_key, file_list in tensor_groups.items():
        train, val, test = split_data(file_list)
        train_set.extend(train)
        val_set.extend(val)
        test_set.extend(test)
    
    # Save test set paths for reproducibility
    test_set_path = os.path.join(project_dir, 'test_set_paths.pkl')
    save_pickle(test_set, test_set_path)
    
    # Load tensor datasets
    logging.info("Loading training tensors...")
    train_data = load_tensors(train_set)
    logging.info("Loading validation tensors...")
    val_data = load_tensors(val_set)
    logging.info("Loading test tensors...")
    test_data = load_tensors(test_set)
    
    # Compute and log normalization parameters from training data
    if train_data is not None:
        train_mean, train_std = compute_mean_std_per_channel(train_data)
        logging.info(f"Training set mean per channel: {train_mean}")
        logging.info(f"Training set std per channel: {train_std}")
        
        # Normalize datasets
        train_data = normalize_dataset(train_data, train_mean, train_std)
        if val_data is not None:
            val_data = normalize_dataset(val_data, train_mean, train_std)
        if test_data is not None:
            test_data = normalize_dataset(test_data, train_mean, train_std)
    else:
        logging.error("Training data is None; cannot compute normalization parameters.")
        sys.exit(1)
    
    # Save normalized datasets
    save_dataset(train_data, 'train_data.pt', project_dir)
    save_dataset(val_data, 'val_data.pt', project_dir)
    save_dataset(test_data, 'test_data.pt', project_dir)
    
    # Also, save train and validation set paths for future reference
    train_set_path = os.path.join(project_dir, 'train_set_paths.pkl')
    val_set_path = os.path.join(project_dir, 'val_set_paths.pkl')
    save_pickle(train_set, train_set_path)
    save_pickle(val_set, val_set_path)
    
    logging.info("Preprocessing complete. All datasets normalized and saved.")


if __name__ == '__main__':
    # Optional: use argparse if you want to allow command-line parameters.
    parser = argparse.ArgumentParser(description="Preprocess tensor data for Alzheimer fMRI analysis.")
    # Add any desired arguments here; for now we just run main.
    args = parser.parse_args()
    main()
