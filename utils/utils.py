import torch
import numpy as np
import logging

def zero_diagonals(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        diag_indices = torch.eye(tensor.shape[0], device=tensor.device).bool()
        tensor[diag_indices] = 0
    elif tensor.dim() == 3:
        for c in range(tensor.shape[0]):
            diag_indices = torch.eye(tensor.shape[1], device=tensor.device).bool()
            tensor[c][diag_indices] = 0
    return tensor

def set_random_seed(seed: int) -> None:
    """
    Set the seed for reproducibility in numpy and torch.
    
    Args:
        seed (int): Seed value.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    logging.info(f"Random seed set to: {seed}")

