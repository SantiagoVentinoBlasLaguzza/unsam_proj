import torch

def zero_diagonals(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        diag_indices = torch.eye(tensor.shape[0], device=tensor.device).bool()
        tensor[diag_indices] = 0
    elif tensor.dim() == 3:
        for c in range(tensor.shape[0]):
            diag_indices = torch.eye(tensor.shape[1], device=tensor.device).bool()
            tensor[c][diag_indices] = 0
    return tensor
