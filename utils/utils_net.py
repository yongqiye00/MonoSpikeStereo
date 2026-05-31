"""Utility functions for training: checkpointing and metrics."""

from typing import Dict, Optional

import math
import os

import torch


def save_checkpoint(state: Dict, filename: str) -> None:
    """Save training checkpoint to `filename`. Creates directories as needed."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)


def load_checkpoint(filename: str, device: Optional[torch.device] = None) -> Dict:
    """Load checkpoint and return the state dict. If device provided, map to device."""
    map_location = (
        None
        if device is None
        else (lambda storage, loc: storage.cuda() if device.type == "cuda" else storage)
    )
    return torch.load(filename, map_location=map_location)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Compute PSNR between pred and target."""
    mse = torch.mean((pred - target) ** 2)
    mse_value = float(mse.item())
    if mse_value <= 0.0:
        return float("inf")
    return 20.0 * math.log10(float(data_range)) - 10.0 * math.log10(mse_value)
