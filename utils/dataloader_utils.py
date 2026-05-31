"""Shared DataLoader construction helpers."""

from __future__ import annotations

import argparse
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from utils.utils_train import collate_fn


def dataloader_common_kwargs(num_workers: int, pin_memory: bool = True) -> dict[str, Any]:
    """Create DataLoader kwargs valid for both worker/no-worker modes."""
    kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
        kwargs["persistent_workers"] = True
    return kwargs


def build_dataloader(
    dataset: Dataset[Any],
    args: argparse.Namespace,
    *,
    shuffle: bool,
) -> DataLoader[Any]:
    pin_memory = str(getattr(args, "device", "")).startswith("cuda") and torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle,
        **dataloader_common_kwargs(int(args.num_workers), pin_memory=pin_memory),
    )
