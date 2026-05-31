"""Shared training utilities."""

from __future__ import annotations

import glob
import os
from typing import Any, Dict, Iterable, Optional, Tuple

import torch


def collate_fn(batch: Iterable[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    l = torch.stack([item["left"] for item in batch], dim=0)
    r = torch.stack([item["right"] for item in batch], dim=0)
    f = torch.stack([item["f_clip"] for item in batch], dim=0)
    a = torch.stack([item["target_a"] for item in batch], dim=0)
    b = torch.stack([item["target_b"] for item in batch], dim=0)
    inten = torch.stack([item["intensity"] for item in batch], dim=0)
    mt = torch.stack([item["mt"] for item in batch], dim=0)

    threshold = torch.tensor(
        [item["threshold"] for item in batch],
        dtype=torch.float32,
        device=mt.device,
    ).view(-1, 1, 1, 1, 1)

    return {
        "left": l,
        "right": r,
        "f_clip": f,
        "target_a": a,
        "target_b": b,
        "intensity": inten,
        "mt": mt,
        "threshold": threshold,
    }


def parse_crop_size(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None:
        return None
    if isinstance(s, (list, tuple)):
        return tuple(map(int, s))  # type: ignore[return-value]

    if "x" in s:
        parts = s.split("x")
    elif "," in s:
        parts = s.split(",")
    else:
        raise ValueError("crop_size must be in format HxW or H,W")

    if len(parts) != 2:
        raise ValueError("crop_size must have two integers H and W")

    return int(parts[0]), int(parts[1])


def find_latest_checkpoint(
    checkpoint_dir: str, net_type: str, name: str
) -> Optional[str]:
    """
    Return the path to the latest checkpoint for the given net_type.

    Preference order:
    1. {name}.pth if exists
    2. best_{net_type}_{name}.pth if exists
    3. newest file matching ckpt_epoch_*_{net_type}*.pth by mtime
    4. newest .pth in the directory

    Returns None if no checkpoint found.
    """
    best = os.path.join(checkpoint_dir, f"{name}.pth")
    if os.path.exists(best):
        return best

    best = os.path.join(checkpoint_dir, f"best_{net_type}_{name}.pth")
    if os.path.exists(best):
        return best

    pattern = os.path.join(checkpoint_dir, f"ckpt_epoch_*_{net_type}*.pth")
    files = glob.glob(pattern)
    if not files:
        files = glob.glob(os.path.join(checkpoint_dir, "*.pth"))
    if not files:
        return None

    return max(files, key=os.path.getmtime)


__all__ = [
    "collate_fn",
    "parse_crop_size",
    "find_latest_checkpoint",
]
