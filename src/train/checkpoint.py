"""Checkpoint utilities for training.

This module provides compatibility helpers for loading checkpoints where the
model used a single `encoder.*` prefix, while the current model expects twin
encoders: `encoder_left.*` and `encoder_right.*`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

import torch
from torch import nn


StateDict = Dict[str, torch.Tensor]
CheckpointLike = Union[Mapping[str, Any], StateDict]


def strip_compile_prefix(state_dict: Mapping[str, torch.Tensor]) -> StateDict:
    """Strip torch.compile's wrapper prefix from checkpoint keys."""
    prefix = "_orig_mod."
    if not any(key.startswith(prefix) for key in state_dict):
        return dict(state_dict)
    return {
        key[len(prefix) :] if key.startswith(prefix) else key: value
        for key, value in state_dict.items()
    }


def adapt_state_dict_for_encoder_twins(
    state_dict: Mapping[str, torch.Tensor],
    src_prefix: str = "encoder",
    left_prefix: str = "encoder_left",
    right_prefix: str = "encoder_right",
) -> StateDict:
    """Adapt a legacy state dict to twin-encoder naming.

    If keys like `encoder.xxx` exist while `encoder_left.xxx` / `encoder_right.xxx`
    are missing, this function copies the legacy parameters to both twin prefixes.
    Existing twin-prefixed keys are never overwritten.

    Parameters
    ----------
    state_dict:
        Original model state dictionary.
    src_prefix:
        Legacy encoder prefix.
    left_prefix:
        Left encoder prefix in current model.
    right_prefix:
        Right encoder prefix in current model.

    Returns
    -------
    dict
        A new state dict containing original keys plus adapted twin keys when needed.
    """
    new_sd: StateDict = dict(state_dict)

    has_src = any(k.startswith(f"{src_prefix}.") for k in state_dict.keys())
    has_left = any(k.startswith(f"{left_prefix}.") for k in state_dict.keys())
    has_right = any(k.startswith(f"{right_prefix}.") for k in state_dict.keys())

    if not has_src:
        return new_sd

    for key, value in state_dict.items():
        if not key.startswith(f"{src_prefix}."):
            continue

        suffix = key[len(src_prefix) + 1 :]  # remove "encoder."
        left_key = f"{left_prefix}.{suffix}"
        right_key = f"{right_prefix}.{suffix}"

        if not has_left and left_key not in new_sd:
            new_sd[left_key] = value
        if not has_right and right_key not in new_sd:
            new_sd[right_key] = value

    return new_sd


def _extract_state_dict(
    checkpoint: CheckpointLike,
    state_key: Optional[str] = "state_dict",
) -> StateDict:
    """Extract model state dict from a checkpoint object."""
    if not isinstance(checkpoint, Mapping):
        # Already a raw state dict-like object
        return dict(checkpoint)  # type: ignore[arg-type]

    if state_key is not None and state_key in checkpoint:
        candidate = checkpoint[state_key]
        if isinstance(candidate, Mapping):
            return dict(candidate)  # type: ignore[arg-type]

    for k in ("state_dict", "model_state", "model"):
        if k in checkpoint and isinstance(checkpoint[k], Mapping):
            return dict(checkpoint[k])  # type: ignore[arg-type]

    # Fallback: treat checkpoint itself as state_dict
    return dict(checkpoint)  # type: ignore[arg-type]


def load_checkpoint_with_encoder_twins(
    model: nn.Module,
    ckpt_path: Union[str, Path],
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
    state_key: Optional[str] = "state_dict",
    src_prefix: str = "encoder",
    left_prefix: str = "encoder_left",
    right_prefix: str = "encoder_right",
) -> Tuple[list[str], list[str]]:
    """Load checkpoint and adapt legacy encoder weights to twin encoders.

    Parameters
    ----------
    model:
        Target model instance.
    ckpt_path:
        Path to checkpoint file.
    map_location:
        `torch.load` map location.
    strict:
        Passed to `model.load_state_dict`.
    state_key:
        Preferred key that stores model parameters inside checkpoint dict.
    src_prefix, left_prefix, right_prefix:
        Prefix names used for adaptation.

    Returns
    -------
    (missing_keys, unexpected_keys):
        Lists reported by `load_state_dict`.
    """
    checkpoint = torch.load(str(ckpt_path), map_location=map_location)
    state_dict = _extract_state_dict(checkpoint, state_key=state_key)
    state_dict = strip_compile_prefix(state_dict)
    state_dict = adapt_state_dict_for_encoder_twins(
        state_dict=state_dict,
        src_prefix=src_prefix,
        left_prefix=left_prefix,
        right_prefix=right_prefix,
    )

    incompatible = model.load_state_dict(state_dict, strict=strict)
    return list(incompatible.missing_keys), list(incompatible.unexpected_keys)
