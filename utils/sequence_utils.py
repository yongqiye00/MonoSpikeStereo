"""Shared sequence-state helpers for train, test, and inference loops."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch


StateEntry = torch.Tensor | tuple[torch.Tensor, ...]
InputValue = torch.Tensor | tuple[StateEntry, ...] | None


def metric_scalar(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert metric output to a scalar tensor robustly."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.to(device=device, dtype=dtype)
        return value.mean().to(device=device, dtype=dtype)

    if isinstance(value, (tuple, list)) and len(value) > 0:
        return metric_scalar(value[0], device=device, dtype=dtype)

    if isinstance(value, (int, float)):
        return torch.tensor(float(value), device=device, dtype=dtype)

    return torch.tensor(0.0, device=device, dtype=dtype)


def prepare_inputs(
    left: torch.Tensor,
    right: torch.Tensor,
    f_clip: torch.Tensor,
    state_left_hist: Sequence[StateEntry],
    state_right_hist: Sequence[StateEntry],
    use_sequence: bool,
) -> dict[str, InputValue]:
    """Build one-frame model input dict."""
    inputs: dict[str, InputValue] = {
        "left": left,
        "right": right,
        "f_clip": f_clip,
    }

    if use_sequence:
        inputs["state_left_seq"] = tuple(state_left_hist)
        inputs["state_right_seq"] = tuple(state_right_hist)
    else:
        inputs["state_left"] = state_left_hist[-1] if state_left_hist else None
        inputs["state_right"] = state_right_hist[-1] if state_right_hist else None

    return inputs


def update_history(
    history: list[StateEntry],
    new_state: torch.Tensor | Sequence[torch.Tensor] | None,
    history_length: int,
) -> None:
    if new_state is None:
        return

    if isinstance(new_state, torch.Tensor):
        history.append(new_state.detach())
    else:
        history.append(tuple(t.detach() for t in new_state))

    if len(history) > history_length:
        history.pop(0)


def append_and_truncate(
    history: list[torch.Tensor],
    item: torch.Tensor,
    history_length: int,
) -> None:
    history.append(item.detach())
    if len(history) > history_length:
        history.pop(0)
