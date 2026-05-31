"""Shared checkpoint and output helpers for test/inference entrypoints."""

from __future__ import annotations

import torch

from src.train.checkpoint import load_checkpoint_with_encoder_twins
from utils.output_saver import OutputSaver


def _print_load_report(missing: list[str], unexpected: list[str]) -> None:
    if missing:
        suffix = " ..." if len(missing) > 8 else ""
        print(f"[INFO] Missing keys ({len(missing)}): {missing[:8]}{suffix}")
    if unexpected:
        suffix = " ..." if len(unexpected) > 8 else ""
        print(f"[INFO] Unexpected keys ({len(unexpected)}): {unexpected[:8]}{suffix}")


def _load_partial_compatible_state(
    model: torch.nn.Module,
    loaded_state: dict[str, torch.Tensor],
) -> None:
    current_state = model.state_dict()
    filtered_state = {}
    skipped_keys = []
    for key, tensor in loaded_state.items():
        if key in current_state and current_state[key].shape == tensor.shape:
            filtered_state[key] = tensor
        else:
            skipped_keys.append(key)

    if not filtered_state:
        raise RuntimeError("No compatible tensors were found in the checkpoint.")

    if skipped_keys:
        print(f"[INFO] Skipping incompatible params ({len(skipped_keys)}).")
    current_state.update(filtered_state)
    model.load_state_dict(current_state, strict=False)


def load_model_checkpoint(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> dict:
    """Load model weights with model-only checkpoint compatibility."""
    ckpt = torch.load(ckpt_path, map_location=device)
    state_key = "model_state" if "model_state" in ckpt else None

    try:
        missing, unexpected = load_checkpoint_with_encoder_twins(
            model,
            ckpt_path,
            map_location=device,
            strict=False,
            state_key=state_key,
        )
        _print_load_report(missing, unexpected)
    except RuntimeError as exc:
        if "model_state" not in ckpt:
            raise
        print(f"[WARN] Partial state_dict load due to mismatch: {exc}")
        _load_partial_compatible_state(model, ckpt["model_state"])

    return ckpt


def save_frame_outputs(
    *,
    saver: OutputSaver,
    method_name: str,
    sample_idx: int,
    frame_idx: int,
    pred_a: torch.Tensor,
    pred_b: torch.Tensor,
    input_a: torch.Tensor | None = None,
    input_b: torch.Tensor | None = None,
    gt_a: torch.Tensor | None = None,
    gt_b: torch.Tensor | None = None,
    ref: torch.Tensor | None = None,
    pred_only: bool = True,
    save_inputs_gt: bool = False,
) -> None:
    tensors: dict[str, torch.Tensor] = {
        "left_pred": pred_a.detach().cpu().clamp(0, 1),
        "right_pred": pred_b.detach().cpu().clamp(0, 1),
    }

    if (not pred_only) or save_inputs_gt:
        if input_a is not None:
            tensors["left_input"] = input_a.detach().cpu().clamp(0, 1)
        if input_b is not None:
            tensors["right_input"] = input_b.detach().cpu().clamp(0, 1)
        if gt_a is not None:
            tensors["left_gt"] = gt_a.detach().cpu().clamp(0, 1)
        if gt_b is not None:
            tensors["right_gt"] = gt_b.detach().cpu().clamp(0, 1)
        if ref is not None:
            tensors["mt"] = ref.detach().cpu().clamp(0, 1)

    saver.save_sample_frame(
        method=method_name,
        sample_idx=sample_idx,
        frame_idx=frame_idx,
        tensors=tensors,
    )
