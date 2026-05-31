"""Training and validation engine for stereo reconstruction."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torchvision.utils as vutils
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from tqdm import tqdm

from utils.sequence_utils import (
    StateEntry,
    append_and_truncate as _append_and_truncate,
    metric_scalar as _metric_scalar,
    prepare_inputs as _prepare_inputs,
    update_history as _update_history,
)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optim: torch.optim.Optimizer,
    recon_criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    step_offset: int,
    history_length: int,
    *,
    scaler: torch.cuda.amp.GradScaler | None = None,
    amp_dtype: torch.dtype | None = None,
    sequence_mode: bool = True,
    grad_clip_norm: float = 0.0,
    **_unused: Any,
) -> tuple[float, int]:
    """Train one epoch (reconstruction only).

    Returns:
        (avg_train_loss, next_step)
    """
    _ = writer
    model.train()

    running_loss = 0.0
    pbar = tqdm(enumerate(loader), total=len(loader))

    for i, batch in pbar:
        left_seq = batch["left"].to(device, non_blocking=True)
        right_seq = batch["right"].to(device, non_blocking=True)
        f_clip = batch["f_clip"].to(device, non_blocking=True)
        target_a = batch["target_a"].to(device, non_blocking=True)
        target_b = batch["target_b"].to(device, non_blocking=True)
        mt = batch["mt"].to(device, non_blocking=True)
        threshold = batch["threshold"].to(device, non_blocking=True)
        mt = mt * threshold

        optim.zero_grad()
        sequence_len = int(left_seq.shape[1])
        sum_loss_for_log = 0.0

        state_hist_left: list[StateEntry] = []
        state_hist_right: list[StateEntry] = []
        left_hist_imgs: list[torch.Tensor] = []
        right_hist_imgs: list[torch.Tensor] = []
        nonfinite_batch = False

        batch_tensors = (left_seq, right_seq, f_clip, target_a, target_b, mt, threshold)
        if not all(torch.isfinite(t).all().item() for t in batch_tensors):
            print(f"[WARN] Skipping train batch {i}: non-finite input tensor.")
            continue

        for j in range(sequence_len):
            inputs = _prepare_inputs(
                left_seq[:, j],
                right_seq[:, j],
                f_clip,
                state_hist_left,
                state_hist_right,
                sequence_mode,
            )
            inputs["target_a"] = None
            inputs["target_b"] = None
            inputs["ref"] = mt[:, j]

            if sequence_mode:
                if len(left_hist_imgs) > 0:
                    inputs["left_history_imgs"] = tuple(left_hist_imgs)
                if len(right_hist_imgs) > 0:
                    inputs["right_history_imgs"] = tuple(right_hist_imgs)

            amp_context = (
                torch.amp.autocast("cuda", dtype=amp_dtype)
                if amp_dtype is not None and device.type == "cuda"
                else nullcontext()
            )
            with amp_context:
                outputs = model(inputs)

            pred_a = outputs["pred_a"].float().clamp(0.0, 1.0)
            pred_b = outputs["pred_b"].float().clamp(0.0, 1.0)

            if not torch.isfinite(pred_a).all().item() or not torch.isfinite(pred_b).all().item():
                print(f"[WARN] Skipping train batch {i}: non-finite model output at frame {j}.")
                nonfinite_batch = True
                break

            _update_history(state_hist_left, outputs.get("state_left"), history_length)
            _update_history(state_hist_right, outputs.get("state_right"), history_length)
            _append_and_truncate(left_hist_imgs, pred_a, history_length)
            _append_and_truncate(right_hist_imgs, pred_b, history_length)

            recon_loss = recon_criterion(pred_a, target_a[:, j]) + recon_criterion(
                pred_b, target_b[:, j]
            )

            if not torch.isfinite(recon_loss).all().item():
                print(f"[WARN] Skipping train batch {i}: non-finite loss at frame {j}.")
                nonfinite_batch = True
                break

            per_frame_loss = recon_loss / max(sequence_len, 1)
            if scaler is not None and device.type == "cuda":
                scaler.scale(per_frame_loss).backward()
            else:
                per_frame_loss.backward()

            sum_loss_for_log += float(per_frame_loss.detach().item())

        if nonfinite_batch:
            optim.zero_grad(set_to_none=True)
            continue

        grad_norm = None
        if scaler is not None and device.type == "cuda":
            scaler.unscale_(optim)
            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm).all().item():
                    print(f"[WARN] Skipping train batch {i}: non-finite gradient norm.")
                    optim.zero_grad(set_to_none=True)
                    scaler.update()
                    continue
            scaler.step(optim)
            scaler.update()
        else:
            if grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                if not torch.isfinite(grad_norm).all().item():
                    print(f"[WARN] Skipping train batch {i}: non-finite gradient norm.")
                    optim.zero_grad(set_to_none=True)
                    continue
            optim.step()

        running_loss += sum_loss_for_log
        pbar.set_description(f"train loss: {running_loss / (i + 1):.4f}")

    avg_train_loss = running_loss / max(1, len(loader))
    return avg_train_loss, step_offset + len(loader)


def validate(
    model: nn.Module,
    loader: DataLoader,
    recon_criterion: nn.Module,
    device: torch.device,
    writer: SummaryWriter,
    epoch: int,
    history_length: int,
    *,
    sequence_mode: bool = True,
    max_val_batches: int = 0,
    **_unused: Any,
) -> tuple[float, float, float, float, float]:
    """Run validation (reconstruction only).

    Returns:
      (
        val_loss,
        val_psnr_a,
        val_psnr_b,
        val_ssim_a,
        val_ssim_b,
      )
    """
    orig_training = model.training
    model.eval()

    running_loss = 0.0
    running_psnr_a = 0.0
    running_psnr_b = 0.0
    running_ssim_a = 0.0
    running_ssim_b = 0.0
    frame_counter = 0

    display_idx = 0

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc="val")):
            left_seq = batch["left"].to(device, non_blocking=True)
            right_seq = batch["right"].to(device, non_blocking=True)
            f_clip = batch["f_clip"].to(device, non_blocking=True)
            target_a = batch["target_a"].to(device, non_blocking=True)
            target_b = batch["target_b"].to(device, non_blocking=True)
            mt = batch["mt"].to(device, non_blocking=True)
            threshold = batch["threshold"].to(device, non_blocking=True)
            mt = mt * threshold

            state_hist_left: list[StateEntry] = []
            state_hist_right: list[StateEntry] = []
            left_hist_imgs: list[torch.Tensor] = []
            right_hist_imgs: list[torch.Tensor] = []

            last_pred_a: torch.Tensor | None = None
            last_pred_b: torch.Tensor | None = None

            sequence_len = int(left_seq.shape[1])

            for j in range(sequence_len):
                inputs = _prepare_inputs(
                    left_seq[:, j],
                    right_seq[:, j],
                    f_clip,
                    state_hist_left,
                    state_hist_right,
                    sequence_mode,
                )
                inputs["target_a"] = None
                inputs["target_b"] = None
                inputs["ref"] = mt[:, j]

                if sequence_mode:
                    if len(left_hist_imgs) > 0:
                        inputs["left_history_imgs"] = tuple(left_hist_imgs)
                    if len(right_hist_imgs) > 0:
                        inputs["right_history_imgs"] = tuple(right_hist_imgs)

                outputs = model(inputs)

                pred_a = outputs["pred_a"].clamp(0.0, 1.0)
                pred_b = outputs["pred_b"].clamp(0.0, 1.0)
                last_pred_a = pred_a
                last_pred_b = pred_b

                _update_history(state_hist_left, outputs.get("state_left"), history_length)
                _update_history(state_hist_right, outputs.get("state_right"), history_length)
                _append_and_truncate(left_hist_imgs, pred_a, history_length)
                _append_and_truncate(right_hist_imgs, pred_b, history_length)

                recon_loss = recon_criterion(pred_a, target_a[:, j]) + recon_criterion(
                    pred_b, target_b[:, j]
                )
                running_loss += float(recon_loss.item())
                frame_counter += 1

                psnr_a_raw = peak_signal_noise_ratio(
                    pred_a,
                    target_a[:, j],
                    data_range=1.0,
                )
                psnr_b_raw = peak_signal_noise_ratio(
                    pred_b,
                    target_b[:, j],
                    data_range=1.0,
                )
                running_psnr_a += float(
                    _metric_scalar(psnr_a_raw, device=device, dtype=pred_a.dtype).item()
                )
                running_psnr_b += float(
                    _metric_scalar(psnr_b_raw, device=device, dtype=pred_b.dtype).item()
                )

                pred_a_4d = pred_a if pred_a.dim() == 4 else pred_a.unsqueeze(1)
                target_a_4d = (
                    target_a[:, j]
                    if target_a[:, j].dim() == 4
                    else target_a[:, j].unsqueeze(1)
                )
                pred_b_4d = pred_b if pred_b.dim() == 4 else pred_b.unsqueeze(1)
                target_b_4d = (
                    target_b[:, j]
                    if target_b[:, j].dim() == 4
                    else target_b[:, j].unsqueeze(1)
                )

                ssim_a_raw = structural_similarity_index_measure(
                    pred_a_4d,
                    target_a_4d,
                    data_range=1.0,
                )
                ssim_b_raw = structural_similarity_index_measure(
                    pred_b_4d,
                    target_b_4d,
                    data_range=1.0,
                )
                running_ssim_a += float(
                    _metric_scalar(ssim_a_raw, device=device, dtype=pred_a.dtype).item()
                )
                running_ssim_b += float(
                    _metric_scalar(ssim_b_raw, device=device, dtype=pred_b.dtype).item()
                )

            if i == display_idx and last_pred_a is not None and last_pred_b is not None:
                try:
                    mid = sequence_len // 2
                    inp_mid = left_seq[:, mid]
                    inp_right = right_seq[:, mid]

                    def prep_img(t: torch.Tensor) -> torch.Tensor:
                        out = t.detach().cpu()
                        if out.dim() == 5 and out.size(2) == 1:
                            out = out.squeeze(2)
                        if out.dim() == 3:
                            out = out.unsqueeze(1)
                        return out.float()

                    inp_img = prep_img(inp_mid)
                    inp_right_img = prep_img(inp_right)
                    ta_img = prep_img(target_a[:, mid])
                    tb_img = prep_img(target_b[:, mid])
                    pred_a_img = prep_img(last_pred_a)
                    pred_b_img = prep_img(last_pred_b)

                    panels = [
                        inp_img.clip(0, 1) ** (1 / 2.2),
                        inp_right_img.clip(0, 1) ** (1 / 2.2),
                        ta_img,
                        tb_img,
                        pred_a_img,
                        pred_b_img,
                    ]

                    imgs = torch.cat(panels, dim=0)
                    batch_size = int(inp_img.shape[0])
                    grid = vutils.make_grid(imgs, nrow=batch_size, normalize=False)
                    writer.add_image("val/preview", grid, epoch)
                except (RuntimeError, ValueError, TypeError, OSError) as exc:
                    print(f"[WARN] Failed to write validation preview image: {exc}")


            if max_val_batches > 0 and (i + 1) >= max_val_batches:
                break

    avg_loss = running_loss / max(1, frame_counter)
    avg_psnr_a = running_psnr_a / max(1, frame_counter)
    avg_psnr_b = running_psnr_b / max(1, frame_counter)
    avg_ssim_a = running_ssim_a / max(1, frame_counter)
    avg_ssim_b = running_ssim_b / max(1, frame_counter)

    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/psnr_a", avg_psnr_a, epoch)
    writer.add_scalar("val/psnr_b", avg_psnr_b, epoch)
    writer.add_scalar("val/ssim_a", avg_ssim_a, epoch)
    writer.add_scalar("val/ssim_b", avg_ssim_b, epoch)

    model.train(orig_training)

    return (
        avg_loss,
        avg_psnr_a,
        avg_psnr_b,
        avg_ssim_a,
        avg_ssim_b,
    )
