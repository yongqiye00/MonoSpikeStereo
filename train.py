"""Train entrypoint.

Refactored to keep orchestration here and move heavy logic into `src.train`.
"""

from __future__ import annotations

import math
import os
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-monospikestereo")
)

import torch
from torch.utils.tensorboard import SummaryWriter

from src.train.checkpoint import load_checkpoint_with_encoder_twins, strip_compile_prefix
from src.train.config import parse_args
from src.train.engine import train_epoch, validate
from src.train.losses import build_reconstruction_loss
from src.train.setup import (
    build_dataloaders,
    build_grad_scaler,
    build_model,
    build_optimizer_scheduler,
    make_device,
    maybe_compile_model,
    resolve_amp_dtype,
    resolve_resume_checkpoint,
)
from utils.utils_net import save_checkpoint


def _restore_training_states(
    checkpoint: dict,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> None:
    required_keys = ("optim_state", "sched_state", "epoch", "step")
    missing_keys = [key for key in required_keys if key not in checkpoint]
    if missing_keys:
        missing = ", ".join(missing_keys)
        raise KeyError(
            f"Resume checkpoint is missing required training state: {missing}. "
            "Use --init_checkpoint for model-only initialization."
        )

    optimizer.load_state_dict(checkpoint["optim_state"])
    scheduler.load_state_dict(checkpoint["sched_state"])


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
    loaded_state = strip_compile_prefix(loaded_state)
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


def _load_model_checkpoint(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    *,
    allow_partial: bool,
) -> dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    state_key = "model_state" if "model_state" in ckpt else None

    try:
        missing, unexpected = load_checkpoint_with_encoder_twins(
            model,
            ckpt_path,
            map_location=device,
            strict=not allow_partial,
            state_key=state_key,
        )
        _print_load_report(missing, unexpected)
    except RuntimeError as exc:
        if not allow_partial or "model_state" not in ckpt:
            raise
        print(f"[WARN] Partial state_dict load due to mismatch: {exc}")
        _load_partial_compatible_state(model, ckpt["model_state"])

    return ckpt


def _model_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    module = getattr(model, "_orig_mod", model)
    return module.state_dict()


def main() -> None:
    args = parse_args()
    device = make_device(args)

    train_loader, val_loader = build_dataloaders(args)

    model, sequence_mode = build_model(args)
    model = model.to(device)

    recon_criterion = build_reconstruction_loss(
        device=device,
        l1_weight=1.0,
        ssim_weight=0.1,
        perceptual_weight=0.0,
    )

    writer = SummaryWriter(args.log_dir)

    best_val = float("inf")
    step = 0
    start_epoch = 0

    init_checkpoint = getattr(args, "init_checkpoint", None)
    resume_ckpt = resolve_resume_checkpoint(args)
    if init_checkpoint and resume_ckpt is not None:
        raise ValueError(
            "--init_checkpoint and --resume/--resume_path are mutually exclusive. "
            "Use --init_checkpoint for fine-tuning or --resume to continue training state."
        )

    resume_checkpoint = None
    if resume_ckpt is not None:
        print(f"Loading checkpoint: {resume_ckpt}")
        resume_checkpoint = _load_model_checkpoint(
            model,
            resume_ckpt,
            device,
            allow_partial=False,
        )

        if "epoch" in resume_checkpoint:
            start_epoch = int(resume_checkpoint["epoch"]) + 1
        if "step" in resume_checkpoint:
            step = int(resume_checkpoint["step"])
        if "val_loss" in resume_checkpoint:
            best_val = float(resume_checkpoint.get("val_loss", best_val))
        print(f"Resuming from epoch {start_epoch}, best_val={best_val:.6f}, step={step}")
    elif init_checkpoint:
        print(f"Initializing model weights from checkpoint: {init_checkpoint}")
        _load_model_checkpoint(
            model,
            str(init_checkpoint),
            device,
            allow_partial=True,
        )
        print("Fine-tuning from epoch 0 with fresh optimizer/scheduler state.")

    model = maybe_compile_model(model, args)
    optimizer, scheduler = build_optimizer_scheduler(model, args)
    if resume_checkpoint is not None:
        _restore_training_states(resume_checkpoint, optimizer, scheduler)

    amp_dtype = resolve_amp_dtype(args, device)
    scaler = build_grad_scaler(args, device)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_every_epochs = max(1, int(getattr(args, "save_every_epochs", 5)))
    best_path = os.path.join(args.checkpoint_dir, f"{args.name}.pth")
    if os.path.exists(best_path):
        try:
            existing_best = torch.load(best_path, map_location="cpu")
            if "val_loss" in existing_best and math.isfinite(float(existing_best["val_loss"])):
                best_val = float(existing_best["val_loss"])
                print(f"Existing best checkpoint found: {best_path}, best_val={best_val:.6f}")
        except (OSError, RuntimeError, KeyError, TypeError, ValueError) as exc:
            print(f"[WARN] Failed to inspect existing best checkpoint: {exc}")

    for epoch in range(start_epoch, args.epochs):
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch} starting, lr={lr:.3e}")

        train_loss, step = train_epoch(
            model,
            train_loader,
            optimizer,
            recon_criterion,
            device,
            writer,
            step,
            history_length=args.state_history,
            scaler=scaler,
            amp_dtype=amp_dtype,
            sequence_mode=sequence_mode,
            grad_clip_norm=float(getattr(args, "grad_clip_norm", 0.0)),
        )
        writer.add_scalar("train/loss", train_loss, epoch)

        val_metrics = validate(
            model,
            val_loader,
            recon_criterion,
            device,
            writer,
            epoch,
            history_length=args.state_history,
            sequence_mode=sequence_mode,
            max_val_batches=int(getattr(args, "max_val_batches", 0)),
        )

        (
            val_loss,
            val_psnr_a,
            val_psnr_b,
            val_ssim_a,
            val_ssim_b,
        ) = val_metrics

        print(
            f"val_loss={val_loss:.4f}, val_psnr_a={val_psnr_a:.2f}, val_psnr_b={val_psnr_b:.2f}, "
            f"val_ssim_a={val_ssim_a:.4f}, val_ssim_b={val_ssim_b:.4f}"
        )

        scheduler.step()

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        finite_val_loss = math.isfinite(float(val_loss))

        # Periodic checkpoint
        if finite_val_loss and (epoch + 1) % save_every_epochs == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir,
                f"ckpt_epoch_{epoch}_{args.net_type}_{args.name}.pth",
            )
            save_checkpoint(
                {
                    "epoch": epoch,
                    "step": step,
                    "model_state": _model_state_dict(model),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_path,
            )

        # Best checkpoint
        if finite_val_loss and (val_loss < best_val or not os.path.exists(best_path)):
            best_val = val_loss
            save_checkpoint(
                {
                    "epoch": epoch,
                    "step": step,
                    "model_state": _model_state_dict(model),
                    "optim_state": optimizer.state_dict(),
                    "sched_state": scheduler.state_dict(),
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"(new best, saved to {best_path})")

    writer.close()


if __name__ == "__main__":
    main()
