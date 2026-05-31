"""Setup helpers for data, model, and optimization in training."""

from __future__ import annotations

import argparse
from typing import Any, cast

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from dataset.dataset import StereoSimulationDataset
from models.refine import SMSNet
from utils.dataloader_utils import build_dataloader
from utils.utils_train import find_latest_checkpoint, parse_crop_size


def make_device(args: argparse.Namespace) -> torch.device:
    """Resolve runtime device from args with safe fallback."""
    requested = str(getattr(args, "device", "cpu")).strip().lower()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


def build_datasets(args: argparse.Namespace) -> tuple[Dataset[Any], Dataset[Any]]:
    """Build train/val datasets according to args."""
    crop = parse_crop_size(args.crop_size)

    train_ds: Dataset[Any] = StereoSimulationDataset(
        args.npz_path,
        sequence_length=args.sequence_length,
        split="train",
        crop_size=crop,
        clip_stride=int(getattr(args, "train_clip_stride", 1)),
    )
    val_ds: Dataset[Any] = StereoSimulationDataset(
        args.npz_path,
        sequence_length=args.sequence_length,
        split="metrics",
        crop_size=None,
    )

    max_train_samples = int(getattr(args, "max_train_samples", 0) or 0)
    if max_train_samples > 0 and len(train_ds) > max_train_samples:
        train_ds = Subset(train_ds, range(max_train_samples))

    return train_ds, val_ds


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader[Any], DataLoader[Any]]:
    """Build train/val dataloaders from args."""
    train_ds, val_ds = build_datasets(args)
    train_loader = build_dataloader(train_ds, args, shuffle=True)
    val_loader = build_dataloader(val_ds, args, shuffle=False)
    return train_loader, val_loader


def build_model(args: argparse.Namespace) -> tuple[nn.Module, bool]:
    """Build model and return (model, sequence_mode)."""
    if args.net_type == "flow":
        model = SMSNet(
            history_length=args.state_history,
        )
        sequence_mode = True
    else:
        raise ValueError(f"Unsupported net_type: {args.net_type}")

    return model, sequence_mode


def maybe_compile_model(model: nn.Module, args: argparse.Namespace) -> nn.Module:
    """Optionally compile model with torch.compile."""
    use_compile = bool(getattr(args, "compile", False))
    if not use_compile:
        return model

    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:
        print("[WARN] torch.compile is not available in this PyTorch build.")
        return model

    try:
        compiled = compile_fn(model)
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:  # pragma: no cover
        print(f"[WARN] torch.compile failed, using eager mode: {exc}")
        return model

    return cast(nn.Module, compiled)


def build_optimizer_scheduler(
    model: nn.Module,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.CosineAnnealingLR]:
    """Build optimizer and LR scheduler."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=1000,
        eta_min=1e-6,
    )
    return optimizer, scheduler


def resolve_amp_dtype(
    args: argparse.Namespace,
    device: torch.device,
) -> torch.dtype | None:
    """Resolve CUDA autocast dtype from config."""
    if not bool(getattr(args, "amp", False)) or device.type != "cuda":
        return None

    value = str(getattr(args, "amp_dtype", "fp16")).strip().lower()
    if value in {"bf16", "bfloat16"}:
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        print("[WARN] CUDA bf16 is not supported on this device. Falling back to fp16.")
        return torch.float16
    if value in {"fp16", "float16", "half"}:
        return torch.float16

    raise ValueError(f"Unsupported amp_dtype: {value!r}. Use 'fp16' or 'bf16'.")


def build_grad_scaler(
    args: argparse.Namespace,
    device: torch.device,
) -> torch.cuda.amp.GradScaler | None:
    """Build AMP gradient scaler for fp16 autocast."""
    amp_dtype = resolve_amp_dtype(args, device)
    if amp_dtype is torch.float16:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            try:
                return torch.amp.GradScaler("cuda")
            except TypeError:
                return torch.amp.GradScaler()
        return torch.cuda.amp.GradScaler()
    return None


def resolve_resume_checkpoint(args: argparse.Namespace) -> str | None:
    """Resolve resume checkpoint path based on args."""
    if not bool(getattr(args, "resume", False)):
        return None

    resume_path = getattr(args, "resume_path", None)
    if resume_path:
        return str(resume_path)

    return find_latest_checkpoint(args.checkpoint_dir, args.net_type, args.name)
