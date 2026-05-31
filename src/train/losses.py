"""Training loss definitions and builder helpers."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure


def _as_scalar_tensor(value: object, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert metric outputs to a scalar tensor safely."""
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.to(device=device, dtype=dtype)
        return value.mean().to(device=device, dtype=dtype)

    if isinstance(value, (tuple, list)) and len(value) > 0:
        return _as_scalar_tensor(value[0], device=device, dtype=dtype)

    if isinstance(value, (float, int)):
        return torch.tensor(float(value), device=device, dtype=dtype)

    return torch.tensor(0.0, device=device, dtype=dtype)


class VGGPerceptualLoss(nn.Module):
    """Perceptual loss based on VGG16 ImageNet feature activations."""

    def __init__(self, selected_layers: tuple[int, ...] = (3, 8, 15)) -> None:
        super().__init__()
        self.selected_layers = set(selected_layers)

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

        self.vgg: nn.Sequential | None = None
        try:
            from torchvision.models import VGG16_Weights, vgg16

            weights = VGG16_Weights.IMAGENET1K_FEATURES
            vgg_model = vgg16(weights=weights)
            features = vgg_model.features
            if isinstance(features, nn.Sequential):
                truncated_features = nn.Sequential(*list(features.children())[:16])
                for param in truncated_features.parameters():
                    param.requires_grad_(False)
                self.vgg = truncated_features.eval()
        except (ImportError, OSError, RuntimeError, ValueError) as exc:  # pragma: no cover
            print(
                f"[WARN] VGG16 weights unavailable for perceptual loss ({exc}). "
                "Perceptual term will be zero."
            )

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x = x.float()

        mean = cast(torch.Tensor, self.mean)
        std = cast(torch.Tensor, self.std)
        return (x - mean) / std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.vgg is None:
            return pred.new_zeros((), dtype=pred.dtype)

        pred_feats = self._preprocess(pred)
        target_feats = self._preprocess(target)

        loss = pred.new_zeros((), dtype=pred.dtype)
        for idx, layer in enumerate(self.vgg):
            pred_feats = layer(pred_feats)
            target_feats = layer(target_feats)
            if idx in self.selected_layers:
                loss = loss + F.l1_loss(pred_feats, target_feats)
        return loss


class ReconstructionLoss(nn.Module):
    """Composite reconstruction loss: L1 + SSIM (+ optional perceptual)."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        perceptual_weight: float = 0.0,
        perceptual_loss: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.l1_weight = float(l1_weight)
        self.ssim_weight = float(ssim_weight)
        self.perceptual_weight = float(perceptual_weight)
        self.l1 = nn.L1Loss()

        self.perceptual: nn.Module | None
        if self.perceptual_weight > 0.0:
            self.perceptual = perceptual_loss if perceptual_loss is not None else VGGPerceptualLoss()
        else:
            self.perceptual = None

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        l1_term = self.l1(pred, target) * self.l1_weight

        ssim_raw = structural_similarity_index_measure(pred, target, data_range=1.0)
        ssim_val = _as_scalar_tensor(ssim_raw, device=pred.device, dtype=pred.dtype)
        ssim_term = (1.0 - ssim_val) * self.ssim_weight

        total = l1_term + ssim_term
        if self.perceptual is not None and self.perceptual_weight > 0.0:
            perceptual_term = self.perceptual(pred, target)
            total = total + self.perceptual_weight * perceptual_term

        return total


def build_reconstruction_loss(
    l1_weight: float = 1.0,
    ssim_weight: float = 0.1,
    perceptual_weight: float = 0.0,
    device: torch.device | None = None,
) -> nn.Module:
    """Create reconstruction loss module with optional perceptual term."""
    perceptual_loss: nn.Module | None = None
    if perceptual_weight > 0.0:
        perceptual_loss = VGGPerceptualLoss()

    loss = ReconstructionLoss(
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        perceptual_weight=perceptual_weight,
        perceptual_loss=perceptual_loss,
    )
    if device is not None:
        loss = loss.to(device)

    return loss


def build_reconstruction_loss_fn(
    device: torch.device | None = None,
    *,
    l1_weight: float = 1.0,
    ssim_weight: float = 0.1,
    perceptual_weight: float = 0.0,
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Build a function-form reconstruction loss for drop-in compatibility."""
    criterion = build_reconstruction_loss(
        l1_weight=l1_weight,
        ssim_weight=ssim_weight,
        perceptual_weight=perceptual_weight,
        device=device,
    )

    def _loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return criterion(pred, target)

    return _loss
