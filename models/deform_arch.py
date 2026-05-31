from typing import Optional

import torch
import torch.nn as nn
from torchvision.ops import DeformConv2d
from .core import ResidualBlock

def _make_norm2d(norm: str, ch: int, num_groups: int = 32) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "in":
        return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)
    g = min(num_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return nn.GroupNorm(g, ch)

class ModulatedDeformConvPack(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            k_h = k_w = kernel_size
        else:
            k_h, k_w = kernel_size
        self.offset_ch = 2 * k_h * k_w
        self.mask_ch = k_h * k_w
        self.op = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=bias,
        )
        self.supports_mask = True

    def forward(
        self,
        x: torch.Tensor,
        offsets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.supports_mask:
            if mask is not None:
                mask = mask.clamp(0.0, 1.0)
            return self.op(x, offsets, mask)
        return self.op(x, offsets)


class ResOffsetNet(nn.Module):
    def __init__(self, in_ch: int, kernel_size: int = 3):
        super().__init__()
        k2 = kernel_size * kernel_size
        self.offset_ch = 2 * k2
        self.mask_ch = k2
        self.out_ch = self.offset_ch + self.mask_ch
        mid_ch = max(32, in_ch // 2)
        self.trunk = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1),
            nn.GELU(),
            ResidualBlock(mid_ch, mid_ch),
        )
        self.offset_head = nn.Conv2d(mid_ch, self.offset_ch, 3, padding=1)
        self.mask_head = nn.Conv2d(mid_ch, self.mask_ch, 3, padding=1)
        self.scale_gain = nn.Parameter(torch.tensor(1.0))

    def forward(
        self, x: torch.Tensor, scale_factor: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.trunk(x)
        clamped_scale_gain = torch.clamp(self.scale_gain, 0.1, 5.0)
        offset = torch.tanh(self.offset_head(feats)) * (
            scale_factor * clamped_scale_gain
        )

        mask = torch.sigmoid(self.mask_head(feats))
        return offset, mask
