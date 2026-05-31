import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import ResidualBlock, make_group_norm


def _make_norm2d(norm: str, ch: int, num_groups: int = 8) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "in":
        return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)
    g = min(num_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class Encoder(nn.Module):
    """A shared encoder to extract multi-scale features from input frames."""

    def __init__(
        self, in_ch: int = 1, base_ch: int = 32, norm: str = "gn", gn_groups: int = 8
    ):
        super().__init__()
        self.norm = norm
        self.gn_groups = gn_groups

        self.conv1 = nn.Conv2d(in_ch, base_ch, 3, stride=1, padding=1)
        self.n1 = _make_norm2d(norm, base_ch, gn_groups)
        self.layer1 = self._make_layer(base_ch, base_ch, 2)

        self.conv2 = nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1)
        self.n2 = _make_norm2d(norm, base_ch * 2, gn_groups)
        self.layer2 = self._make_layer(base_ch * 2, base_ch * 2, 2)

        self.conv3 = nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1)
        self.n3 = _make_norm2d(norm, base_ch * 4, gn_groups)
        self.layer3 = self._make_layer(base_ch * 4, base_ch * 4, 2)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int) -> nn.Sequential:
        layers = [
            ResidualBlock(
                in_channels, out_channels, norm=self.norm, gn_groups=self.gn_groups
            )
        ]
        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock(
                    out_channels, out_channels, norm=self.norm, gn_groups=self.gn_groups
                )
            )
        return nn.Sequential(*layers)

    def forward(
        self, x: torch.Tensor
    ) -> list[torch.Tensor]:
        feat_lvl1 = self.layer1(F.gelu(self.n1(self.conv1(x))))
        feat_lvl2 = self.layer2(F.gelu(self.n2(self.conv2(feat_lvl1))))
        feat_lvl3 = self.layer3(F.gelu(self.n3(self.conv3(feat_lvl2))))
        return [feat_lvl3, feat_lvl2, feat_lvl1]

class SharedDecoder(nn.Module):
    """Upsample coarse-to-fine hidden states into a full-resolution residual."""

    def __init__(
        self,
        coarse_ch: int,
        mid_ch: int,
        fine_ch: int,
        out_ch: int = 1,
    ) -> None:
        super().__init__()

        self.coarse_refine = ResidualBlock(coarse_ch, coarse_ch)
        self.coarse_to_mid = nn.Sequential(
            nn.Conv2d(coarse_ch, mid_ch, 3, padding=1, bias=False),
            make_group_norm(mid_ch),
            nn.GELU(),
        )
        self.mid_fuse = nn.Sequential(
            nn.Conv2d(mid_ch * 2, mid_ch, 3, padding=1, bias=False),
            make_group_norm(mid_ch),
            nn.GELU(),
            ResidualBlock(mid_ch, mid_ch),
        )

        self.mid_to_fine = nn.Sequential(
            nn.Conv2d(mid_ch, fine_ch, 3, padding=1, bias=False),
            make_group_norm(fine_ch),
            nn.GELU(),
        )
        self.fine_fuse = nn.Sequential(
            nn.Conv2d(fine_ch * 2, fine_ch, 3, padding=1, bias=False),
            make_group_norm(fine_ch),
            nn.GELU(),
            ResidualBlock(fine_ch, fine_ch),
        )

        self.output_head = nn.Sequential(
            ResidualBlock(fine_ch, fine_ch),
            nn.Conv2d(fine_ch, out_ch, 3, padding=1),
        )

    def forward(
        self,
        coarse: torch.Tensor,
        mid: torch.Tensor,
        fine: torch.Tensor,
    ) -> torch.Tensor:
        coarse_refined = self.coarse_refine(coarse)

        up_mid = F.interpolate(
            coarse_refined,
            size=mid.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up_mid = self.coarse_to_mid(up_mid)
        mid_refined = self.mid_fuse(torch.cat([up_mid, mid], dim=1))

        up_fine = F.interpolate(
            mid_refined,
            size=fine.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        up_fine = self.mid_to_fine(up_fine)
        fine_refined = self.fine_fuse(torch.cat([up_fine, fine], dim=1))

        return self.output_head(fine_refined)
