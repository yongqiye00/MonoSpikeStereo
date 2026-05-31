from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core import (
    EnhanceBlock,
    QueryGuidedTemporalFuser,
    ResidualBlock,
    SCAM,
    make_group_norm,
)
from .deform_arch import (
    ModulatedDeformConvPack,
    ResOffsetNet,
)

Tensor = torch.Tensor
StateLevels = Sequence[Tensor]
StateSeq = Sequence[StateLevels]


class RSFModule(nn.Module):
    """Temporal core leveraging a multi-scale deformable alignment pyramid."""

    def __init__(
        self,
        feat_ch: int | Sequence[int],
        kernel_size: int = 3,
        max_history: int = 3,
    ):
        super().__init__()
        if isinstance(feat_ch, Sequence):
            self.pyramid_channels = list(feat_ch)
        else:
            self.pyramid_channels = [feat_ch]

        self.num_scales = len(self.pyramid_channels)
        self.max_history = max_history

        self.offset_net_left = nn.ModuleList()
        self.deform_conv_left = nn.ModuleList()
        self.offset_net_right = nn.ModuleList()
        self.deform_conv_right = nn.ModuleList()
        for ch in self.pyramid_channels:
            self.offset_net_left.append(
                ResOffsetNet(in_ch=ch * 2, kernel_size=kernel_size)
            )
            self.deform_conv_left.append(
                ModulatedDeformConvPack(ch, ch, kernel_size, padding=kernel_size // 2)
            )
            self.offset_net_right.append(
                ResOffsetNet(in_ch=ch * 2, kernel_size=kernel_size)
            )
            self.deform_conv_right.append(
                ModulatedDeformConvPack(ch, ch, kernel_size, padding=kernel_size // 2)
            )

        self.state_up_projs_left = nn.ModuleList()
        self.state_up_projs_right = nn.ModuleList()

        for idx in range(1, self.num_scales):
            prev_ch = self.pyramid_channels[idx - 1]
            curr_ch = self.pyramid_channels[idx]
            self.state_up_projs_left.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, curr_ch, 3, padding=1),
                    nn.GELU(),
                    ResidualBlock(curr_ch, curr_ch),
                )
            )
            self.state_up_projs_right.append(
                nn.Sequential(
                    nn.Conv2d(prev_ch, curr_ch, 3, padding=1),
                    nn.GELU(),
                    ResidualBlock(curr_ch, curr_ch),
                )
            )

        self.scale_enhancers_left = nn.ModuleList()
        self.scale_enhancers_right = nn.ModuleList()
        for idx in range(0, self.num_scales):
            ch = self.pyramid_channels[idx]
            self.scale_enhancers_left.append(
                nn.Sequential(
                    EnhanceBlock(ch * 2),
                    nn.Conv2d(ch * 2, ch, 1, bias=False),
                    nn.GELU(),
                )
            )
            self.scale_enhancers_right.append(
                nn.Sequential(
                    EnhanceBlock(ch * 2),
                    nn.Conv2d(ch * 2, ch, 1, bias=False),
                    nn.GELU(),
                )
            )

        self.cross_attns = nn.ModuleList(
            [SCAM(c=self.pyramid_channels[i]) for i in range(self.num_scales)]
        )

        self.temporal_fusers_left = nn.ModuleList(
            [
                QueryGuidedTemporalFuser(ch, num_heads=1, mode="guided")
                for ch in self.pyramid_channels
            ]
        )

        self.temporal_fusers_right = nn.ModuleList(
            [
                QueryGuidedTemporalFuser(ch, num_heads=1, mode="guided")
                for ch in self.pyramid_channels
            ]
        )

        # State memory heads for stabilizing historical states per side and scale.
        self.state_mem_heads_left = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                    make_group_norm(ch),
                    nn.GELU(),
                    ResidualBlock(ch, ch),
                )
                for ch in self.pyramid_channels
            ]
        )
        self.state_mem_heads_right = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=1, bias=False),
                    make_group_norm(ch),
                    nn.GELU(),
                    ResidualBlock(ch, ch),
                )
                for ch in self.pyramid_channels
            ]
        )

    def _normalize_sequence(
        self,
        seq: StateSeq | None,
        limit: int,
    ) -> list[list[Tensor]]:
        if seq is None or limit <= 0:
            return []

        normalized: list[list[Tensor]] = []
        for item in list(seq)[-limit:]:
            levels = [lvl for lvl in item if isinstance(lvl, torch.Tensor)]
            if levels:
                normalized.append(levels)
        return normalized

    def deform(
        self,
        scale_idx: int,
        hist_left_lvl: Tensor,
        target_left: Tensor,
        hist_right_lvl: Tensor,
        target_right: Tensor,
    ) -> tuple[Tensor, Tensor]:
        offset_left, mask_left = self.offset_net_left[scale_idx](
            torch.cat([hist_left_lvl, target_left], dim=1)
        )
        offset_right, mask_right = self.offset_net_right[scale_idx](
            torch.cat([hist_right_lvl, target_right], dim=1)
        )
        aligned_left = self.deform_conv_left[scale_idx](
            hist_left_lvl, offset_left, mask_left
        )
        aligned_right = self.deform_conv_right[scale_idx](
            hist_right_lvl, offset_right, mask_right
        )
        return aligned_left, aligned_right

    def _select_history_levels(
        self,
        history_states: list[list[Tensor]],
        hist_idx: int,
        fallback_levels: list[Tensor],
    ) -> list[Tensor]:
        return (
            history_states[hist_idx]
            if hist_idx < len(history_states)
            else fallback_levels
        )

    def _fuse_history_at_scale(
        self,
        scale_idx: int,
        target_left: Tensor,
        target_right: Tensor,
        hist_left_states: list[list[Tensor]],
        hist_right_states: list[list[Tensor]],
        history_len: int,
        prev_aligned_left: list[Tensor | None],
        prev_aligned_right: list[Tensor | None],
    ) -> tuple[Tensor, Tensor]:
        if history_len <= 0:
            return target_left, target_right

        aligned_left_list: list[Tensor] = []
        aligned_right_list: list[Tensor] = []

        for hist_idx in range(history_len):
            left_levels = self._select_history_levels(
                hist_left_states, hist_idx, hist_left_states[-1]
            )
            right_levels = self._select_history_levels(
                hist_right_states, hist_idx, hist_right_states[-1]
            )

            hist_left_lvl = left_levels[scale_idx]
            hist_right_lvl = right_levels[scale_idx]

            if scale_idx == 0 or prev_aligned_left[hist_idx] is None:
                aligned_left_lvl, aligned_right_lvl = self.deform(
                    scale_idx,
                    hist_left_lvl,
                    target_left,
                    hist_right_lvl,
                    target_right,
                )
            else:
                aligned_hist_left, aligned_hist_right = self.deform(
                    scale_idx,
                    hist_left_lvl,
                    target_left,
                    hist_right_lvl,
                    target_right,
                )

                prev_left = prev_aligned_left[hist_idx]
                assert prev_left is not None
                propagated_left = F.interpolate(
                    prev_left,
                    size=target_left.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                propagated_left = self.state_up_projs_left[scale_idx - 1](
                    propagated_left
                )
                aligned_left_lvl = aligned_hist_left + propagated_left

                prev_right = prev_aligned_right[hist_idx]
                assert prev_right is not None
                propagated_right = F.interpolate(
                    prev_right,
                    size=target_right.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
                propagated_right = self.state_up_projs_right[scale_idx - 1](
                    propagated_right
                )
                aligned_right_lvl = aligned_hist_right + propagated_right

            aligned_left_list.append(aligned_left_lvl)
            aligned_right_list.append(aligned_right_lvl)
            prev_aligned_left[hist_idx] = aligned_left_lvl
            prev_aligned_right[hist_idx] = aligned_right_lvl

        fused_left_lvl = self.temporal_fusers_left[scale_idx](
            aligned_left_list, target_left
        )
        fused_right_lvl = self.temporal_fusers_right[scale_idx](
            aligned_right_list, target_right
        )
        return fused_left_lvl, fused_right_lvl

    def _enhance_and_cross_attend(
        self,
        scale_idx: int,
        fused_left_lvl: Tensor,
        fused_right_lvl: Tensor,
        target_left: Tensor,
        target_right: Tensor,
    ) -> tuple[Tensor, Tensor]:
        enhanced_left_lvl = self.scale_enhancers_left[scale_idx](
            torch.cat([fused_left_lvl, target_left], dim=1)
        )
        enhanced_right_lvl = self.scale_enhancers_right[scale_idx](
            torch.cat([fused_right_lvl, target_right], dim=1)
        )
        return self.cross_attns[scale_idx](enhanced_left_lvl, enhanced_right_lvl)

    def forward(
        self,
        feat_left: Tensor | Sequence[Tensor],
        feat_right: Tensor | Sequence[Tensor],
        state_left_seq: StateSeq,
        state_right_seq: StateSeq,
    ) -> tuple[
        tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...]
    ]:
        feat_left_levels = (
            feat_left if isinstance(feat_left, Sequence) else [feat_left]
        )
        feat_right_levels = (
            feat_right if isinstance(feat_right, Sequence) else [feat_right]
        )

        hist_left_states = self._normalize_sequence(state_left_seq, self.max_history)
        hist_right_states = self._normalize_sequence(state_right_seq, self.max_history)
        history_len: int = max(len(hist_left_states), len(hist_right_states))

        new_state_left_levels: list[Tensor] = []
        new_state_right_levels: list[Tensor] = []
        dec_left_levels: list[Tensor] = []
        dec_right_levels: list[Tensor] = []

        if history_len > 0:
            hist_left_states = hist_left_states or [list(feat_left_levels)]
            hist_right_states = hist_right_states or [list(feat_right_levels)]
            prev_aligned_left: list[Tensor | None] = [None] * history_len
            prev_aligned_right: list[Tensor | None] = [None] * history_len
        else:
            prev_aligned_left = []
            prev_aligned_right = []

        for scale_idx in range(self.num_scales):
            target_left = feat_left_levels[scale_idx]
            target_right = feat_right_levels[scale_idx]

            fused_left_lvl, fused_right_lvl = self._fuse_history_at_scale(
                scale_idx=scale_idx,
                target_left=target_left,
                target_right=target_right,
                hist_left_states=hist_left_states,
                hist_right_states=hist_right_states,
                history_len=history_len,
                prev_aligned_left=prev_aligned_left,
                prev_aligned_right=prev_aligned_right,
            )

            enhanced_left_lvl, enhanced_right_lvl = self._enhance_and_cross_attend(
                scale_idx=scale_idx,
                fused_left_lvl=fused_left_lvl,
                fused_right_lvl=fused_right_lvl,
                target_left=target_left,
                target_right=target_right,
            )

            dec_left_levels.append(enhanced_left_lvl)
            dec_right_levels.append(enhanced_right_lvl)

            mem_left_lvl = self.state_mem_heads_left[scale_idx](enhanced_left_lvl)
            mem_right_lvl = self.state_mem_heads_right[scale_idx](enhanced_right_lvl)
            new_state_left_levels.append(mem_left_lvl)
            new_state_right_levels.append(mem_right_lvl)

        return (
            tuple(new_state_left_levels),
            tuple(new_state_right_levels),
            tuple(dec_left_levels),
            tuple(dec_right_levels),
        )
