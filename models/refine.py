from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn

from .components import ABCModule, CBAModule
from .enc_dec import Encoder, SharedDecoder
from .temporal import RSFModule

Tensor = torch.Tensor
StateLevels = Sequence[Tensor]
StateSeq = Sequence[StateLevels]


class SMSNet(nn.Module):
    """SMS-Net variant that accepts multi-frame state history."""

    def __init__(
        self,
        base_ch: int = 32,
        kernel_size: int = 3,
        history_length: int = 3,
        detach_history_seq: bool = False,
    ) -> None:
        super().__init__()
        self.history_length = max(1, history_length)
        self.detach_history_seq = detach_history_seq

        self.pyramid_channels = (
            base_ch * 4,  # coarse
            base_ch * 2,  # mid
            base_ch,      # fine
        )
        coarse_ch, mid_ch, fine_ch = self.pyramid_channels

        self.encoder_left = Encoder(in_ch=1, base_ch=base_ch)
        self.encoder_right = Encoder(in_ch=1, base_ch=base_ch)
        self.encoder_ref = Encoder(in_ch=1, base_ch=base_ch)

        self.abc_modules_left = nn.ModuleList(
            [ABCModule(ch) for ch in self.pyramid_channels]
        )
        self.abc_modules_right = nn.ModuleList(
            [ABCModule(ch) for ch in self.pyramid_channels]
        )

        self.cba_module = CBAModule(
            coarse_ch=coarse_ch,
            mid_ch=mid_ch,
            fine_ch=fine_ch,
        )

        self.rsf_module = RSFModule(
            feat_ch=self.pyramid_channels,
            kernel_size=kernel_size,
            max_history=self.history_length,
        )
        self.shared_decoder = SharedDecoder(
            coarse_ch=coarse_ch,
            mid_ch=mid_ch,
            fine_ch=fine_ch,
            out_ch=1,
        )



    def _recent_history_steps(
        self,
        history_seq: StateSeq | None,
    ) -> list[Sequence[Tensor]]:
        """Return the most recent history steps up to `history_length`."""
        return list(history_seq)[-self.history_length :] if history_seq else []

    def _align_history_step(
        self,
        history_levels: Sequence[Tensor],
        current_levels: Sequence[Tensor],
    ) -> list[Tensor]:
        """Align one history step to current pyramid shapes."""
        for hist_lvl, cur_lvl in zip(history_levels, current_levels, strict=False):
            assert hist_lvl.shape[-2:] == cur_lvl.shape[-2:], "History/current feature shape mismatch"
        return [lvl.detach() if self.detach_history_seq else lvl for lvl in history_levels]

    def _prepare_state_sequence(
        self,
        state: StateSeq | None,
        current_levels: Sequence[Tensor],
    ) -> list[list[Tensor]]:
        """Build recurrent-core history input as `list[time][pyramid_level]`."""
        return [
            self._align_history_step(history_levels, current_levels)
            for history_levels in self._recent_history_steps(state)
        ]

    def forward(self, inputs: dict[str, Any]) -> dict[str, Any]:
        left_noisy = inputs["left"]
        right_noisy = inputs["right"]
        state_left_seq = inputs["state_left_seq"]
        state_right_seq = inputs["state_right_seq"]

        feat_left_pyramid = self.encoder_left(left_noisy)
        feat_right_pyramid = self.encoder_right(right_noisy)

        ref = inputs["ref"]
        ref_pyramid = self.encoder_ref(ref)

        for i in range(len(feat_left_pyramid)):
            feat_left_pyramid[i] = self.abc_modules_left[i](
                feat_left_pyramid[i], ref_pyramid[i]
            )

            feat_right_pyramid[i] = self.abc_modules_right[i](
                feat_right_pyramid[i], ref_pyramid[i]
            )

        feat_left_pyramid, feat_right_pyramid = self.cba_module(
            feat_left_pyramid, feat_right_pyramid
        )
        left_seq = self._prepare_state_sequence(state_left_seq, feat_left_pyramid)
        right_seq = self._prepare_state_sequence(state_right_seq, feat_right_pyramid)

        new_state_left, new_state_right, dec_left_levels, dec_right_levels = self.rsf_module(
            feat_left_pyramid,
            feat_right_pyramid,
            left_seq,
            right_seq,
        )

        refined_left = self.shared_decoder(
            dec_left_levels[0],
            dec_left_levels[1],
            dec_left_levels[2],
        )
        refined_right = self.shared_decoder(
            dec_right_levels[0],
            dec_right_levels[1],
            dec_right_levels[2],
        )

        outputs: dict[str, Any] = {
            "pred_a": refined_left,
            "pred_b": refined_right,
            "state_left": tuple(new_state_left),
            "state_right": tuple(new_state_right),
        }

        return outputs
