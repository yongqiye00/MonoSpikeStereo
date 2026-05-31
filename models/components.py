import math
import torch
import torch.nn as nn
from .core import SCAM

class ABCModule(nn.Module):
    """Affine brightness adjustment using right-view + input statistics."""

    def __init__(
        self,
        ch: int,
        reduction: int = 16,
        alpha_bounds: tuple[float, float] = (0.1, 10.0),
        beta_scale: float = 0.5,
        use_in_feat: bool = True,  # Whether to include in_feat statistics.
        use_std: bool = True,  # Whether to include channel std along with mean.
        identity_init: bool = True,  # Initialize close to identity behavior.
    ):
        super().__init__()
        self.ch = ch
        self.use_in_feat = use_in_feat
        self.use_std = use_std
        stats_factor = (2 if use_in_feat else 1) * (2 if use_std else 1)
        in_ch = ch * stats_factor
        hidden = max(1, in_ch // reduction)

        self.alpha_min, self.alpha_max = alpha_bounds
        self.beta_scale = float(beta_scale)
        self.affine = nn.Sequential(
            nn.Conv2d(in_ch, hidden, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, ch * 2, kernel_size=1, bias=True),
        )

        # identity-like init: set alpha≈1 and beta≈0 safely
        if identity_init:
            final = self.affine[-1]
            if isinstance(final, nn.Conv2d) and final.bias is not None:
                with torch.no_grad():
                    final.weight.zero_()
                    p = (1.0 - self.alpha_min) / (self.alpha_max - self.alpha_min)
                    p = float(min(max(p, 1e-6), 1 - 1e-6))
                    alpha_logit = math.log(p) - math.log(1.0 - p)
                    b = torch.zeros(
                        self.ch * 2,
                        dtype=final.bias.dtype,
                        device=final.bias.device,
                    )
                    b[: self.ch].fill_(alpha_logit)  # alpha -> 1
                    b[self.ch :].zero_()  # beta -> 0
                    final.bias.copy_(b)

        # Cache values for optional consistency/regularization losses during training.
        self._last_alpha = None
        self._last_beta = None
        self._last_in = None
        self._last_ref = None

    def forward(self, in_feat: torch.Tensor, ref_feat: torch.Tensor) -> torch.Tensor:
        # Compute global statistics.
        def stats(x):
            mean = x.mean(dim=(-2, -1), keepdim=True)
            if self.use_std:
                std = x.std(dim=(-2, -1), unbiased=False, keepdim=True)
                return torch.cat([mean, std], dim=1)
            return mean

        context_parts = [stats(ref_feat)]
        if self.use_in_feat:
            context_parts.append(stats(in_feat))
        context = torch.cat(context_parts, dim=1)  # [B, in_ch, 1, 1]
        alpha_beta = self.affine(context)
        alpha_raw, beta_raw = alpha_beta.chunk(2, dim=1)
        alpha = torch.sigmoid(alpha_raw)
        alpha = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha
        beta = torch.tanh(beta_raw) * self.beta_scale

        # Cache.
        self._last_alpha, self._last_beta = alpha, beta
        self._last_in, self._last_ref = in_feat, ref_feat

        return alpha * in_feat + beta

class CBAModule(nn.Module):
    """
    Collaborative Binocular Augment (CBA) module.
    - Apply one left/right SCAM per scale.
    - Fuse enhanced features back with learnable residual weights.
    all_scales=False: fine scale only.
    all_scales=True:  coarse/mid/fine scales.
    """

    def __init__(
        self,
        coarse_ch: int | None = None,
        mid_ch: int | None = None,
        fine_ch: int | None = None,
        all_scales: bool = True,
    ):
        super().__init__()
        self.all_scales = all_scales

        if coarse_ch is None or mid_ch is None or fine_ch is None:
            raise ValueError("CBAModule requires coarse_ch, mid_ch, and fine_ch.")

        # Per-scale SCAM and residual weights.
        self.cross_f = SCAM(c=fine_ch)
        self.w_f = nn.Parameter(torch.tensor(0.5))  # Residual weight after sigmoid.

        if self.all_scales:
            self.cross_m = SCAM(c=mid_ch)
            self.cross_c = SCAM(c=coarse_ch)
            self.w_m = nn.Parameter(torch.tensor(0.5))
            self.w_c = nn.Parameter(torch.tensor(0.5))

    def forward(self, feat_L_pyramid, feat_R_pyramid):
        L_coarse, L_mid, L_fine = feat_L_pyramid
        R_coarse, R_mid, R_fine = feat_R_pyramid

        # Fine scale: SCAM followed by residual fusion.
        L_f_enh, R_f_enh = self.cross_f(L_fine, R_fine)
        wf = torch.sigmoid(self.w_f)
        L_f_out = wf * L_f_enh + (1 - wf) * L_fine
        R_f_out = wf * R_f_enh + (1 - wf) * R_fine

        if not self.all_scales:
            return (L_coarse, L_mid, L_f_out), (R_coarse, R_mid, R_f_out)

        # Mid scale.
        L_m_enh, R_m_enh = self.cross_m(L_mid, R_mid)
        wm = torch.sigmoid(self.w_m)
        L_m_out = wm * L_m_enh + (1 - wm) * L_mid
        R_m_out = wm * R_m_enh + (1 - wm) * R_mid

        # Coarse scale.
        L_c_enh, R_c_enh = self.cross_c(L_coarse, R_coarse)
        wc = torch.sigmoid(self.w_c)
        L_c_out = wc * L_c_enh + (1 - wc) * L_coarse
        R_c_out = wc * R_c_enh + (1 - wc) * R_coarse

        return (L_c_out, L_m_out, L_f_out), (R_c_out, R_m_out, R_f_out)
