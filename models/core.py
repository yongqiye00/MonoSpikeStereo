import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor





def make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(8, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _make_norm2d(norm: str, ch: int, num_groups: int = 32) -> nn.Module:
    norm = norm.lower()
    if norm == "bn":
        return nn.BatchNorm2d(ch)
    if norm == "in":
        return nn.InstanceNorm2d(ch, affine=True, track_running_stats=False)
    # GroupNorm: choose the largest divisible group count.
    g = min(num_groups, ch)
    while g > 1 and (ch % g) != 0:
        g -= 1
    return nn.GroupNorm(g, ch)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, norm: str = "gn", gn_groups: int = 8
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.n1 = _make_norm2d(norm, out_channels, gn_groups)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.n2 = _make_norm2d(norm, out_channels, gn_groups)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _make_norm2d(norm, out_channels, gn_groups),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.gelu(self.n1(self.conv1(x)))
        out = self.n2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.gelu(out)


class ECA(nn.Module):
    def __init__(
        self, c: int, gamma: float = 2.0, b: float = 1.0, k_min: int = 3, k_max: int = 9
    ):
        super().__init__()
        # Use log2 for a more interpretable kernel estimate, with odd/range constraints.
        k = int(abs((math.log2(c) + b) / gamma))
        if k % 2 == 0:
            k += 1
        k = max(k_min, min(k, k_max))
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, H, W]
        y = x.mean(dim=(2, 3))  # [B, C]
        y = self.conv(y.unsqueeze(1)).squeeze(1)  # [B, C]
        y = y.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        return x * torch.sigmoid(y)


class EnhanceBlock(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 2.0, res_scale: float = 1.0):
        super().__init__()
        # 1) depthwise conv
        self.dwconv = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False)
        # 2) channel attention
        self.eca = ECA(dim)
        # 3) MLP
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, hidden, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1),
        )
        self.res_scale = res_scale  # Lower to 0.1-0.5 for a more conservative residual.

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.dwconv(x)
        x = self.eca(x)
        x = self.mlp(x)
        return identity + x * self.res_scale


class QueryGuidedTemporalFuser(nn.Module):
    def __init__(self, dim: int, mode: str = "self", num_heads: int = 4):
        super().__init__()
        assert mode in ("guided", "self"), "mode must be 'guided' or 'self'"
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.mode = mode
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

        use_norm = True
        self.use_norm = use_norm
        if use_norm:
            self.ln_hist = nn.LayerNorm(dim)
            self.ln_cur = nn.LayerNorm(dim)

    @staticmethod
    def sinusoid_pos_embed(t: int, c: int, device):
        if t == 0:
            return torch.zeros(0, c, device=device)
        position = torch.arange(t, device=device).float().unsqueeze(1)  # [t, 1]
        div_term = torch.exp(
            torch.arange(0, c, 2, device=device).float() * (-math.log(10000.0) / c)
        )
        pe = torch.zeros(t, c, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # [t, c]

    def forward(self, history: List[Tensor], current: Tensor) -> Tensor:
        # history: [B, C, H, W] × T, current: [B, C, H, W]
        B, C, H, W = current.shape
        T = len(history)
        if T == 0:
            return current

        # reshape
        hist = (
            torch.stack(history, dim=1).permute(0, 3, 4, 1, 2).contiguous()
        )  # [B, H, W, T, C]
        hist_flat = hist.view(B * H * W, T, C)  # [BHW, T, C]
        cur_flat = current.permute(0, 2, 3, 1).reshape(B * H * W, C)  # [BHW, C]

        if self.use_norm:
            hist_flat = self.ln_hist(hist_flat)
            cur_flat = self.ln_cur(cur_flat)
        if self.mode == "guided":
            # Use the current frame as query and history as key/value.
            pos = self.sinusoid_pos_embed(T, C, hist_flat.device)  # [T, C]
            hist_flat = hist_flat + pos.unsqueeze(0)

            q = self.q_proj(cur_flat).view(
                -1, self.num_heads, 1, self.head_dim
            )  # [BHW, h, 1, d]
            k = self.k_proj(hist_flat).view(
                -1, self.num_heads, T, self.head_dim
            )  # [BHW, h, T, d]
            v = self.v_proj(hist_flat).view(
                -1, self.num_heads, T, self.head_dim
            )  # [BHW, h, T, d]

            attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)  # [BHW, h, 1, T]
            attn = F.softmax(attn, dim=-1)
            # print(attn.mean(0).mean(1).detach().cpu().numpy())
            out = (attn @ v).squeeze(2).contiguous()  # [BHW, h, d]
            out = out.view(-1, self.num_heads * self.head_dim)  # [BHW, C]
            out = self.out_proj(out)  # [BHW, C]
        else:
            # Full temporal self-attention over history + current; keep the current output.
            pos = self.sinusoid_pos_embed(T + 1, C, hist_flat.device)  # [T+1, C]
            seq = torch.cat([hist_flat, cur_flat.unsqueeze(1)], dim=1)  # [BHW, T+1, C]
            seq = seq + pos.unsqueeze(0)

            q = (
                self.q_proj(seq)
                .view(-1, T + 1, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )  # [BHW,h,T+1,d]
            k = (
                self.k_proj(seq)
                .view(-1, T + 1, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )
            v = (
                self.v_proj(seq)
                .view(-1, T + 1, self.num_heads, self.head_dim)
                .permute(0, 2, 1, 3)
            )

            attn = (q @ k.transpose(-2, -1)) * (self.head_dim**-0.5)  # [BHW,h,T+1,T+1]
            attn = F.softmax(attn, dim=-1)
            out_all = attn @ v  # [BHW,h,T+1,d]
            out = out_all[:, :, -1, :].contiguous()  # Current-frame output [BHW,h,d].
            out = out.view(-1, self.num_heads * self.head_dim)  # [BHW, C]
            out = self.out_proj(out)  # [BHW, C]

            # print(f'attn {attn.mean(0).mean(1).detach().cpu().numpy()}')

        return out.view(B, H, W, C).permute(0, 3, 1, 2)  # [B, C, H, W]


class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, *grad_outputs):
        eps = ctx.eps
        grad_output = grad_outputs[0]

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1.0 / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return (
            gx,
            (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0),
            grad_output.sum(dim=3).sum(dim=2).sum(dim=0),
            None,
        )


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter("weight", nn.Parameter(torch.ones(channels)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SCAM(nn.Module):
    """
    Stereo Cross Attention Module (SCAM)
    """

    def __init__(self, c):
        super().__init__()
        self.scale = c**-0.5

        self.norm_l = LayerNorm2d(c)
        self.norm_r = LayerNorm2d(c)
        self.l_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(c, c, kernel_size=1, stride=1, padding=0)

    def forward(self, x_l, x_r):
        Q_l = self.l_proj1(self.norm_l(x_l)).permute(0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(self.norm_r(x_r)).permute(
            0, 2, 1, 3
        )  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_l).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_r).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(
            torch.softmax(attention.permute(0, 1, 3, 2), dim=-1), V_l
        )  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return x_l + F_r2l, x_r + F_l2r
