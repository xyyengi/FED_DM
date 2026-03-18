"""
Seasonally Conditional Decomposed Diffusion Model (SDCDM)
==========================================================
Implements the conditional diffusion model described in §3.2 of the paper.

Architecture
------------
A 1-D temporal denoising network (UNet-style with residual blocks) that takes:
  - noisy input  : x_t  of shape (B, pred_len)
  - noise level  : σ    scalar per sample
  - condition    : [T, S, R] – trend/seasonal/residual from STL decomposition
                    of the FEDformer forecast, each (B, pred_len)

The network outputs the denoised estimate x̂₀ (or equivalently the noise ε).
We use the Karras et al. (2022) pre-conditioning (c_skip, c_out, c_in).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ───────────────────────── helpers ─────────────────────────────────────────

def sinusoidal_embedding(sigma: torch.Tensor, dim: int) -> torch.Tensor:
    """Map noise level σ → embedding vector of size `dim`."""
    device   = sigma.device
    half     = dim // 2
    freqs    = torch.exp(
        -math.log(10000) * torch.arange(half, dtype=torch.float32, device=device) / (half - 1)
    )
    # log σ so the embedding spans a reasonable range
    args     = torch.log(sigma.float().clamp(min=1e-6)).unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)   # (B, dim)


# ───────────────────────── building blocks ─────────────────────────────────

class ResBlock1d(nn.Module):
    """1-D residual block with adaptive group normalisation conditioned on σ."""

    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        # affine parameters from condition (noise level + spatial cond)
        self.cond_proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.cond_proj.weight)
        nn.init.zeros_(self.cond_proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L), cond: (B, cond_dim)
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)   # (B, C)
        h = self.norm1(x)
        h = h * (scale.unsqueeze(-1) + 1) + shift.unsqueeze(-1)
        h = F.silu(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = F.silu(h)
        h = self.conv2(h)
        return x + h


class DownBlock1d(nn.Module):
    def __init__(self, in_c: int, out_c: int, cond_dim: int, n_res: int = 2):
        super().__init__()
        self.in_proj  = nn.Conv1d(in_c, out_c, 1)
        self.res      = nn.ModuleList([ResBlock1d(out_c, cond_dim) for _ in range(n_res)])
        self.downsamp = nn.Conv1d(out_c, out_c, 3, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.in_proj(x)
        for r in self.res:
            x = r(x, cond)
        skip = x
        x    = self.downsamp(x)
        return x, skip


class UpBlock1d(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, cond_dim: int, n_res: int = 2):
        super().__init__()
        self.upsamp   = nn.ConvTranspose1d(in_c, in_c, 4, stride=2, padding=1)
        self.in_proj  = nn.Conv1d(in_c + skip_c, out_c, 1)
        self.res      = nn.ModuleList([ResBlock1d(out_c, cond_dim) for _ in range(n_res)])

    def forward(self, x, skip, cond):
        x = self.upsamp(x)
        # align lengths (rounding from odd lengths)
        if x.shape[-1] != skip.shape[-1]:
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.in_proj(x)
        for r in self.res:
            x = r(x, cond)
        return x


class MidBlock1d(nn.Module):
    def __init__(self, channels: int, cond_dim: int):
        super().__init__()
        self.res1 = ResBlock1d(channels, cond_dim)
        self.res2 = ResBlock1d(channels, cond_dim)

    def forward(self, x, cond):
        x = self.res1(x, cond)
        x = self.res2(x, cond)
        return x


# ───────────────────────── main SDCDM network ──────────────────────────────

class SDCDMNet(nn.Module):
    """
    Denoising network for SDCDM.

    Input channels per time step:
      1   noisy target
      3   condition (T, S, R) ← STL decomposition of FEDformer forecast

    Karras pre-conditioning (c_skip / c_out / c_in) is applied externally
    by the SDCDMDenoiser wrapper below.
    """

    def __init__(
        self,
        sigma_data : float = 1.0,
        base_channels : int = 64,
        channel_mults : tuple = (1, 2, 4),
        sigma_emb_dim : int = 128,
        cond_channels : int = 3,   # T, S, R
        input_channels: int = 1,
    ):
        super().__init__()
        self.sigma_data = sigma_data

        # input: noisy + T + S + R = 1 + 3 = 4 channels
        in_ch  = input_channels + cond_channels
        ch0    = base_channels
        ch_seq = [ch0 * m for m in channel_mults]

        # sigma embedding → condition vector
        self.sigma_mlp = nn.Sequential(
            nn.Linear(sigma_emb_dim, sigma_emb_dim * 2),
            nn.SiLU(),
            nn.Linear(sigma_emb_dim * 2, sigma_emb_dim),
        )
        self.sigma_emb_dim = sigma_emb_dim
        # cond_dim is the sigma embedding
        cond_dim = sigma_emb_dim

        # encoder / UNet down-path
        self.in_conv  = nn.Conv1d(in_ch, ch_seq[0], 3, padding=1)
        self.downs    = nn.ModuleList()
        for i in range(len(ch_seq) - 1):
            self.downs.append(DownBlock1d(ch_seq[i], ch_seq[i+1], cond_dim))

        # bottleneck
        self.mid = MidBlock1d(ch_seq[-1], cond_dim)

        # decoder / UNet up-path
        self.ups = nn.ModuleList()
        for i in range(len(ch_seq) - 1, 0, -1):
            # skip from the corresponding down-block has ch_seq[i] channels (= out_c of that block)
            self.ups.append(UpBlock1d(ch_seq[i], ch_seq[i], ch_seq[i-1], cond_dim))

        self.out_norm = nn.GroupNorm(8, ch_seq[0])
        self.out_conv = nn.Conv1d(ch_seq[0], input_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x_noisy : torch.Tensor,   # (B, 1, L)
        sigma   : torch.Tensor,   # (B,)
        cond_T  : torch.Tensor,   # (B, 1, L)
        cond_S  : torch.Tensor,
        cond_R  : torch.Tensor,
    ) -> torch.Tensor:

        # ── condition vector (sigma embedding) ──────────────────────────
        sigma_emb = sinusoidal_embedding(sigma, self.sigma_emb_dim)
        sigma_emb = self.sigma_mlp(sigma_emb)                          # (B, cond_dim)

        # ── concatenate spatial condition along channel dim ──────────────
        x = torch.cat([x_noisy, cond_T, cond_S, cond_R], dim=1)       # (B, 4, L)
        x = self.in_conv(x)

        # ── down ────────────────────────────────────────────────────────
        skips = []
        for down in self.downs:
            x, skip = down(x, sigma_emb)
            skips.append(skip)

        # ── mid ─────────────────────────────────────────────────────────
        x = self.mid(x, sigma_emb)

        # ── up ──────────────────────────────────────────────────────────
        for up, skip in zip(self.ups, reversed(skips)):
            x = up(x, skip, sigma_emb)

        x = self.out_norm(x)
        x = F.silu(x)
        return self.out_conv(x)                                        # (B, 1, L)


# ───────────────────────── Karras pre-conditioned denoiser ─────────────────

class SDCDMDenoiser(nn.Module):
    """
    Wraps SDCDMNet with the Karras et al. (2022) pre-conditioning:
        c_skip  = σ_data² / (σ² + σ_data²)
        c_out   = σ · σ_data / (σ² + σ_data²)^0.5
        c_in    = 1 / (σ² + σ_data²)^0.5
    and exposes a .loss() method for training.
    """

    def __init__(self, inner: SDCDMNet, sigma_data: float = 1.0):
        super().__init__()
        self.inner      = inner
        self.sigma_data = sigma_data

    def get_scalings(self, sigma: torch.Tensor):
        sd2    = self.sigma_data ** 2
        c_skip = sd2 / (sigma ** 2 + sd2)
        c_out  = sigma * self.sigma_data / (sigma ** 2 + sd2) ** 0.5
        c_in   = 1.0 / (sigma ** 2 + sd2) ** 0.5
        return c_skip, c_out, c_in

    def _expand(self, v: torch.Tensor, ndim: int) -> torch.Tensor:
        return v.view(v.shape + (1,) * (ndim - v.ndim))

    def forward(
        self,
        x     : torch.Tensor,   # (B, 1, L) – noisy input
        sigma : torch.Tensor,   # (B,)
        cond_T: torch.Tensor,
        cond_S: torch.Tensor,
        cond_R: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in = self.get_scalings(sigma)
        x_in    = x * self._expand(c_in,   x.ndim)
        raw_out = self.inner(x_in, sigma, cond_T, cond_S, cond_R)
        return raw_out * self._expand(c_out, x.ndim) + x * self._expand(c_skip, x.ndim)

    def loss(
        self,
        x0    : torch.Tensor,   # (B, 1, L) clean target
        noise : torch.Tensor,   # (B, 1, L) sampled ε ~ N(0,1)
        sigma : torch.Tensor,   # (B,)
        cond_T: torch.Tensor,
        cond_S: torch.Tensor,
        cond_R: torch.Tensor,
    ) -> torch.Tensor:
        c_skip, c_out, c_in = self.get_scalings(sigma)
        noised = x0 + noise * self._expand(sigma, x0.ndim)
        x_in   = noised * self._expand(c_in, x0.ndim)
        pred   = self.inner(x_in, sigma, cond_T, cond_S, cond_R)
        # noise-prediction target
        target = (x0 - self._expand(c_skip, x0.ndim) * noised) / self._expand(c_out, x0.ndim)
        return ((pred - target) ** 2).mean(dim=[1, 2])   # (B,)
