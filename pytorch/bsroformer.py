"""BSRoformer — Band-Split RoPE Transformer for music source separation.

Architecture faithfully reconstructed from CoreML MIL forensics of
ke77nsfms3_8024.mlmodelc (Apple Logic Pro stem splitter).

MIL-verified constants:
    dim=256, depth=12, heads=8, dim_head=64, ff_mult=4
    62 frequency bands, 6 stems, stereo, complex mask estimation
    STFT: n_fft=2048, hop_length=512, hann window
    Normalization: L2-norm * sqrt(dim) * gamma  (≡ RMSNorm)
    Attention: QKV fused, RoPE on Q/K, sigmoid per-head gating
    Residuals: fp32 accumulation
    Dual-axis: time attention first, then band attention
    Mask estimator: Linear→Tanh→Linear→GLU (sigmoid variant)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ---------------------------------------------------------------------------
# Constants from MIL forensics
# ---------------------------------------------------------------------------

DEFAULT_BAND_WIDTHS = (
    (8,) * 24
    + (16,) * 12
    + (48,) * 8
    + (96,) * 8
    + (192,) * 8
    + (512, 516)
)  # 62 bands, sum = 4100 = 1025 freq × 2 channels × 2 (real/imag)

STEM_NAMES = ["bass", "drums", "other", "vocals", "guitar", "piano"]


# ---------------------------------------------------------------------------
# Normalization  (MIL: reduce_l2_norm → scale(sqrt(dim)) → gamma)
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    """RMS normalization matching the MIL L2-norm pattern:
       x / ||x||_2 * sqrt(dim) * gamma
    which is equivalent to standard RMSNorm.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.norm(2, dim=-1, keepdim=True).clamp(min=self.eps)
        return (x / norm) * self.scale * self.gamma


# ---------------------------------------------------------------------------
# Rotary Positional Embeddings  (inline, no external dependency)
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputed RoPE sin/cos tables.

    Applied to pairs of dimensions in Q and K.  The MIL shows the standard
    rotate-half pattern: [-y, x] * sin + [x, y] * cos.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # [seq_len, dim//2]
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to tensor x of shape [..., seq_len, dim].

    MIL pattern: split dim into pairs, rotate each pair.
    """
    seq_len = x.shape[-2]
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)  # [1, 1, T, dim//2]
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)

    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    # Rotate: [x1, x2] → [x1*cos - x2*sin, x2*cos + x1*sin]
    rx1 = x1 * cos - x2 * sin
    rx2 = x2 * cos + x1 * sin
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


# ---------------------------------------------------------------------------
# Feed-Forward Network  (MIL: Linear→GELU(exact)→Linear)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner = dim * mult
        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Attention  (MIL: fused QKV, RoPE, sigmoid head gating, shared out bias)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """Multi-head attention with RoPE and sigmoid per-head gating.

    MIL verified:
        to_qkv:  Linear(256, 1536, bias=True)   — shared bias across layers
        to_out:   Linear(512, 256, bias=True)    — shared bias across layers
        to_gates: Linear(256, 8, bias=True)      — per-layer, from pre-norm input
        scale:    1/sqrt(64) = 0.125
    """

    def __init__(self, dim: int = 256, heads: int = 8, dim_head: int = 64):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = heads * dim_head  # 512

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=True)
        self.to_out = nn.Linear(inner_dim, dim, bias=True)
        self.to_gates = nn.Linear(dim, heads, bias=True)

        self.scale = dim_head ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   [B, N, dim]  where N is seq_len (time or bands)
            cos: [max_seq, dim_head//2] from RotaryEmbedding
            sin: [max_seq, dim_head//2] from RotaryEmbedding
        Returns:
            [B, N, dim]
        """
        B, N, _ = x.shape
        h = self.heads

        # Gate computation from pre-norm input (MIL: to_gates on input, not output)
        gates = self.to_gates(x)  # [B, N, h]
        gates = gates.sigmoid().unsqueeze(-1)  # [B, N, h, 1]

        # QKV
        qkv = self.to_qkv(x)  # [B, N, 3*inner]
        qkv = qkv.reshape(B, N, 3, h, self.dim_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, h, N, d]
        q, k, v = qkv.unbind(0)  # each [B, h, N, d]

        # Apply RoPE to Q and K
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Scaled dot-product attention (uses flash attention when available)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        # out: [B, h, N, d]

        # Apply sigmoid gates per head
        # gates: [B, N, h, 1] → [B, h, N, 1]
        out = out * gates.permute(0, 2, 1, 3)

        # Merge heads and project
        out = out.permute(0, 2, 1, 3).reshape(B, N, -1)  # [B, N, inner]
        return self.to_out(out)


# ---------------------------------------------------------------------------
# Transformer Block  (one depth level: time attn+FFN, then band attn+FFN)
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    """One depth level of dual-axis processing.

    MIL verified order:
        layers_{d}_0 = time attention  (each band attends across time)
        layers_{d}_1 = band attention  (each time step attends across bands)
    Each sub-block has: RMSNorm → Attention → fp32 residual → RMSNorm → FFN → fp32 residual
    """

    def __init__(self, dim: int = 256, heads: int = 8, dim_head: int = 64, ff_mult: int = 4):
        super().__init__()
        # Time axis sub-block
        self.time_norm_attn = RMSNorm(dim)
        self.time_attn = Attention(dim, heads, dim_head)
        self.time_norm_ff = RMSNorm(dim)
        self.time_ff = FeedForward(dim, ff_mult)

        # Band axis sub-block
        self.band_norm_attn = RMSNorm(dim)
        self.band_attn = Attention(dim, heads, dim_head)
        self.band_norm_ff = RMSNorm(dim)
        self.band_ff = FeedForward(dim, ff_mult)

    def forward(
        self,
        x: torch.Tensor,
        time_cos: torch.Tensor,
        time_sin: torch.Tensor,
        band_cos: torch.Tensor,
        band_sin: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: [B, T, num_bands, dim]
        Returns:
            [B, T, num_bands, dim]
        """
        B, T, F, D = x.shape

        # --- Time attention: each band attends across time frames ---
        # Reshape to [B*F, T, D]
        xt = x.permute(0, 2, 1, 3).reshape(B * F, T, D)

        # Attention with fp32 residual
        residual = xt.float()
        xt = self.time_norm_attn(xt)
        xt = self.time_attn(xt, time_cos, time_sin)
        xt = (residual + xt.float()).to(x.dtype)

        # FFN with fp32 residual
        residual = xt.float()
        xt = self.time_norm_ff(xt)
        xt = self.time_ff(xt)
        xt = (residual + xt.float()).to(x.dtype)

        # Reshape back to [B, T, F, D]
        x = xt.reshape(B, F, T, D).permute(0, 2, 1, 3)

        # --- Band attention: each time step attends across frequency bands ---
        # Reshape to [B*T, F, D]
        xf = x.reshape(B * T, F, D)

        # Attention with fp32 residual
        residual = xf.float()
        xf = self.band_norm_attn(xf)
        xf = self.band_attn(xf, band_cos, band_sin)
        xf = (residual + xf.float()).to(x.dtype)

        # FFN with fp32 residual
        residual = xf.float()
        xf = self.band_norm_ff(xf)
        xf = self.band_ff(xf)
        xf = (residual + xf.float()).to(x.dtype)

        # Reshape back to [B, T, F, D]
        return xf.reshape(B, T, F, D)


# ---------------------------------------------------------------------------
# Band Split  (MIL: per-band L2Norm→scale→gamma→Linear)
# ---------------------------------------------------------------------------

class BandSplit(nn.Module):
    """Split flattened spectrogram into frequency bands, project each to dim."""

    def __init__(self, dim: int, band_widths: tuple[int, ...]):
        super().__init__()
        self.dim = dim
        self.band_widths = band_widths
        self.norms = nn.ModuleList([RMSNorm(w) for w in band_widths])
        self.projections = nn.ModuleList([nn.Linear(w, dim) for w in band_widths])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, 4100]  (flattened freq × channels × real/imag)
        Returns:
            [B, T, num_bands, dim]
        """
        B, T, _ = x.shape
        bands = x.split(self.band_widths, dim=-1)
        out = x.new_empty(B, T, len(self.band_widths), self.dim)
        for i, (band, norm, proj) in enumerate(zip(bands, self.norms, self.projections)):
            out[:, :, i, :] = proj(norm(band))
        return out


# ---------------------------------------------------------------------------
# Mask Estimator  (MIL: per-stem, per-band: Linear→Tanh→Linear→GLU)
# ---------------------------------------------------------------------------

class MaskEstimator(nn.Module):
    """Estimate complex masks for all stems.

    MIL pattern per band per stem:
        Linear(dim, 1024) → Tanh → Linear(1024, band_width * 2) → split → sigmoid_gate
        The GLU variant: out = first_half * sigmoid(second_half)
    """

    def __init__(self, dim: int, band_widths: tuple[int, ...], num_stems: int, hidden: int = 1024):
        super().__init__()
        self.num_stems = num_stems
        self.band_widths = band_widths

        # One ModuleList per stem, each containing one MLP per band
        self.estimators = nn.ModuleList()
        for _ in range(num_stems):
            stem_mlps = nn.ModuleList()
            for bw in band_widths:
                stem_mlps.append(nn.Sequential(
                    nn.Linear(dim, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, bw * 2),  # *2 for GLU
                ))
            self.estimators.append(stem_mlps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, num_bands, dim]
        Returns:
            [B, num_stems, T, total_freq]  where total_freq = sum(band_widths) = 4100
        """
        B, T, num_bands, D = x.shape
        total_freq = sum(self.band_widths)
        out_masks = x.new_empty(B, self.num_stems, T, total_freq)

        # Unbind bands to avoid repeated slicing in the loop
        bands_feat = x.unbind(2)

        for s, stem_mlps in enumerate(self.estimators):
            offset = 0
            for i, mlp in enumerate(stem_mlps):
                bw = self.band_widths[i]
                out = mlp(bands_feat[i])  # [B, T, bw*2]
                # GLU: split in half, sigmoid gate
                a, b = out.chunk(2, dim=-1)
                out_masks[:, s, :, offset : offset + bw] = a * b.sigmoid()
                offset += bw

        return out_masks


# ---------------------------------------------------------------------------
# BSRoformer  (top-level model)
# ---------------------------------------------------------------------------

class BSRoformer(nn.Module):
    """Band-Split RoPE Transformer for 6-stem music source separation.

    Forward: [B, 2, 588800] → [B, 6, 2, 588800]

    Pipeline:
        1. STFT → complex spectrogram [B, 2, F, T] where F=1025, T=1151
        2. Flatten to [B, T, 4100] (freq × channels × real/imag interleaved)
        3. BandSplit → [B, T, 62, 256]
        4. 12× TransformerBlock (time attn + band attn per block)
        5. MaskEstimator → [B, 6, T, 4100] complex masks
        6. Reshape masks, apply via complex multiplication to STFT
        7. iSTFT → separated waveforms
    """

    def __init__(
        self,
        dim: int = 256,
        depth: int = 12,
        heads: int = 8,
        dim_head: int = 64,
        ff_mult: int = 4,
        num_stems: int = 6,
        n_fft: int = 2048,
        hop_length: int = 512,
        segment_samples: int = 588800,
        band_widths: tuple[int, ...] = DEFAULT_BAND_WIDTHS,
        mask_hidden: int = 1024,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.num_stems = num_stems
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.segment_samples = segment_samples
        self.band_widths = band_widths
        self.gradient_checkpointing = gradient_checkpointing

        # Derived constants
        self.n_freqs = n_fft // 2 + 1  # 1025
        # Time frames for exact segment length (with center=True padding)
        self.n_frames = segment_samples // hop_length + 1  # 1151
        self.total_freq_dim = sum(band_widths)  # 4100

        # STFT window
        self.register_buffer("window", torch.hann_window(n_fft))

        # RoPE embeddings for both axes
        self.time_rope = RotaryEmbedding(dim_head, max_seq_len=self.n_frames)
        self.band_rope = RotaryEmbedding(dim_head, max_seq_len=len(band_widths))

        # Band split
        self.band_split = BandSplit(dim, band_widths)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head, ff_mult)
            for _ in range(depth)
        ])

        # Final norm (before mask estimator, matches CoreML final_norm_gamma)
        self.final_norm = RMSNorm(dim)

        # Mask estimator
        self.mask_estimator = MaskEstimator(dim, band_widths, num_stems, mask_hidden)

    def _stft(self, audio: torch.Tensor) -> torch.Tensor:
        """Compute STFT for stereo audio.

        Args:
            audio: [B, 2, T]
        Returns:
            Complex tensor [B, 2, F, T_frames]
        """
        B, C, T = audio.shape
        x = audio.reshape(B * C, T)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )
        # spec: [B*C, F, T_frames]
        _, F, Tf = spec.shape
        return spec.reshape(B, C, F, Tf)

    def _istft(self, spec: torch.Tensor) -> torch.Tensor:
        """Inverse STFT for stereo complex spectrogram.

        Args:
            spec: [B, 2, F, T_frames] complex
        Returns:
            [B, 2, segment_samples]
        """
        B, C, F, Tf = spec.shape
        x = spec.reshape(B * C, F, Tf)
        audio = torch.istft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.n_fft,
            window=self.window,
            center=True,
            length=self.segment_samples,
        )
        return audio.reshape(B, C, -1)

    def _flatten_stft(self, spec: torch.Tensor) -> torch.Tensor:
        """Flatten complex STFT to the band-split input format.

        MIL verified reshape chain:
            [B, 2, 1025, 1151] complex
            → view_as_real → [B, 2, 1025, 1151, 2]
            → permute/reshape to [B, 1151, 4100]

        The 4100 = 1025 freq × 2 channels × 2 (real/imag)
        MIL flattening order (lines 38-46):
            [B, 2, F, T, 2] → transpose to [B, F, 2, T, 2]
            → reshape [B, 2050, T, 2] → transpose [B, T, 2050, 2]
            → reshape [B, T, 4100]
        So the order is: freq-major, then channel, then real/imag.
        """
        B, C, F, T = spec.shape
        # Convert to real representation
        x = torch.view_as_real(spec)  # [B, 2, F, T, 2]
        # MIL order: [B, F, C, T, ri] → [B, F*C, T, ri] → [B, T, F*C, ri] → [B, T, F*C*ri]
        x = x.permute(0, 2, 1, 3, 4)  # [B, F, C, T, 2]
        x = x.reshape(B, F * C, T, 2)  # [B, 2050, T, 2]
        x = x.permute(0, 2, 1, 3)  # [B, T, 2050, 2]
        x = x.reshape(B, T, F * C * 2)  # [B, T, 4100]
        return x

    def _unflatten_mask(self, mask: torch.Tensor, T: int) -> torch.Tensor:
        """Unflatten mask from [B, num_stems, T, 4100] to complex form.

        Returns [B, num_stems, 2, F, T, 2] where last dim is [real, imag]
        matching the STFT representation for complex multiplication.

        MIL (lines 9579-9597):
            [B, 6, T, 4100] → reshape [B, 6, T, 2050, 2]
            → transpose to [B, 6, 2050, T, 2]
        Then 2050 is split as F*C = 1025*2.
        """
        B, S, Tf, D = mask.shape
        F = self.n_freqs  # 1025
        C = 2  # stereo

        # Reverse the flattening
        x = mask.reshape(B, S, Tf, F * C, 2)  # [B, S, T, 2050, 2]
        x = x.permute(0, 1, 3, 2, 4)  # [B, S, 2050, T, 2]
        x = x.reshape(B, S, F, C, Tf, 2)  # [B, S, F, C, T, 2]
        x = x.permute(0, 1, 3, 2, 4, 5)  # [B, S, C, F, T, 2]
        return x

    def _apply_complex_mask(
        self, stft: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply complex mask via real-valued multiplication.

        MIL (lines 9604-9631):
            real_out = stft_real * mask_real - stft_imag * mask_imag
            imag_out = stft_real * mask_imag + stft_imag * mask_real

        Args:
            stft: [B, 2, F, T] complex
            mask: [B, num_stems, 2, F, T, 2] real (last dim = real/imag of mask)
        Returns:
            [B, num_stems, 2, F, T] complex
        """
        # Convert STFT to real representation
        stft_ri = torch.view_as_real(stft)  # [B, 2, F, T, 2]
        stft_r = stft_ri[..., 0]  # [B, 2, F, T]
        stft_i = stft_ri[..., 1]

        mask_r = mask[..., 0]  # [B, S, 2, F, T]
        mask_i = mask[..., 1]

        # Broadcast stft [B, 1, 2, F, T] × mask [B, S, 2, F, T]
        stft_r = stft_r.unsqueeze(1)
        stft_i = stft_i.unsqueeze(1)

        out_r = stft_r * mask_r - stft_i * mask_i
        out_i = stft_r * mask_i + stft_i * mask_r

        return torch.complex(out_r, out_i)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Args:
            audio: [B, 2, T] stereo waveform, T should be segment_samples (588800)
        Returns:
            [B, num_stems, 2, T] separated stereo stems
        """
        B = audio.shape[0]
        T_audio = audio.shape[2]

        # 1. STFT
        stft = self._stft(audio)  # [B, 2, 1025, 1151] complex

        # 2. Flatten for band split
        x = self._flatten_stft(stft)  # [B, 1151, 4100]

        # 3. Band split
        x = self.band_split(x)  # [B, 1151, 62, 256]

        T_frames = x.shape[1]
        num_bands = x.shape[2]

        # Get RoPE embeddings
        time_cos, time_sin = self.time_rope(T_frames)
        band_cos, band_sin = self.band_rope(num_bands)

        # 4. Transformer blocks
        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(
                    block, x, time_cos, time_sin, band_cos, band_sin,
                    use_reentrant=False,
                )
            else:
                x = block(x, time_cos, time_sin, band_cos, band_sin)

        # 5. Final norm
        x = self.final_norm(x)

        # 6. Mask estimation
        masks_flat = self.mask_estimator(x)  # [B, 6, 1151, 4100]

        # 6. Unflatten masks to complex form
        masks = self._unflatten_mask(masks_flat, T_frames)  # [B, 6, 2, 1025, 1151, 2]

        # 7. Apply complex masks
        separated = self._apply_complex_mask(stft, masks)  # [B, 6, 2, 1025, 1151] complex

        # 8. iSTFT per stem
        stems = audio.new_empty(B, self.num_stems, 2, self.segment_samples)
        for s in range(self.num_stems):
            stems[:, s] = self._istft(separated[:, s])

        return stems
