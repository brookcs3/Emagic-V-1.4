"""Model factory: creation, initialization, and checkpoint utilities."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import yaml

from .bsroformer import BSRoformer, DEFAULT_BAND_WIDTHS


def create_model(config: dict | str) -> BSRoformer:
    """Create a fresh BSRoformer from config dict or YAML path.

    Args:
        config: dict of model params, or path to config.yaml

    Returns:
        Initialized BSRoformer model.
    """
    if isinstance(config, str):
        with open(config) as f:
            config = yaml.safe_load(f)

    model_cfg = config.get("model", config)

    band_widths = model_cfg.get("band_widths", None)
    if band_widths is not None:
        band_widths = tuple(band_widths)
    else:
        band_widths = DEFAULT_BAND_WIDTHS

    model = BSRoformer(
        dim=model_cfg.get("dim", 256),
        depth=model_cfg.get("depth", 12),
        heads=model_cfg.get("heads", 8),
        dim_head=model_cfg.get("dim_head", 64),
        ff_mult=model_cfg.get("ff_mult", 4),
        num_stems=model_cfg.get("num_stems", 6),
        n_fft=model_cfg.get("n_fft", 2048),
        hop_length=model_cfg.get("hop_length", 512),
        segment_samples=model_cfg.get("segment_samples", 588800),
        band_widths=band_widths,
        mask_hidden=model_cfg.get("mask_hidden", 1024),
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
    )

    _init_weights(model)
    return model


def _init_weights(model: BSRoformer) -> None:
    """Initialize weights following transformer best practices.

    - Xavier uniform for attention projections (QKV, output)
    - Kaiming normal for FFN and mask estimator linears
    - Zeros for all biases
    - Zeros for gate biases (sigmoid(0)=0.5, neutral start)
    - Ones for norm gamma parameters
    """
    for name, param in model.named_parameters():
        if param.dim() < 2:
            # Bias or 1D param
            if "gamma" in name:
                nn.init.ones_(param)
            elif "bias" in name and "to_gates" in name:
                nn.init.zeros_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            continue

        # 2D+ parameters (weight matrices)
        if "to_qkv" in name or "to_out" in name:
            nn.init.xavier_uniform_(param)
        elif "to_gates" in name:
            nn.init.xavier_uniform_(param, gain=0.1)
        else:
            nn.init.kaiming_normal_(param, nonlinearity="linear")


def model_summary(model: BSRoformer) -> str:
    """Return a human-readable model summary."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    buffers = sum(b.numel() for b in model.buffers())

    mem_fp32 = total * 4 / 1e9
    mem_bf16 = total * 2 / 1e9

    lines = [
        f"BSRoformer Summary",
        f"  dim:       {model.dim}",
        f"  depth:     {model.depth}",
        f"  stems:     {model.num_stems}",
        f"  bands:     {len(model.band_widths)}",
        f"  n_fft:     {model.n_fft}",
        f"  hop:       {model.hop_length}",
        f"  segment:   {model.segment_samples} samples ({model.segment_samples/44100:.2f}s)",
        f"  ---",
        f"  Total params:     {total:>12,}",
        f"  Trainable params: {trainable:>12,}",
        f"  Buffers:          {buffers:>12,}",
        f"  Memory (fp32):    {mem_fp32:.2f} GB",
        f"  Memory (bf16):    {mem_bf16:.2f} GB",
    ]
    return "\n".join(lines)


def save_checkpoint(
    model: BSRoformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    loss: float,
    path: str,
) -> None:
    """Save training checkpoint."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
            "loss": loss,
        },
        path,
    )


def load_checkpoint(
    path: str,
    config: dict | str,
    device: str = "cpu",
) -> tuple[BSRoformer, dict[str, Any]]:
    """Load model from checkpoint.

    Returns:
        (model, checkpoint_dict) where checkpoint_dict has optimizer state, epoch, etc.
    """
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = create_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt
