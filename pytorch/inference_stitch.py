"""GPU-native overlap-add inference for BSRoformer.

Mirrors the chunking logic from cli.py but operates entirely on
GPU tensors with batched forward passes.
"""

from __future__ import annotations

import torch

from .bsroformer import BSRoformer

SEGMENT_SAMPLES = 588800


@torch.inference_mode()
def separate(
    model: BSRoformer,
    audio: torch.Tensor,
    chunk_size: int = SEGMENT_SAMPLES,
    overlap: float = 0.5,
    batch_size: int = 4,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Separate a full-length song into stems using overlap-add.

    Args:
        model:      BSRoformer model (already on device)
        audio:      [2, T] stereo waveform (any length, any device)
        chunk_size:  samples per chunk (default 588800)
        overlap:    overlap ratio 0.0–0.9 (default 0.5)
        batch_size: max chunks per forward pass
        device:     inference device

    Returns:
        [num_stems, 2, T] separated stems, same length as input
    """
    model.eval()
    audio = audio.to(device)
    T = audio.shape[-1]
    num_stems = model.num_stems

    step = int(chunk_size * (1 - overlap))

    # Build chunk list: (start_idx, padded_chunk)
    chunks = []
    starts = []
    pos = 0
    while pos < T:
        end = pos + chunk_size
        chunk = audio[:, pos:end]

        if chunk.shape[-1] < chunk_size:
            pad_len = chunk_size - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_len), mode="reflect")

        chunks.append(chunk)
        starts.append(pos)

        if end >= T:
            break
        pos += step

    # Build crossfade window
    fade = torch.ones(chunk_size, device=device)
    fade_len = int(chunk_size * overlap)
    if fade_len > 0:
        ramp_up = torch.linspace(0.0, 1.0, fade_len, device=device)
        ramp_down = torch.linspace(1.0, 0.0, fade_len, device=device)
        fade[:fade_len] = ramp_up
        fade[-fade_len:] = ramp_down

    # Allocate output
    output = torch.zeros(num_stems, 2, T, device=device)
    weight = torch.zeros(T, device=device)

    # Process in batches
    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch_chunks = torch.stack(chunks[batch_start:batch_end])  # [bs, 2, chunk_size]
        batch_starts = starts[batch_start:batch_end]

        # Forward pass
        batch_stems = model(batch_chunks)  # [bs, num_stems, 2, chunk_size]

        # Overlap-add
        for i, start in enumerate(batch_starts):
            end = min(start + chunk_size, T)
            length = end - start
            w = fade[:length]

            output[:, :, start:end] += batch_stems[i, :, :, :length] * w
            weight[start:end] += w

    # Normalize by accumulated weights
    weight = weight.clamp(min=1e-8)
    output = output / weight

    return output
