"""Separate audio into stems using PyTorch model with CoreML weights.

Usage:
    cd /path/to/Emagic-V-1.3
    pixi run python training/separate_pytorch.py input.wav [-o output_dir]
    pixi run python training/separate_pytorch.py input.wav --max-chunks 2
"""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import soxr
import numpy as np
import soundfile as sf
import torch

# Ensure training/ is on sys.path for direct script execution
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

SEGMENT_SAMPLES = 588800
SAMPLE_RATE = 44100
STEM_NAMES = ["bass", "drums", "other", "vocals", "guitar", "piano"]

P = lambda *a, **k: print(*a, **k, flush=True)


def main():
    parser = argparse.ArgumentParser(description="PyTorch BSRoformer stem separator")
    parser.add_argument("input", help="Input audio file")
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory")
    parser.add_argument("--overlap", type=float, default=0.5, help="Chunk overlap (0.0-0.9)")
    parser.add_argument("--weights", default=None, help="Path to mapped_state_dict.pt")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Max chunks to process (0=all). Useful for quick tests.")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        P(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Output dir
    if args.output_dir is None:
        args.output_dir = os.path.expanduser("~/Desktop/stems")
    os.makedirs(args.output_dir, exist_ok=True)

    # Weights
    weights_path = args.weights or os.path.join(_SCRIPT_DIR, "mapped_state_dict.pt")
    if not os.path.isfile(weights_path):
        P(f"Error: {weights_path} not found. Run convert_coreml_weights first.", file=sys.stderr)
        sys.exit(1)

    # Load audio
    P(f"Input: {args.input}")
    data, sr = sf.read(args.input, dtype="float32", always_2d=True)
    if sr != SAMPLE_RATE:
        data = soxr.resample(data, sr, SAMPLE_RATE)
        P(f"  Resampled {sr} -> {SAMPLE_RATE}")
    if data.shape[1] == 1:
        data = np.concatenate([data, data], axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    total_samples = data.shape[0]
    P(f"  {total_samples / SAMPLE_RATE:.1f}s, {total_samples} samples")

    # Load model
    from bsroformer import BSRoformer

    P("Loading model...")
    model = BSRoformer()
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.float().eval()

    # Chunk
    overlap = max(0.0, min(0.9, args.overlap))
    step = int(SEGMENT_SAMPLES * (1 - overlap))
    chunks, starts = [], []
    pos = 0
    while pos < total_samples:
        end = pos + SEGMENT_SAMPLES
        chunk = data[pos:end]
        if chunk.shape[0] < SEGMENT_SAMPLES:
            chunk = np.pad(chunk, ((0, SEGMENT_SAMPLES - chunk.shape[0]), (0, 0)), mode="reflect")
        chunks.append(chunk)
        starts.append(pos)
        if end >= total_samples:
            break
        pos += step

    # Limit chunks if requested
    if args.max_chunks > 0 and len(chunks) > args.max_chunks:
        P(f"  Limiting to {args.max_chunks}/{len(chunks)} chunks")
        chunks = chunks[:args.max_chunks]
        starts = starts[:args.max_chunks]
        # Adjust total_samples to cover only processed region
        last_end = min(starts[-1] + SEGMENT_SAMPLES, total_samples)
        total_samples = last_end

    # Crossfade window
    fade = np.ones(SEGMENT_SAMPLES, dtype=np.float32)
    fade_len = int(SEGMENT_SAMPLES * overlap)
    if fade_len > 0:
        fade[:fade_len] = np.linspace(0, 1, fade_len, dtype=np.float32)
        fade[-fade_len:] = np.linspace(1, 0, fade_len, dtype=np.float32)

    output = np.zeros((6, 2, total_samples), dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)

    n_chunks = len(chunks)
    P(f"Separating ({n_chunks} chunks, {overlap*100:.0f}% overlap)...")
    t0 = time.time()

    for i, (start, chunk) in enumerate(zip(starts, chunks)):
        tc = time.time()
        audio_tensor = torch.from_numpy(chunk.T).unsqueeze(0)

        with torch.no_grad():
            stft = model._stft(audio_tensor.float())
            x = model._flatten_stft(stft).half()
            x = model.band_split(x)

            time_cos, time_sin = model.time_rope(x.shape[1])
            band_cos, band_sin = model.band_rope(x.shape[2])

            for block in model.blocks:
                x = block(x, time_cos, time_sin, band_cos, band_sin)

            x = model.final_norm(x)
            masks_flat = model.mask_estimator(x)
            masks = model._unflatten_mask(masks_flat.float(), stft.shape[-1])
            separated = model._apply_complex_mask(stft, masks)

            stem_list = []
            for s in range(6):
                stem_list.append(model._istft(separated[:, s]))
            stems = torch.stack(stem_list, dim=1).squeeze(0).numpy()

        end = min(start + SEGMENT_SAMPLES, total_samples)
        length = end - start
        w = fade[:length]
        output[:, :, start:end] += stems[:, :, :length] * w[np.newaxis, np.newaxis, :]
        weight[start:end] += w

        elapsed = time.time() - tc
        eta = elapsed * (n_chunks - i - 1)
        P(f"  [{i+1}/{n_chunks}] {start/SAMPLE_RATE:.1f}s ({elapsed:.1f}s, ETA {eta:.0f}s)")

    weight = np.maximum(weight, 1e-8)
    output /= weight[np.newaxis, np.newaxis, :]

    P(f"  Total: {time.time() - t0:.1f}s")

    # Save
    P(f"Output: {args.output_dir}/")
    for i, name in enumerate(STEM_NAMES):
        out_path = os.path.join(args.output_dir, f"{name}.wav")
        sf.write(out_path, output[i].T, SAMPLE_RATE, subtype="FLOAT")
        rms = np.sqrt(np.mean(output[i] ** 2))
        P(f"  {name}.wav  RMS={rms:.6f}")

    P("Done.")


if __name__ == "__main__":
    main()
