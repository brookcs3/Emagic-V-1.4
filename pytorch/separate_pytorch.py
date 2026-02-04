"""Separate audio into stems using a PyTorch BSRoformer model.

Supports:
- processing multiple input files in one run (batch)
- processing multiple chunks per forward pass (--chunk-batch-size)
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
import torch.nn.functional as F

# Ensure training/ is on sys.path for direct script execution
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

SEGMENT_SAMPLES = 588800
SAMPLE_RATE = 44100
STEM_NAMES = ["bass", "drums", "other", "vocals", "guitar", "piano"]
STEM_INDEX = {name: i for i, name in enumerate(STEM_NAMES)}

P = lambda *a, **k: print(*a, **k, flush=True)

def _read_batch_file(batch_file: str) -> list[str]:
    batch_file = os.path.abspath(os.path.expanduser(batch_file))
    base_dir = os.path.dirname(batch_file)
    paths: list[str] = []

    with open(batch_file, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            line = os.path.expandvars(os.path.expanduser(line))
            if not os.path.isabs(line):
                line = os.path.join(base_dir, line)
            paths.append(os.path.abspath(line))

    return paths


def _parse_stem_pair_value(spec: str) -> tuple[str, str, float]:
    # Format: "target:source=0.2"
    left, value_str = spec.split("=", 1)
    target, source = left.split(":", 1)
    return target.strip(), source.strip(), float(value_str.strip())


def _parse_stem_value(spec: str) -> tuple[str, float]:
    # Format: "stem=1.2"
    stem, value_str = spec.split("=", 1)
    return stem.strip(), float(value_str.strip())


def _smooth_1d(x: torch.Tensor, kernel: int, dim: int) -> torch.Tensor:
    if kernel <= 1:
        return x
    if kernel % 2 == 0:
        kernel += 1

    pad = kernel // 2
    weight = torch.ones(1, 1, kernel, device=x.device, dtype=x.dtype) / kernel

    if dim < 0:
        dim += x.ndim

    x_perm = x.movedim(dim, -1)
    orig_shape = x_perm.shape
    x_1d = x_perm.reshape(-1, 1, orig_shape[-1])
    x_out = F.conv1d(x_1d, weight, padding=pad)
    x_out = x_out.reshape(*orig_shape)
    return x_out.movedim(-1, dim)


def _forward_with_mask_controls(
    *,
    model,
    audio: torch.Tensor,
    mask_sharpness: float,
    competition: float,
    competition_temp: float,
    smooth_time: int,
    smooth_freq: int,
) -> torch.Tensor:
    """
    audio: [B, 2, T] float
    returns: [B, 6, 2, T] float
    """
    eps = 1e-8

    stft = model._stft(audio)  # [B, 2, F, Tframes] complex
    x = model._flatten_stft(stft).float()  # [B, Tframes, 4100]
    x = model.band_split(x)  # [B, Tframes, 62, dim]

    time_cos, time_sin = model.time_rope(x.shape[1])
    band_cos, band_sin = model.band_rope(x.shape[2])

    for block in model.blocks:
        x = block(x, time_cos, time_sin, band_cos, band_sin)

    x = model.final_norm(x)
    masks_flat = model.mask_estimator(x)  # [B, 6, Tframes, 4100]
    masks_ri = model._unflatten_mask(masks_flat.float(), x.shape[1])  # [B, 6, 2, F, Tframes, 2]

    m = torch.complex(masks_ri[..., 0], masks_ri[..., 1])  # [B, 6, 2, F, Tframes]
    mag = m.abs()
    phase = m / (mag + eps)

    # Mask sharpness: adjust mask magnitude while preserving phase.
    if mask_sharpness != 1.0:
        mag = mag.clamp_min(eps).pow(mask_sharpness)

    # Stem competition: redistribute magnitude across stems per TF bin.
    competition = float(max(0.0, min(1.0, competition)))
    if competition > 0.0:
        temp = float(max(0.1, competition_temp))
        mag_sum = mag.sum(dim=1, keepdim=True).clamp_min(eps)
        w = mag.clamp_min(eps).pow(temp)
        w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)
        mag_comp = w * mag_sum
        mag = (1.0 - competition) * mag + competition * mag_comp

    # Smoothing: reduce warble/artifacts by smoothing mask magnitude in time/freq.
    if smooth_time and smooth_time > 1:
        mag = _smooth_1d(mag, smooth_time, dim=-1)
    if smooth_freq and smooth_freq > 1:
        mag = _smooth_1d(mag, smooth_freq, dim=-2)

    m = phase * mag
    masks_ri = torch.stack((m.real, m.imag), dim=-1)

    separated = model._apply_complex_mask(stft, masks_ri)  # [B, 6, 2, F, Tframes] complex
    stems = []
    for s in range(6):
        stems.append(model._istft(separated[:, s]))  # [B, 2, T]
    return torch.stack(stems, dim=1)  # [B, 6, 2, T]


def _apply_stem_gains_and_cancels(
    stems: np.ndarray,
    stem_gains: list[str],
    cancels: list[str],
) -> np.ndarray:
    # stems: [B, 6, 2, T]
    out = stems

    for spec in stem_gains:
        stem, gain = _parse_stem_value(spec)
        if stem not in STEM_INDEX:
            raise ValueError(f"Unknown stem in --stem-gain: {stem} (expected one of {STEM_NAMES})")
        out[:, STEM_INDEX[stem]] *= float(gain)

    for spec in cancels:
        target, source, amount = _parse_stem_pair_value(spec)
        if target not in STEM_INDEX:
            raise ValueError(f"Unknown target stem in --cancel: {target} (expected one of {STEM_NAMES})")
        if source not in STEM_INDEX:
            raise ValueError(f"Unknown source stem in --cancel: {source} (expected one of {STEM_NAMES})")
        out[:, STEM_INDEX[target]] -= float(amount) * out[:, STEM_INDEX[source]]

    return out

def _resolve_output_dir(output_dir: str | None, input_path: str, multiple_inputs: bool) -> str:
    if output_dir is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(os.path.dirname(os.path.abspath(input_path)), f"{base}_stems_pytorch")

    output_dir = os.path.abspath(output_dir)
    if not multiple_inputs:
        return output_dir

    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}_stems_pytorch")


def _load_audio(path: str, target_sr: int | None) -> tuple[np.ndarray, int, int]:
    data, file_sr = sf.read(path, dtype="float32", always_2d=True)

    if data.shape[1] == 1:
        data = np.concatenate([data, data], axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    if target_sr is None or file_sr == target_sr:
        return data, file_sr, file_sr

    data = soxr.resample(data, file_sr, target_sr)
    return data, file_sr, target_sr


def _separate_one(
    *,
    model,
    input_path: str,
    output_dir: str,
    overlap: float,
    max_chunks: int,
    chunk_batch_size: int,
    sr_mode: str,
    model_sr: int,
    mask_sharpness: float,
    competition: float,
    competition_temp: float,
    smooth_time: int,
    smooth_freq: int,
    stem_gain: list[str],
    cancel: list[str],
) -> None:
    if not os.path.isfile(input_path):
        P(f"Error: {input_path} not found", file=sys.stderr)
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    # Load audio
    P(f"Input: {input_path}")
    target_sr = None if sr_mode == "source" else int(model_sr)
    data, file_sr, used_sr = _load_audio(input_path, target_sr=target_sr)
    if sr_mode != "source" and file_sr != used_sr:
        P(f"  Resampled {file_sr} -> {used_sr}")
    else:
        P(f"  Sample rate: {used_sr}")
    total_samples = data.shape[0]
    P(f"  {total_samples / used_sr:.1f}s, {total_samples} samples")

    # Chunk
    overlap = max(0.0, min(0.9, overlap))
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
    if max_chunks > 0 and len(chunks) > max_chunks:
        P(f"  Limiting to {max_chunks}/{len(chunks)} chunks")
        chunks = chunks[:max_chunks]
        starts = starts[:max_chunks]
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
    chunk_batch_size = max(1, int(chunk_batch_size))
    P(f"Separating ({n_chunks} chunks, {overlap*100:.0f}% overlap, chunk_batch={chunk_batch_size})...")
    t_total = time.time()

    use_mask_controls = (
        mask_sharpness != 1.0
        or competition != 0.0
        or smooth_time > 1
        or smooth_freq > 1
    )
    use_mix_controls = bool(stem_gain) or bool(cancel)

    with torch.no_grad():
        for i0 in range(0, n_chunks, chunk_batch_size):
            i1 = min(i0 + chunk_batch_size, n_chunks)
            batch_chunks = chunks[i0:i1]
            batch_starts = starts[i0:i1]

            tc = time.time()
            batch_audio = np.stack([c.T for c in batch_chunks], axis=0)  # [B, 2, T]
            audio_tensor = torch.from_numpy(batch_audio).float()

            if use_mask_controls:
                stems_t = _forward_with_mask_controls(
                    model=model,
                    audio=audio_tensor,
                    mask_sharpness=mask_sharpness,
                    competition=competition,
                    competition_temp=competition_temp,
                    smooth_time=smooth_time,
                    smooth_freq=smooth_freq,
                )
            else:
                stems_t = model(audio_tensor)

            stems_batch = stems_t.cpu().numpy()  # [B, 6, 2, T]
            if use_mix_controls:
                stems_batch = _apply_stem_gains_and_cancels(stems_batch, stem_gain, cancel)

            for j, start in enumerate(batch_starts):
                end = min(start + SEGMENT_SAMPLES, total_samples)
                length = end - start
                w = fade[:length]
                stems = stems_batch[j]
                output[:, :, start:end] += stems[:, :, :length] * w[np.newaxis, np.newaxis, :]
                weight[start:end] += w

            elapsed = time.time() - tc
            done = i1
            remaining = n_chunks - done
            eta = (elapsed / max(1, (i1 - i0))) * remaining
            P(f"  [{done}/{n_chunks}] +{i1-i0} chunks ({elapsed:.1f}s, ETA {eta:.0f}s)")

    weight = np.maximum(weight, 1e-8)
    output /= weight[np.newaxis, np.newaxis, :]

    P(f"  Total: {time.time() - t_total:.1f}s")

    out_audio = output
    out_sr = used_sr
    if sr_mode != "source" and file_sr != used_sr:
        # Resample stems back to the file's original SR for correct tempo/pitch.
        resampled: list[np.ndarray] = []
        for s in range(6):
            stem_tc = out_audio[s].T  # [T, 2]
            stem_tc = soxr.resample(stem_tc, used_sr, file_sr)
            resampled.append(stem_tc.T)  # [2, T']
        out_audio = np.stack(resampled, axis=0).astype(np.float32)  # [6, 2, T']
        out_sr = file_sr

    # Save
    P(f"Output: {output_dir}/")
    for i, name in enumerate(STEM_NAMES):
        out_path = os.path.join(output_dir, f"{name}.wav")
        sf.write(out_path, out_audio[i].T, out_sr, subtype="FLOAT")
        rms = np.sqrt(np.mean(out_audio[i] ** 2))
        P(f"  {name}.wav  RMS={rms:.6f}")

    P("Done.")


def main():
    parser = argparse.ArgumentParser(description="PyTorch BSRoformer stem separator")
    parser.add_argument("inputs", nargs="*", help="Input audio file(s)")
    parser.add_argument(
        "--batch-file",
        default=None,
        metavar="PATH",
        help="Text file with one input path per line (blank lines and # comments ignored).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail fast on the first missing/unreadable input (default: skip and continue).",
    )
    parser.add_argument("-o", "--output-dir", default=None, help="Output directory (or base dir for multiple inputs)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Chunk overlap (0.0-0.9)")
    parser.add_argument("--weights", default=None, help="Path to mapped_state_dict.pt")
    parser.add_argument("--sr-mode", choices=["source", "model"], default="source",
                        help="Sample-rate handling: source=no resample; model=resample to --model-sr and back.")
    parser.add_argument("--model-sr", type=int, default=44100,
                        help="When --sr-mode model, resample audio to this SR for separation (default: 44100).")
    parser.add_argument("--max-chunks", type=int, default=0,
                        help="Max chunks to process (0=all). Useful for quick tests.")
    parser.add_argument("--chunk-batch-size", type=int, default=1,
                        help="How many chunks to run per forward pass (default: 1).")
    parser.add_argument("--mask-sharpness", type=float, default=1.0,
                        help=">1.0 reduces bleed but adds artifacts; <1.0 softens separation (default: 1.0).")
    parser.add_argument("--competition", type=float, default=0.0,
                        help="0..1: make stems compete per TF bin (reduces bleed, may add artifacts).")
    parser.add_argument("--competition-temp", type=float, default=1.0,
                        help="Competition temperature (>1 more aggressive).")
    parser.add_argument("--smooth-time", type=int, default=0,
                        help="Smooth mask magnitude over time (frames). Odd kernel recommended.")
    parser.add_argument("--smooth-freq", type=int, default=0,
                        help="Smooth mask magnitude over frequency bins. Odd kernel recommended.")
    parser.add_argument("--stem-gain", action="append", default=[],
                        help="Repeatable: e.g. --stem-gain bass=1.1")
    parser.add_argument("--cancel", action="append", default=[],
                        help="Repeatable: subtract a bit of one stem from another, e.g. --cancel bass:other=0.2")
    args = parser.parse_args()

    inputs: list[str] = []
    if args.batch_file:
        try:
            inputs.extend(_read_batch_file(args.batch_file))
        except OSError as e:
            P(f"Error: failed to read batch file: {args.batch_file} ({e})", file=sys.stderr)
            sys.exit(1)
    inputs.extend(args.inputs or [])

    if not inputs:
        P("Error: provide at least one input path or --batch-file", file=sys.stderr)
        sys.exit(2)

    # Weights
    weights_path = args.weights or os.path.join(_SCRIPT_DIR, "mapped_state_dict.pt")
    if not os.path.isfile(weights_path):
        P(f"Error: {weights_path} not found. Run convert_coreml_weights first.", file=sys.stderr)
        sys.exit(1)

    # Load model
    from bsroformer import BSRoformer

    P("Loading model...")
    model = BSRoformer()
    sd = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=False)
    model.float().eval()
    multiple_inputs = len(inputs) > 1
    for idx, input_path in enumerate(inputs):
        if not os.path.isfile(input_path):
            P(f"Error: {input_path} not found", file=sys.stderr)
            if args.strict:
                sys.exit(1)
            continue

        if multiple_inputs:
            print(f"\n=== [{idx+1}/{len(inputs)}] {os.path.basename(input_path)} ===")
        out_dir = _resolve_output_dir(args.output_dir, input_path, multiple_inputs=multiple_inputs)
        _separate_one(
            model=model,
            input_path=input_path,
            output_dir=out_dir,
            overlap=args.overlap,
            max_chunks=args.max_chunks,
            chunk_batch_size=args.chunk_batch_size,
            sr_mode=args.sr_mode,
            model_sr=args.model_sr,
            mask_sharpness=args.mask_sharpness,
            competition=args.competition,
            competition_temp=args.competition_temp,
            smooth_time=args.smooth_time,
            smooth_freq=args.smooth_freq,
            stem_gain=args.stem_gain,
            cancel=args.cancel,
        )


if __name__ == "__main__":
    main()
