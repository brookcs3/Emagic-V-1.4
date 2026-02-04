#!/usr/bin/env python3
"""
BSRoformer CoreML inference for music source separation.

Separates a stereo audio file into 6 stems using a compiled
CoreML BSRoformer model (.mlmodelc) via coremltools.CompiledMLModel.
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore", category=UserWarning)

import coremltools as ct
import soundfile as sf
import torch

SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 588800  # ~13.35 seconds
N_FFT = 2048
HOP_LENGTH = 512
N_STEMS = 6
OVERLAP = 0.5

STEM_NAMES = ["stem_0", "stem_1", "stem_2", "stem_3", "stem_4", "stem_5"]


def load_audio(path: str) -> np.ndarray:
    """Load audio file, return stereo float32 at 44100 Hz."""
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    if sr != SAMPLE_RATE:
        try:
            import torchaudio
            waveform = torch.from_numpy(data.T)
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            data = waveform.numpy().T
            print(f"  Resampled {sr} -> {SAMPLE_RATE} Hz")
        except ImportError:
            print(f"  WARNING: Audio is {sr} Hz, model expects {SAMPLE_RATE} Hz.")
            print(f"  Install torchaudio for resampling: pixi add --pypi torchaudio")

    if data.shape[1] == 1:
        data = np.concatenate([data, data], axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    return data


def make_chunks(audio: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """Split audio into overlapping chunks of SEGMENT_SAMPLES."""
    total_samples = audio.shape[0]
    step = int(SEGMENT_SAMPLES * (1 - OVERLAP))
    chunks = []

    start = 0
    while start < total_samples:
        end = start + SEGMENT_SAMPLES
        chunk = audio[start:end]

        if chunk.shape[0] < SEGMENT_SAMPLES:
            pad_len = SEGMENT_SAMPLES - chunk.shape[0]
            chunk = np.pad(chunk, ((0, pad_len), (0, 0)), mode="reflect")

        chunks.append((start, chunk))

        if end >= total_samples:
            break
        start += step

    return chunks


def run_model(model, chunk: np.ndarray) -> np.ndarray:
    """Run CoreML model on a single [SEGMENT_SAMPLES, 2] chunk.

    Returns [N_STEMS, 2, SEGMENT_SAMPLES] float32 audio.
    """
    input_data = chunk.T[np.newaxis, :, :].astype(np.float32)
    result = model.predict({"input": input_data})

    real_part = result["var_11707"]  # [1, 12, 1151, 1025]
    imag_part = result["var_11751"]

    complex_spec = torch.from_numpy(real_part) + 1j * torch.from_numpy(imag_part)
    complex_spec = complex_spec.squeeze(0).reshape(N_STEMS, 2, 1151, 1025)

    window = torch.hann_window(N_FFT)
    stems = []
    for s in range(N_STEMS):
        channels = []
        for c in range(2):
            spec = complex_spec[s, c].T  # [1025, 1151]
            audio_out = torch.istft(
                spec,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=N_FFT,
                window=window,
                center=True,
                length=SEGMENT_SAMPLES,
            )
            channels.append(audio_out)
        stems.append(torch.stack(channels, dim=0))

    return torch.stack(stems, dim=0).numpy()


def overlap_add(chunks_results: list[tuple[int, np.ndarray]], total_samples: int) -> np.ndarray:
    """Combine overlapping chunks with linear crossfade."""
    output = np.zeros((N_STEMS, 2, total_samples), dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)

    fade = np.ones(SEGMENT_SAMPLES, dtype=np.float32)
    fade_len = int(SEGMENT_SAMPLES * OVERLAP)
    fade[:fade_len] = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade[-fade_len:] = np.linspace(1, 0, fade_len, dtype=np.float32)

    for start, stems in chunks_results:
        end = min(start + SEGMENT_SAMPLES, total_samples)
        length = end - start
        w = fade[:length]
        output[:, :, start:end] += stems[:, :, :length] * w[np.newaxis, np.newaxis, :]
        weight[start:end] += w

    weight = np.maximum(weight, 1e-8)
    output /= weight[np.newaxis, np.newaxis, :]
    return output


def main():
    parser = argparse.ArgumentParser(description="Separate audio into stems using BSRoformer CoreML model")
    parser.add_argument("inputs", nargs="*", help="Input audio file path(s)")
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
    parser.add_argument("--model", default=os.path.normpath(os.path.join(os.path.dirname(__file__), "model.mlmodelc")),
                        help="Path to compiled CoreML model (.mlmodelc)")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (default: <input_name>_stems/ next to input)")
    parser.add_argument("--stems", nargs="+", default=None,
                        help="Custom stem names (6 names)")
    parser.add_argument("--compute-units", choices=["cpu", "gpu", "all"], default="gpu",
                        help="CoreML compute units (default: gpu)")
    args = parser.parse_args()

    compute_map = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }

    stem_names = args.stems if args.stems and len(args.stems) == N_STEMS else STEM_NAMES

    def read_batch_file(batch_file: str) -> list[str]:
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

    print(f"Loading model ({args.compute_units})...")
    t0 = time.time()
    model = ct.models.CompiledMLModel(
        args.model,
        compute_units=compute_map[args.compute_units],
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    inputs: list[str] = []
    if args.batch_file:
        try:
            inputs.extend(read_batch_file(args.batch_file))
        except OSError as e:
            print(f"ERROR: Failed to read batch file: {args.batch_file} ({e})", file=sys.stderr)
            sys.exit(1)
    inputs.extend(args.inputs or [])

    if not inputs:
        print("ERROR: Provide at least one input path or --batch-file", file=sys.stderr)
        sys.exit(2)

    multiple_inputs = len(inputs) > 1
    for idx, input_path in enumerate(inputs):
        if not os.path.isfile(input_path):
            print(f"ERROR: File not found: {input_path}", file=sys.stderr)
            if args.strict:
                sys.exit(1)
            continue

        if args.output_dir is None:
            base = os.path.splitext(os.path.basename(input_path))[0]
            out_dir = os.path.join(os.path.dirname(input_path) or ".", f"{base}_stems")
        else:
            out_dir = os.path.abspath(args.output_dir)
            if multiple_inputs:
                base = os.path.splitext(os.path.basename(input_path))[0]
                out_dir = os.path.join(out_dir, f"{base}_stems")

        os.makedirs(out_dir, exist_ok=True)

        if multiple_inputs:
            print(f"\n=== [{idx+1}/{len(inputs)}] {os.path.basename(input_path)} ===")

        print(f"Loading audio: {input_path}")
        audio = load_audio(input_path)
        total_samples = audio.shape[0]
        print(f"  {total_samples} samples, {total_samples/SAMPLE_RATE:.1f}s, stereo")

        chunks = make_chunks(audio)
        print(f"  {len(chunks)} chunk(s), {OVERLAP*100:.0f}% overlap")

        print("Separating...")
        chunk_results = []
        for i, (start, chunk) in enumerate(chunks):
            t0 = time.time()
            stems = run_model(model, chunk)
            elapsed = time.time() - t0
            print(f"  Chunk {i+1}/{len(chunks)} @ {start/SAMPLE_RATE:.1f}s  [{elapsed:.1f}s]")
            chunk_results.append((start, stems))

        final_stems = overlap_add(chunk_results, total_samples)

        print(f"Saving to {out_dir}/")
        for i, name in enumerate(stem_names):
            out_path = os.path.join(out_dir, f"{name}.wav")
            sf.write(out_path, final_stems[i].T, SAMPLE_RATE, subtype="FLOAT")
            print(f"  {out_path}")

        print("Done.")


if __name__ == "__main__":
    main()
