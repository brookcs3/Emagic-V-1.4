#!/usr/bin/env python3
"""
emagic - BSRoformer CoreML stem separator

Separates stereo audio into 6 stems using Apple's compiled BSRoformer
model (.mlmodelc) via CoreML on macOS.
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

VERSION = "1.3.0"
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 588800
N_FFT = 2048
HOP_LENGTH = 512
N_STEMS = 6

DEFAULT_OVERLAP = 0.5
DEFAULT_STEM_NAMES = ["bass", "drums", "other", "vocals", "guitar", "piano"]
DEFAULT_MODEL = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "model.mlmodelc")
)


def load_audio(path: str, quiet: bool = False) -> np.ndarray:
    data, sr = sf.read(path, dtype="float32", always_2d=True)

    if sr != SAMPLE_RATE:
        import soxr
        data = soxr.resample(data, sr, SAMPLE_RATE, quality="HQ")
        if not quiet:
            print(f"  Resampled {sr} -> {SAMPLE_RATE} Hz")

    if data.shape[1] == 1:
        data = np.concatenate([data, data], axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]

    return data


def make_chunks(audio: np.ndarray, overlap: float) -> list[tuple[int, np.ndarray]]:
    total_samples = audio.shape[0]
    step = int(SEGMENT_SAMPLES * (1 - overlap))
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
    input_data = chunk.T[np.newaxis, :, :].astype(np.float32)
    result = model.predict({"input": input_data})

    complex_spec = (
        torch.from_numpy(result["var_11707"])
        + 1j * torch.from_numpy(result["var_11751"])
    )
    complex_spec = complex_spec.squeeze(0).reshape(N_STEMS, 2, 1151, 1025)

    window = torch.hann_window(N_FFT)
    stems = []
    for s in range(N_STEMS):
        channels = []
        for c in range(2):
            audio_out = torch.istft(
                complex_spec[s, c].T,
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


def overlap_add(chunks_results: list[tuple[int, np.ndarray]], total_samples: int, overlap: float) -> np.ndarray:
    output = np.zeros((N_STEMS, 2, total_samples), dtype=np.float32)
    weight = np.zeros(total_samples, dtype=np.float32)

    fade = np.ones(SEGMENT_SAMPLES, dtype=np.float32)
    fade_len = int(SEGMENT_SAMPLES * overlap)
    if fade_len > 0:
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


def _resolve_output_dir(
    output_dir: str | None,
    input_path: str,
    multiple_inputs: bool,
) -> str:
    if output_dir is None:
        base = os.path.splitext(os.path.basename(input_path))[0]
        return os.path.join(os.path.dirname(os.path.abspath(input_path)), f"{base}_stems")

    output_dir = os.path.abspath(output_dir)
    if not multiple_inputs:
        return output_dir

    base = os.path.splitext(os.path.basename(input_path))[0]
    return os.path.join(output_dir, f"{base}_stems")


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


def _separate_one(
    *,
    model,
    input_path: str,
    output_dir: str,
    stem_names: list[str],
    only: set[str] | None,
    overlap: float,
    out_format: str,
    quiet: bool,
) -> None:
    log = (lambda *a, **kw: None) if quiet else print

    os.makedirs(output_dir, exist_ok=True)

    log(f"Input:   {input_path}")
    audio = load_audio(input_path, quiet=quiet)
    total_samples = audio.shape[0]
    log(f"  {total_samples/SAMPLE_RATE:.1f}s, {total_samples} samples, stereo")

    chunks = make_chunks(audio, overlap)
    log(f"  {len(chunks)} chunk(s), {overlap*100:.0f}% overlap")

    log("Separating...")
    t_total = time.time()
    chunk_results = []
    for i, (start, chunk) in enumerate(chunks):
        t0 = time.time()
        stems = run_model(model, chunk)
        elapsed = time.time() - t0
        log(f"  [{i+1}/{len(chunks)}] {start/SAMPLE_RATE:.1f}s  ({elapsed:.1f}s)")
        chunk_results.append((start, stems))

    final_stems = overlap_add(chunk_results, total_samples, overlap)
    log(f"  Total: {time.time()-t_total:.1f}s")

    ext = out_format
    subtype = "FLOAT" if ext == "wav" else "PCM_24"

    log(f"Output:  {output_dir}/")
    for i, name in enumerate(stem_names):
        if only and name not in only:
            continue
        out_path = os.path.join(output_dir, f"{name}.{ext}")
        sf.write(out_path, final_stems[i].T, SAMPLE_RATE, subtype=subtype)
        log(f"  {name}.{ext}")

    log("Done.")


def main():
    parser = argparse.ArgumentParser(
        prog="emagic",
        description="Separate audio into stems using BSRoformer CoreML inference.",
        epilog="Examples:\n"
               "  emagic song.wav\n"
               "  emagic song.mp3 -o separated/\n"
               "  emagic song.flac --only vocals drums\n"
               "  emagic song.wav --compute-units cpu --format flac\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

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
    parser.add_argument("-o", "--output-dir", default=None, metavar="DIR",
                        help="Output directory (default: <input>_stems/)")
    parser.add_argument("-m", "--model", default=None, metavar="PATH",
                        help="Path to .mlmodelc model (default: bundled model)")
    parser.add_argument("--stems", nargs="+", default=None, metavar="NAME",
                        help="Custom stem names, 6 names in order "
                             "(default: vocals drums bass guitar piano other)")
    parser.add_argument("--only", nargs="+", default=None, metavar="NAME",
                        help="Only output these stems (e.g. --only vocals drums)")
    parser.add_argument("--compute-units", choices=["cpu", "gpu", "all"], default="gpu",
                        help="CoreML compute units (default: gpu)")
    parser.add_argument("--format", choices=["wav", "flac"], default="wav",
                        help="Output audio format (default: wav)")
    parser.add_argument("--overlap", type=float, default=DEFAULT_OVERLAP, metavar="F",
                        help=f"Chunk overlap ratio 0.0-0.9 (default: {DEFAULT_OVERLAP})")
    parser.add_argument("-q", "--quiet", action="store_true",
                        help="Minimal output")
    parser.add_argument("-v", "--version", action="version", version=f"emagic {VERSION}")

    args = parser.parse_args()

    inputs: list[str] = []
    if args.batch_file:
        try:
            inputs.extend(_read_batch_file(args.batch_file))
        except OSError as e:
            print(f"Error: failed to read batch file: {args.batch_file} ({e})", file=sys.stderr)
            sys.exit(1)
    inputs.extend(args.inputs or [])

    if not inputs:
        print("Error: provide at least one input path or --batch-file", file=sys.stderr)
        sys.exit(2)

    model_path = args.model or DEFAULT_MODEL
    if not os.path.isdir(model_path):
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        print("Provide --model /path/to/model.mlmodelc", file=sys.stderr)
        sys.exit(1)

    compute_map = {
        "cpu": ct.ComputeUnit.CPU_ONLY,
        "gpu": ct.ComputeUnit.CPU_AND_GPU,
        "all": ct.ComputeUnit.ALL,
    }

    stem_names = args.stems if args.stems and len(args.stems) == N_STEMS else DEFAULT_STEM_NAMES
    overlap = max(0.0, min(0.9, args.overlap))

    multiple_inputs = len(inputs) > 1
    log = (lambda *a, **kw: None) if args.quiet else print
    only = set(args.only) if args.only else None

    log(f"emagic v{VERSION}")
    log(f"Model:   {model_path}")
    log(f"Compute: {args.compute_units}")

    t0 = time.time()
    model = ct.models.CompiledMLModel(model_path, compute_units=compute_map[args.compute_units])
    log(f"  Loaded in {time.time()-t0:.1f}s")

    for idx, input_path in enumerate(inputs):
        if not os.path.isfile(input_path):
            print(f"Error: file not found: {input_path}", file=sys.stderr)
            if args.strict:
                sys.exit(1)
            continue

        if multiple_inputs and not args.quiet:
            print(f"\n=== [{idx+1}/{len(inputs)}] {os.path.basename(input_path)} ===")

        out_dir = _resolve_output_dir(args.output_dir, input_path, multiple_inputs=multiple_inputs)
        _separate_one(
            model=model,
            input_path=input_path,
            output_dir=out_dir,
            stem_names=stem_names,
            only=only,
            overlap=overlap,
            out_format=args.format,
            quiet=args.quiet,
        )


if __name__ == "__main__":
    main()
