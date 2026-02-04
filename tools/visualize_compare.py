#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class VizParams:
    n_fft: int
    hop_length: int
    n_mels: int
    title: str


DEFAULT_VIZ: list[VizParams] = [
    VizParams(n_fft=2048, hop_length=512, n_mels=128, title="n_fft=2048 hop=512 mel=128"),
    VizParams(n_fft=4096, hop_length=1024, n_mels=128, title="n_fft=4096 hop=1024 mel=128"),
    VizParams(n_fft=2048, hop_length=512, n_mels=256, title="n_fft=2048 hop=512 mel=256"),
]


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    # shape: [T, C]
    return y, int(sr)


def _to_mono(y_tc: np.ndarray) -> np.ndarray:
    if y_tc.ndim != 2:
        raise ValueError("Expected audio in shape [T, C].")
    return y_tc.mean(axis=1)


def _slice_seconds(y_tc: np.ndarray, sr: int, start_s: float, dur_s: float) -> np.ndarray:
    start = int(round(start_s * sr))
    end = int(round((start_s + dur_s) * sr))
    start = max(0, min(start, y_tc.shape[0]))
    end = max(start, min(end, y_tc.shape[0]))
    return y_tc[start:end]


def _stft_mag_power(y: np.ndarray, n_fft: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    D = librosa.stft(y=y, n_fft=n_fft, hop_length=hop_length, center=True)
    mag = np.abs(D)
    power = mag * mag
    return mag, power


def _mel_db_from_power(S_power: np.ndarray, sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel = mel_basis @ S_power
    return librosa.power_to_db(mel, ref=np.max)


def _save_fig(fig: plt.Figure, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create librosa visual comparisons for two audio files.",
    )
    parser.add_argument("file_a", help="First audio file")
    parser.add_argument("file_b", help="Second audio file")
    parser.add_argument("--start", type=float, default=0.0, help="Start time (seconds)")
    parser.add_argument("--dur", type=float, default=10.0, help="Duration to visualize (seconds)")
    parser.add_argument(
        "--out-dir",
        default="viz_compare_out",
        help="Output directory for PNGs",
    )
    args = parser.parse_args()

    file_a = os.path.expanduser(args.file_a)
    file_b = os.path.expanduser(args.file_b)
    out_dir = Path(os.path.expanduser(args.out_dir)).resolve()

    yA_tc, srA = _read_audio(file_a)
    yB_tc, srB = _read_audio(file_b)
    if srA != srB:
        raise SystemExit(f"Sample-rate mismatch: {srA} vs {srB}. Resample first for fair viz.")

    yA_seg = _slice_seconds(yA_tc, srA, args.start, args.dur)
    yB_seg = _slice_seconds(yB_tc, srB, args.start, args.dur)

    if yA_seg.shape != yB_seg.shape:
        n = min(yA_seg.shape[0], yB_seg.shape[0])
        yA_seg = yA_seg[:n]
        yB_seg = yB_seg[:n]

    monoA = _to_mono(yA_seg)
    monoB = _to_mono(yB_seg)

    # Waveforms (mono)
    t = np.arange(monoA.shape[0]) / srA
    fig, ax = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    ax[0].plot(t, monoA, linewidth=0.6)
    ax[0].set_title("A waveform (mono)")
    ax[1].plot(t, monoB, linewidth=0.6)
    ax[1].set_title("B waveform (mono)")
    ax[2].plot(t, monoA - monoB, linewidth=0.6)
    ax[2].set_title("A - B waveform (mono)")
    ax[2].set_xlabel("Time (s)")
    for a in ax:
        a.grid(True, alpha=0.2)
    _save_fig(fig, out_dir / "waveforms.png")

    # Feature grids for a few parameter sets
    for i, vp in enumerate(DEFAULT_VIZ, start=1):
        magA, powA = _stft_mag_power(monoA, vp.n_fft, vp.hop_length)
        magB, powB = _stft_mag_power(monoB, vp.n_fft, vp.hop_length)

        melA = _mel_db_from_power(powA, srA, vp.n_fft, vp.n_mels)
        melB = _mel_db_from_power(powB, srB, vp.n_fft, vp.n_mels)
        melD = melA - melB

        stA = librosa.amplitude_to_db(magA, ref=np.max)
        stB = librosa.amplitude_to_db(magB, ref=np.max)
        stD = stA - stB

        fig, ax = plt.subplots(2, 3, figsize=(18, 9), sharex=False, sharey=False)
        fig.suptitle(vp.title, fontsize=14)

        librosa.display.specshow(melA, sr=srA, hop_length=vp.hop_length, x_axis="time", y_axis="mel", ax=ax[0, 0])
        ax[0, 0].set_title("A mel dB")
        librosa.display.specshow(melB, sr=srA, hop_length=vp.hop_length, x_axis="time", y_axis="mel", ax=ax[0, 1])
        ax[0, 1].set_title("B mel dB")
        librosa.display.specshow(melD, sr=srA, hop_length=vp.hop_length, x_axis="time", y_axis="mel", ax=ax[0, 2], cmap="coolwarm")
        ax[0, 2].set_title("A - B mel (dB diff)")

        librosa.display.specshow(stA, sr=srA, hop_length=vp.hop_length, x_axis="time", y_axis="linear", ax=ax[1, 0])
        ax[1, 0].set_title("A STFT mag dB")
        librosa.display.specshow(stB, sr=srA, hop_length=vp.hop_length, x_axis="time", y_axis="linear", ax=ax[1, 1])
        ax[1, 1].set_title("B STFT mag dB")
        librosa.display.specshow(stD, sr=srA, hop_length=vp.hop_length, x_axis="time", y_axis="linear", ax=ax[1, 2], cmap="coolwarm")
        ax[1, 2].set_title("A - B STFT (dB diff)")

        for a in ax.ravel():
            a.label_outer()

        _save_fig(fig, out_dir / f"features_{i}.png")

    # Summary text
    identical = bool(np.array_equal(yA_seg, yB_seg))
    summary = (
        f"A: {file_a}\n"
        f"B: {file_b}\n"
        f"sr: {srA}\n"
        f"segment: start={args.start}s dur={args.dur}s\n"
        f"segment_samples: {yA_seg.shape[0]}\n"
        f"identical_segment: {identical}\n"
        f"max_abs_diff: {float(np.max(np.abs(yA_seg - yB_seg)))}\n"
    )
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(f"Wrote visualizations to {out_dir}")


if __name__ == "__main__":
    main()
