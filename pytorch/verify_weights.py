"""Verify CoreML weights loaded into PyTorch match CoreML inference.

Runs both CoreML and PyTorch on the same input, compares spectrogram
outputs and final audio.

Usage:
    cd /path/to/Emagic-V-1.3
    pixi run python -m training.verify_weights
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import coremltools as ct


SEGMENT_SAMPLES = 588800
N_FFT = 2048
HOP_LENGTH = 512
N_STEMS = 6


def run_coreml(model_path: str, audio_np: np.ndarray) -> np.ndarray:
    """Run CoreML inference on a single chunk.

    Args:
        model_path: path to .mlmodelc
        audio_np: [samples, 2] float32

    Returns:
        [6, 2, samples] float32 stems
    """
    model = ct.models.CompiledMLModel(model_path, compute_units=ct.ComputeUnit.CPU_ONLY)

    # CoreML expects [1, 2, T]
    input_data = audio_np.T[np.newaxis, :, :].astype(np.float32)
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
                complex_spec[s, c].T,  # [1025, 1151] -> needs [F, T]
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
                win_length=N_FFT,
                window=window,
                center=True,
                length=SEGMENT_SAMPLES,
            )
            channels.append(audio_out)
        stems.append(torch.stack(channels, dim=0))

    return torch.stack(stems, dim=0).numpy()  # [6, 2, T]


def run_pytorch(state_dict_path: str, audio_np: np.ndarray) -> np.ndarray:
    """Run PyTorch inference with loaded CoreML weights.

    Args:
        state_dict_path: path to mapped_state_dict.pt
        audio_np: [samples, 2] float32

    Returns:
        [6, 2, samples] float32 stems
    """
    from .bsroformer import BSRoformer

    model = BSRoformer()
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(sd, strict=False)

    if missing:
        # Filter out computed buffers
        real_missing = [k for k in missing if not any(
            b in k for b in ("cos_cached", "sin_cached", "inv_freq", "window")
        )]
        if real_missing:
            print(f"  WARNING: {len(real_missing)} missing weight keys:")
            for k in real_missing[:10]:
                print(f"    {k}")
    if unexpected:
        print(f"  WARNING: {len(unexpected)} unexpected keys:")
        for k in unexpected[:10]:
            print(f"    {k}")

    model.eval()

    # PyTorch expects [B, 2, T]
    audio_tensor = torch.from_numpy(audio_np.T).unsqueeze(0).float()

    with torch.no_grad():
        output = model(audio_tensor)  # [1, 6, 2, T]

    return output.squeeze(0).numpy()  # [6, 2, T]


def compare(coreml_out: np.ndarray, pytorch_out: np.ndarray) -> dict:
    """Compare two [6, 2, T] stem arrays."""
    stem_names = ["bass", "drums", "other", "vocals", "guitar", "piano"]

    results = {}
    for s in range(N_STEMS):
        c_stem = coreml_out[s]   # [2, T]
        p_stem = pytorch_out[s]  # [2, T]

        # Truncate to same length
        min_len = min(c_stem.shape[1], p_stem.shape[1])
        c_stem = c_stem[:, :min_len]
        p_stem = p_stem[:, :min_len]

        # L1 error
        l1 = np.mean(np.abs(c_stem - p_stem))

        # Max absolute error
        max_err = np.max(np.abs(c_stem - p_stem))

        # Correlation (flatten both channels)
        c_flat = c_stem.flatten()
        p_flat = p_stem.flatten()

        if np.std(c_flat) > 1e-8 and np.std(p_flat) > 1e-8:
            corr = np.corrcoef(c_flat, p_flat)[0, 1]
        else:
            corr = float("nan")

        # Signal-to-noise ratio (treat difference as noise)
        signal_power = np.mean(c_flat ** 2)
        noise_power = np.mean((c_flat - p_flat) ** 2)
        if noise_power > 0:
            snr_db = 10 * np.log10(signal_power / noise_power)
        else:
            snr_db = float("inf")

        results[stem_names[s]] = {
            "l1": l1,
            "max_err": max_err,
            "corr": corr,
            "snr_db": snr_db,
            "rms_coreml": np.sqrt(np.mean(c_stem ** 2)),
            "rms_pytorch": np.sqrt(np.mean(p_stem ** 2)),
        }

    return results


def main():
    project_root = Path(__file__).resolve().parent.parent
    model_path = str(project_root / "ke77nsfms3_8024.mlmodelc")
    sd_path = str(project_root / "training" / "mapped_state_dict.pt")

    if not os.path.isdir(model_path):
        # Try symlink name
        model_path = str(project_root / "model.mlmodelc")
        if not os.path.isdir(model_path):
            print("ERROR: CoreML model not found")
            sys.exit(1)

    if not os.path.isfile(sd_path):
        print(f"ERROR: {sd_path} not found. Run convert_coreml_weights first.")
        sys.exit(1)

    # Generate deterministic test input
    print("Generating test input...")
    rng = np.random.RandomState(42)
    # Use pink-noise-like signal (more realistic than white noise)
    audio = rng.randn(SEGMENT_SAMPLES, 2).astype(np.float32) * 0.1

    # Run CoreML
    print("Running CoreML inference (CPU)...")
    coreml_out = run_coreml(model_path, audio)
    print(f"  CoreML output: {coreml_out.shape}, RMS={np.sqrt(np.mean(coreml_out**2)):.6f}")

    # Run PyTorch
    print("Running PyTorch inference...")
    pytorch_out = run_pytorch(sd_path, audio)
    print(f"  PyTorch output: {pytorch_out.shape}, RMS={np.sqrt(np.mean(pytorch_out**2)):.6f}")

    # Compare
    print("\n--- Comparison (CoreML vs PyTorch) ---")
    results = compare(coreml_out, pytorch_out)

    print(f"{'Stem':<10} {'L1':>10} {'MaxErr':>10} {'Corr':>8} {'SNR(dB)':>10} {'RMS_CM':>10} {'RMS_PT':>10}")
    print("-" * 70)
    for stem, r in results.items():
        print(
            f"{stem:<10} {r['l1']:>10.6f} {r['max_err']:>10.6f} "
            f"{r['corr']:>8.5f} {r['snr_db']:>10.1f} "
            f"{r['rms_coreml']:>10.6f} {r['rms_pytorch']:>10.6f}"
        )

    # Overall verdict
    avg_snr = np.mean([r["snr_db"] for r in results.values() if np.isfinite(r["snr_db"])])
    avg_corr = np.mean([r["corr"] for r in results.values() if np.isfinite(r["corr"])])
    avg_l1 = np.mean([r["l1"] for r in results.values()])

    print(f"\nOverall: avg_L1={avg_l1:.6f}  avg_corr={avg_corr:.5f}  avg_SNR={avg_snr:.1f} dB")

    if avg_snr > 30:
        print("PASS: Excellent match (>30 dB SNR — fp16 precision level)")
    elif avg_snr > 20:
        print("PASS: Good match (>20 dB SNR — minor numerical differences)")
    elif avg_snr > 10:
        print("WARNING: Moderate match (10-20 dB SNR — check for systematic errors)")
    else:
        print("FAIL: Poor match (<10 dB SNR — likely a mapping error)")


if __name__ == "__main__":
    main()
