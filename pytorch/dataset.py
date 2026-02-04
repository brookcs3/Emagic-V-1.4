"""BSRoformer training dataset.

Expects a directory of songs, each containing 6 stem files:
    root/
        song_001/
            bass.wav
            drums.wav
            other.wav
            vocals.wav
            guitar.wav
            piano.wav
        song_002/
            ...

Mixtures are created on-the-fly by summing stems.
"""

from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    import soundfile as sf
except ImportError:
    sf = None

try:
    import torchaudio
except ImportError:
    torchaudio = None

STEM_NAMES = ["bass", "drums", "other", "vocals", "guitar", "piano"]
SAMPLE_RATE = 44100
SEGMENT_SAMPLES = 588800


def _load_audio(path: str, target_sr: int = SAMPLE_RATE) -> torch.Tensor:
    """Load audio file as [C, T] float32 tensor at target sample rate."""
    if torchaudio is not None:
        wav, sr = torchaudio.load(path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, sr, target_sr)
    elif sf is not None:
        data, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)  # [C, T]
        if sr != target_sr:
            try:
                import soxr
                import numpy as np
                data_rs = soxr.resample(data, sr, target_sr, quality="HQ")
                wav = torch.from_numpy(data_rs.T)
            except ImportError:
                raise RuntimeError(
                    f"Audio at {sr}Hz but target is {target_sr}Hz. "
                    "Install torchaudio or soxr for resampling."
                )
    else:
        raise ImportError("Install either torchaudio or soundfile to load audio.")

    # Ensure stereo
    if wav.shape[0] == 1:
        wav = wav.expand(2, -1)
    elif wav.shape[0] > 2:
        wav = wav[:2]

    return wav


class BSRoformerDataset(Dataset):
    """Dataset for BSRoformer training.

    Each item returns a mixture (sum of stems) and 6 individual stem waveforms,
    cropped or padded to exactly segment_samples.
    """

    def __init__(
        self,
        root: str,
        segment_samples: int = SEGMENT_SAMPLES,
        sample_rate: int = SAMPLE_RATE,
        stem_names: list[str] = STEM_NAMES,
        augment: bool = True,
        audio_extensions: tuple[str, ...] = (".wav", ".flac", ".mp3", ".m4a"),
    ):
        super().__init__()
        self.root = Path(root)
        self.segment_samples = segment_samples
        self.sample_rate = sample_rate
        self.stem_names = stem_names
        self.augment = augment

        # Discover songs: directories containing all required stems
        self.songs: list[Path] = []
        for entry in sorted(self.root.iterdir()):
            if not entry.is_dir():
                continue
            # Check all stems exist
            found = True
            for stem in stem_names:
                stem_path = None
                for ext in audio_extensions:
                    candidate = entry / f"{stem}{ext}"
                    if candidate.exists():
                        stem_path = candidate
                        break
                if stem_path is None:
                    found = False
                    break
            if found:
                self.songs.append(entry)

        if not self.songs:
            raise FileNotFoundError(
                f"No valid song directories found in {root}. "
                f"Expected subdirectories with files: {stem_names}"
            )

    def __len__(self) -> int:
        return len(self.songs)

    def _find_stem_file(self, song_dir: Path, stem_name: str) -> Path:
        """Find the stem audio file with any supported extension."""
        for ext in (".wav", ".flac", ".mp3", ".m4a"):
            p = song_dir / f"{stem_name}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No audio file for stem '{stem_name}' in {song_dir}")

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        song_dir = self.songs[idx]

        # Load all stems
        stems = []
        for stem_name in self.stem_names:
            path = self._find_stem_file(song_dir, stem_name)
            wav = _load_audio(str(path), self.sample_rate)
            stems.append(wav)

        # Stack: [6, 2, T] — all stems should have same length
        min_len = min(s.shape[-1] for s in stems)
        stems = [s[:, :min_len] for s in stems]
        stems_tensor = torch.stack(stems, dim=0)  # [6, 2, T]
        T = stems_tensor.shape[-1]

        # Crop or pad to segment_samples
        if T > self.segment_samples:
            # Random crop
            start = random.randint(0, T - self.segment_samples)
            stems_tensor = stems_tensor[:, :, start:start + self.segment_samples]
        elif T < self.segment_samples:
            # Reflect pad (matching Apple inference logic)
            pad_len = self.segment_samples - T
            stems_tensor = F.pad(stems_tensor, (0, pad_len), mode="reflect")

        # Augmentations
        if self.augment:
            # Random gain per stem: ±6 dB
            gain_db = torch.empty(6, 1, 1).uniform_(-6.0, 6.0)
            gain_linear = (10.0 ** (gain_db / 20.0))
            stems_tensor = stems_tensor * gain_linear

            # Random channel swap (50% chance)
            if random.random() < 0.5:
                stems_tensor = stems_tensor.flip(1)

            # Random polarity flip (50% chance)
            if random.random() < 0.5:
                stems_tensor = -stems_tensor

        # Create mixture by summing stems
        mixture = stems_tensor.sum(dim=0)  # [2, segment_samples]

        return {
            "mixture": mixture,
            "stems": stems_tensor,
        }
