# Emagic Project Notes

## Commands

- **Run emagic CLI**: `pixi run emagic` (from project root) or `emagic` (shell alias)
- **Python with librosa**: Use `python` (NOT `python3`) — system `python` has librosa 0.11.0 installed
- **Pixi environment Python**: `pixi run python` — Python 3.12, has torch/coremltools/soundfile/soxr but NOT librosa

## Output Paths

- **Stem output**: `<input>_stems/` next to the input file (default)
- **Visualization images**: `/Users/cameronbrooks/Desktop/` (ad-hoc analysis PNGs)
  - `warmgun_chords.png` — chord template matching + top-N quantization
  - `warmgun_chroma.png` — multi-panel chroma comparison
  - `warmgun_guitar_deep.png` — deep guitar analysis (chroma, beat-synced, filtered)
  - `warmgun_librosa.png` — librosa-style mel spectrograms
  - `warmgun_analysis.png` — matplotlib stem analysis

## Common Pitfalls

- **`python3` vs `python`**: System `python3` does NOT have librosa; `python` does. Always use `python` for librosa/visualization scripts.
- **CoreML model loading**: Must use `ct.models.CompiledMLModel` for `.mlmodelc` files, NOT `ct.models.MLModel` (which expects `.mlmodel`/`.mlpackage`).
- **Model directory must end in `.mlmodelc`**: CoreML requires the extension. A directory named `model/` won't load — use `model.mlmodelc` (symlink is fine).
- **KMP_DUPLICATE_LIB_OK=TRUE**: Required to avoid OpenMP duplicate library crash when torch + coremltools coexist. Set in pixi activation env.
- **Resampling**: Model is locked to 44100 Hz / 588800 samples per chunk. Non-44.1kHz audio must be resampled with `soxr` (librosa/torchaudio won't install cleanly in pixi Python 3.12).
- **Stem ordering**: `[bass, drums, other, vocals, guitar, piano]` — verified by ear. Do NOT change without re-testing.
- **Git LFS**: Weight files (`*.bin`) tracked via LFS. Run `git lfs install` before cloning.
- **librosa in pixi**: Won't install under Python 3.12 due to numba dependency. Use system `python` for any librosa work.

## Architecture Quick Reference

- **Model**: BSRoformer (Band-Split RoPE Transformer) from Logic Pro, compiled CoreML (.mlmodelc)
- **Input**: `[1, 2, 588800]` — stereo audio, ~13.35s at 44.1kHz
- **Output**: Two tensors `var_11707` (real) + `var_11751` (imaginary), shape `[1, 12, 1151, 1025]` — 6 stems x 2 channels, complex spectrogram
- **Reconstruction**: Reshape to `[6, 2, 1151, 1025]`, combine real+imag, `torch.istft` per stem/channel
- **STFT params**: n_fft=2048, hop_length=512, hann window
- **Overlap-add**: 50% overlap with linear crossfade between chunks
# Batch file helper

Generate a `songs.txt` (one audio path per line) from a folder:

`python tools/make_batch_file.py /path/to/songs -r -o songs.txt`

Then run:

`python src/bsroformer_sep/cli.py --batch-file songs.txt -o out_dir`
