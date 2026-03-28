# Emagic (CoreML) + PyMagic (PyTorch)

Stem separation utilities for macOS CoreML (`emagic`) and a PyTorch reference runner (`pymagic`).

## Layout

- `coreml/` — CoreML model + Swift helper + CLI
- `pytorch/` — PyTorch BSRoformer model, training pipeline, and `pymagic` CLI
- `tools/` — small helper tools (batch list generation, visualization)

## CLI entrypoints

### `emagic` (CoreML)

Separate one or more inputs:

```bash
emagic song.wav
emagic song1.wav song2.wav -o out_dir
```

Batch list file (one path per line, blank lines and `#` comments ignored):

```bash
emagic --batch-file songs.txt -o out_dir
```

### `pymagic` (PyTorch)

Separate one or more inputs:

```bash
pymagic song.wav
pymagic song1.wav song2.wav -o out_dir
```

Batch list file:

```bash
pymagic --batch-file songs.txt -o out_dir
```

## Batch list generator

Create `songs.txt` from a folder:

```bash
python tools/make_batch_file.py /path/to/songs -r -o songs.txt
```

## PyTorch "control knobs" (post-processing)

These are intended as "slider-like" parameters to trade bleed vs artifacts without retraining:

```bash
pymagic song.wav \
  --mask-sharpness 1.2 \
  --competition 0.4 \
  --smooth-time 5 \
  --smooth-freq 5 \
  --cancel bass:other=0.15 \
  -o out_dir
```

- `--mask-sharpness` — >1 reduces bleed (can add artifacts)
- `--competition` / `--competition-temp` — force stems to "compete" per time‑freq bin
- `--smooth-time` / `--smooth-freq` — smooth mask magnitude to reduce warble/musical noise
- `--stem-gain stem=value` — repeatable gain per stem (e.g. `--stem-gain vocals=1.1`)
- `--cancel target:source=value` — repeatable "subtract bleed" mixer (e.g. `--cancel bass:other=0.15`)

## Sample-rate modes (PyTorch)

By default, `pymagic` keeps the file's sample rate end-to-end (no resampling):

```bash
pymagic song.wav --sr-mode source
```

To force separation at a fixed model SR (resample in, separate, resample stems back to the original SR):

```bash
pymagic song.wav --sr-mode model --model-sr 44100
```

## Installation notes

If `emagic` / `pymagic` aren't on your PATH yet, install the package (editable):

```bash
pip install -e .
```

(`pixi` is optional; you can run the scripts with any Python environment that has the dependencies.)
