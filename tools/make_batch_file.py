#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import random
from pathlib import Path


DEFAULT_GLOBS = [
    "*.wav",
    "*.mp3",
    "*.flac",
    "*.m4a",
    "*.aiff",
    "*.aif",
    "*.aac",
    "*.ogg",
]


def _iter_matches(root: Path, patterns: list[str], recursive: bool) -> list[Path]:
    matches: list[Path] = []
    if recursive:
        for pattern in patterns:
            matches.extend(root.rglob(pattern))
    else:
        for pattern in patterns:
            matches.extend(root.glob(pattern))

    out: list[Path] = []
    seen: set[Path] = set()
    for p in matches:
        try:
            p = p.resolve()
        except OSError:
            continue
        if p.is_file() and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a batch list file (one audio path per line).",
        epilog="Examples:\n"
        "  python tools/make_batch_file.py /path/to/songs -o songs.txt\n"
        "  python tools/make_batch_file.py /path/to/songs -r --shuffle --seed 123 -o songs.txt\n"
        "  python tools/make_batch_file.py /path/to/songs --relative-to /path/to -o songs.txt\n",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Folder(s) and/or file(s). Folders are scanned for audio files.",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="songs.txt",
        help="Output text file path (default: songs.txt).",
    )
    parser.add_argument(
        "--glob",
        action="append",
        dest="globs",
        default=[],
        help="Glob pattern(s) to include. Repeatable. Default: common audio types.",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recurse into subfolders.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the file list before writing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for --shuffle (optional).",
    )
    parser.add_argument(
        "--relative-to",
        default=None,
        help="Write paths relative to this directory (useful for portable batch files).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output if it already exists.",
    )
    args = parser.parse_args()

    patterns = args.globs or DEFAULT_GLOBS
    files: list[Path] = []

    for item in args.inputs:
        p = Path(os.path.expandvars(os.path.expanduser(item)))
        if p.is_dir():
            files.extend(_iter_matches(p, patterns, recursive=args.recursive))
        elif p.is_file():
            files.append(p.resolve())

    # De-dupe while preserving order
    uniq: list[Path] = []
    seen: set[Path] = set()
    for p in files:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    files = uniq

    files.sort(key=lambda p: p.as_posix().lower())
    if args.shuffle:
        rng = random.Random(args.seed)
        rng.shuffle(files)

    out_path = Path(os.path.expanduser(args.output)).resolve()
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"Refusing to overwrite existing file: {out_path} (use --overwrite)")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rel_base: Path | None = None
    if args.relative_to:
        rel_base = Path(os.path.expandvars(os.path.expanduser(args.relative_to))).resolve()

    lines: list[str] = []
    for p in files:
        if rel_base is not None:
            try:
                lines.append(p.relative_to(rel_base).as_posix())
            except ValueError:
                lines.append(p.as_posix())
        else:
            lines.append(p.as_posix())

    out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    print(f"Wrote {len(lines)} path(s) to {out_path}")


if __name__ == "__main__":
    main()

