"""
Scans data/train and data/val for image files that TensorFlow's image
pipeline can't actually decode (corrupt files, files with a misleading
extension, truncated downloads, etc.) and moves them out of the way into
a `data/_quarantine/` folder (preserving their train/val/class subpath),
so that training doesn't crash mid-epoch on them.

Why this is needed: `evaluate.py` already found one such file
(`data/val/Normal/Normal-2.png: cannot identify image file`) and skipped
it safely with a try/except. But `training/train.py` uses
`tf.keras.utils.image_dataset_from_directory`, which decodes images lazily
*inside* the TensorFlow graph while training is running - so a single bad
file doesn't raise a clean Python exception, it crashes the whole
`model.fit()` call partway through an epoch with a low-level
`InvalidArgumentError` instead. Easiest fix: find and remove every bad
file *before* training starts, once, rather than trying to make the
in-graph pipeline swallow errors silently.

Usage:
    python find_bad_images.py
    python find_bad_images.py --data-dir data --apply
(dry-run by default: only lists what it would move; pass --apply to
actually quarantine the files)
"""
from __future__ import annotations

import argparse
import os
import shutil

from PIL import Image


def is_bad(path: str) -> str | None:
    """Returns an error string if the file can't be fully read as an
    image, else None. Uses two checks: PIL's own verify() (cheap, catches
    truncated/corrupt files) plus an actual full load (catches some files
    that pass verify() but still fail to decode)."""
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB").load()
        return None
    except Exception as exc:
        return str(exc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--apply", action="store_true", help="Actually move bad files (default: dry run)")
    args = parser.parse_args()

    quarantine_root = os.path.join(args.data_dir, "_quarantine")
    bad_files = []

    for split in ("train", "val"):
        split_dir = os.path.join(args.data_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            class_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(class_dir):
                continue
            for fname in sorted(os.listdir(class_dir)):
                if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
                    continue
                path = os.path.join(class_dir, fname)
                err = is_bad(path)
                if err:
                    bad_files.append((split, cls, fname, path, err))

    if not bad_files:
        print("No bad image files found. Your dataset is clean - training should run without decode errors.")
        return

    print(f"Found {len(bad_files)} unreadable file(s):\n")
    for split, cls, fname, path, err in bad_files:
        print(f"  {path}\n      -> {err}")

    if not args.apply:
        print(
            f"\nDry run only - nothing was moved. Re-run with --apply to move these "
            f"{len(bad_files)} file(s) into '{quarantine_root}/' (kept, not deleted, "
            f"in case you want to inspect them later)."
        )
        return

    for split, cls, fname, path, err in bad_files:
        dest_dir = os.path.join(quarantine_root, split, cls)
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(path, os.path.join(dest_dir, fname))

    print(f"\nMoved {len(bad_files)} file(s) into '{quarantine_root}/'. Your training/val folders are now clean.")


if __name__ == "__main__":
    main()
