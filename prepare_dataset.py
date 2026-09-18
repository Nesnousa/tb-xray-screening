"""
Prepare the TB Chest Radiography dataset for training.

Takes the Kaggle download (a .zip, or an already-unzipped folder) and
splits it into the data/train/<class> and data/val/<class> layout that
training/train.py expects, without needing to already know the exact
internal folder names the Kaggle archive uses.

Usage (run from inside the project root, i.e. tb-xray-screening/):

    python prepare_dataset.py --source "C:\\Users\\Nesrine\\Downloads\\archive (1).zip"

Or, if you already extracted it somewhere:

    python prepare_dataset.py --source "C:\\Users\\Nesrine\\Downloads\\TB_Chest_Radiography_Database"
"""
from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
# The two class names we're looking for, matched case-insensitively against
# folder names anywhere in the extracted tree (the public Kaggle dataset
# ships them as top-level folders, but we don't rely on the exact layout).
CLASS_ALIASES = {
    "normal": "Normal",
    "tuberculosis": "Tuberculosis",
    "tb": "Tuberculosis",
}


def find_class_folders(root: Path) -> dict[str, Path]:
    """Search the extracted tree for folders matching our known class names."""
    found: dict[str, Path] = {}
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        key = path.name.strip().lower()
        if key in CLASS_ALIASES and CLASS_ALIASES[key] not in found:
            # Prefer folders that actually contain images.
            if any(p.suffix.lower() in IMAGE_EXTS for p in path.iterdir() if p.is_file()):
                found[CLASS_ALIASES[key]] = path
    return found


def extract_if_needed(source: Path, work_dir: Path) -> Path:
    if source.is_dir():
        return source
    if source.suffix.lower() != ".zip":
        raise ValueError(f"--source must be a folder or a .zip file, got: {source}")

    extract_to = work_dir / "extracted"
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"Reusing previously extracted files in {extract_to}")
        return extract_to

    print(f"Extracting {source.name} ... (this can take a minute for large archives)")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as zf:
        zf.extractall(extract_to)
    return extract_to


def split_and_copy(class_folders: dict[str, Path], dest_root: Path, val_split: float, seed: int) -> None:
    rng = random.Random(seed)

    for class_name, folder in class_folders.items():
        images = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
        rng.shuffle(images)

        n_val = max(1, int(len(images) * val_split))
        val_images, train_images = images[:n_val], images[n_val:]

        train_dir = dest_root / "train" / class_name
        val_dir = dest_root / "val" / class_name
        train_dir.mkdir(parents=True, exist_ok=True)
        val_dir.mkdir(parents=True, exist_ok=True)

        for img in train_images:
            shutil.copy2(img, train_dir / img.name)
        for img in val_images:
            shutil.copy2(img, val_dir / img.name)

        print(f"{class_name}: {len(train_images)} train, {len(val_images)} val (of {len(images)} total)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Path to the downloaded .zip, or an already-extracted folder")
    parser.add_argument("--dest", default="data", help="Output root (default: ./data)")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of each class held out for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Source not found: {source}")

    dest_root = Path(args.dest).expanduser().resolve()
    work_dir = Path(".prepare_dataset_work").resolve()
    work_dir.mkdir(exist_ok=True)

    extracted_root = extract_if_needed(source, work_dir)

    class_folders = find_class_folders(extracted_root)
    missing = set(CLASS_ALIASES.values()) - set(class_folders)
    if missing:
        raise SystemExit(
            f"Could not find folder(s) for: {', '.join(sorted(missing))} inside {extracted_root}.\n"
            f"Open that folder and check the actual subfolder names, then adjust CLASS_ALIASES "
            f"in this script if the dataset uses different names."
        )

    for name, folder in class_folders.items():
        print(f"Found '{name}' images in: {folder}")

    split_and_copy(class_folders, dest_root, args.val_split, args.seed)

    print(f"\nDone. Data ready under: {dest_root}")
    print("Next step: python training/train.py --config training/config.yaml")


if __name__ == "__main__":
    main()
