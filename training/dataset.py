"""
Data loading helpers for training.

Expects a directory layout of:

    data/train/Normal/*.png
    data/train/Tuberculosis/*.png
    data/val/Normal/*.png
    data/val/Tuberculosis/*.png

which is how the public TB Chest Radiography Database (Kaggle) unpacks
once split into train/val. Using tf.keras's image_dataset_from_directory
keeps this simple and memory-efficient (streams from disk instead of
loading everything up front).
"""
from __future__ import annotations

import tensorflow as tf


def load_datasets(train_dir: str, val_dir: str, img_size: tuple[int, int], batch_size: int, seed: int = 42):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        seed=seed,
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="binary",
        seed=seed,
    )

    class_names = train_ds.class_names  # e.g. ['Normal', 'Tuberculosis']

    # NOTE: no Rescaling(1/255) here on purpose. tf.keras.applications.EfficientNetB0
    # already contains its own internal Rescaling(1/255) + Normalization layers as
    # the first two layers of the backbone (it's built to accept raw [0,255] pixel
    # values). Normalizing here as well double-rescales every image down to a
    # ~[0, 0.0039] range before it even reaches the backbone, crushing the signal
    # to near-zero - this was the root cause of the model learning to predict the
    # same output for every image. image_dataset_from_directory already yields
    # float32 images in [0,255], which is exactly what the backbone wants.
    # A bit more augmentation diversity than before (translation + brightness
    # on top of the original flip/rotate/zoom/contrast) - the goal is to make
    # the model less sensitive to exact positioning/exposure quirks of a
    # single training image, which is one lever for reducing confident wrong
    # predictions on hard-but-genuinely-Normal cases.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomTranslation(0.05, 0.05),
            tf.keras.layers.RandomBrightness(0.1, value_range=(0, 255)),
        ],
        name="augmentation",
    )

    train_ds = train_ds.map(lambda x, y: (augmentation(x, training=True), y))

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, class_names


def compute_class_weights(train_dir: str, max_ratio: float = 2.5) -> dict[int, float]:
    """Balances the loss against the typical ~5:1 Normal:TB imbalance seen
    in the public dataset, so the model doesn't just learn to predict
    'Normal' for everything.

    `max_ratio` caps how aggressive that correction is allowed to be. The
    naive inverse-frequency weighting for this dataset works out to roughly
    Normal=0.6 / Tuberculosis=3.0 (a 5:1 ratio) - which is great for
    Tuberculosis recall but pushes the model to flag Normal images as
    Tuberculosis more readily than it needs to, since the decision threshold
    (0.96, see app/config.py) already does most of the work of protecting
    recall. Capping the ratio to something milder (2.5:1 by default) keeps
    the correction's direction but tones down its strength, aiming for fewer
    confident false positives on Normal without giving up much - if any -
    Tuberculosis recall. Re-run evaluate.py after retraining to confirm the
    actual trade-off and re-tune the threshold if needed.
    """
    import os

    counts = {}
    for idx, label in enumerate(sorted(os.listdir(train_dir))):
        label_dir = os.path.join(train_dir, label)
        if os.path.isdir(label_dir):
            counts[idx] = len(os.listdir(label_dir))

    total = sum(counts.values())
    n_classes = len(counts)
    weights = {idx: total / (n_classes * count) for idx, count in counts.items()}

    lo, hi = min(weights.values()), max(weights.values())
    if lo > 0 and hi / lo > max_ratio:
        scale = (max_ratio * lo) / hi
        weights = {
            idx: (w * scale if w == hi else w)
            for idx, w in weights.items()
        }
    return weights
