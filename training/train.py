"""
Training entrypoint: transfer learning on a chest-X-ray classifier.

Two-stage strategy:
  1. Train a fresh classification head on top of a frozen ImageNet-pretrained
     backbone (fast, avoids destroying useful pretrained features early).
  2. Optionally unfreeze the backbone and fine-tune end-to-end at a much
     lower learning rate.

Usage:
    python training/train.py --config training/config.yaml
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import tensorflow as tf
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import compute_class_weights, load_datasets


def build_model(backbone_name: str, img_size: tuple[int, int], dropout: float, dense_units: int):
    backbone_cls = getattr(tf.keras.applications, backbone_name)
    backbone = backbone_cls(
        include_top=False,
        weights="imagenet",
        input_shape=(*img_size, 3),
        pooling="avg",
    )
    backbone.trainable = False

    inputs = tf.keras.Input(shape=(*img_size, 3))
    x = backbone(inputs, training=False)
    x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="tb_probability")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, backbone


def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    img_size = tuple(cfg["data"]["img_size"])
    train_ds, val_ds, class_names = load_datasets(
        cfg["data"]["train_dir"],
        cfg["data"]["val_dir"],
        img_size,
        cfg["data"]["batch_size"],
        cfg["data"]["seed"],
    )
    print(f"Classes (index order): {class_names}")

    model, backbone = build_model(
        cfg["model"]["backbone"], img_size, cfg["model"]["dropout"], cfg["model"]["dense_units"]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["train"]["learning_rate"]),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
    )

    class_weight = None
    if cfg["train"]["class_weighting"]:
        class_weight = compute_class_weights(cfg["data"]["train_dir"])
        print(f"Using class weights: {class_weight}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(cfg["output"]["model_path"], monitor="val_auc", mode="max", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
        tf.keras.callbacks.CSVLogger(cfg["output"]["history_csv"]),
    ]

    print("\n=== Stage 1: training classification head (backbone frozen) ===")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg["train"]["epochs"],
        class_weight=class_weight,
        callbacks=callbacks,
    )

    if cfg["train"]["fine_tune_epochs"] > 0:
        print("\n=== Stage 2: fine-tuning with backbone unfrozen ===")
        backbone.trainable = True
        model.compile(
            optimizer=tf.keras.optimizers.Adam(cfg["train"]["fine_tune_learning_rate"]),
            loss="binary_crossentropy",
            metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Recall(name="recall")],
        )
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=cfg["train"]["fine_tune_epochs"],
            class_weight=class_weight,
            callbacks=callbacks,
        )

    _plot_history(cfg["output"]["history_csv"], cfg["output"]["history_plot"])
    print(f"\nDone. Best model saved to {cfg['output']['model_path']}")


def _plot_history(csv_path: str, out_path: str) -> None:
    import csv

    if not os.path.exists(csv_path):
        return
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, [float(r["loss"]) for r in rows], label="train")
    axes[0].plot(epochs, [float(r["val_loss"]) for r in rows], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(epochs, [float(r["auc"]) for r in rows], label="train")
    axes[1].plot(epochs, [float(r["val_auc"]) for r in rows], label="val")
    axes[1].set_title("AUC")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training/config.yaml")
    args = parser.parse_args()
    main(args.config)
