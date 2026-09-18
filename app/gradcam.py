"""
Grad-CAM explainability overlay.

A raw "Tuberculosis: 87%" number is hard to trust in a clinical-adjacent
tool. Grad-CAM highlights *which regions* of the radiograph pushed the
model toward its decision, which is the single feature the reference
project this one is inspired by does not have. It is a well-known
technique (Selvaraju et al., 2017) reimplemented here against
tf.keras directly.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

from app.config import settings


def _find_nested_submodel(model):
    """Our training script builds the network as Input -> backbone(...) ->
    Dense -> Dropout -> Dense, where `backbone` (e.g. EfficientNetB0) is
    itself a whole Functional model reused as a single layer. Grad-CAM
    needs a conv layer *inside* that backbone, so we have to find it
    first — model.get_layer(...) only sees the outer model's top-level
    layers, not what's nested inside the backbone."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            return layer
    return None


def _build_grad_pipeline(model, last_conv_layer: str):
    """Returns (conv_extractor, forward_from_conv) where:
      - conv_extractor(batch) -> conv activations (what we compute Grad-CAM from)
      - forward_from_conv(conv_activations) -> final model prediction

    Split this way because the conv layer usually lives inside a nested
    submodel (see _find_nested_submodel): its output tensor belongs to
    that submodel's own graph and can't be wired directly into a Model
    spanning the *outer* model's inputs -> outputs (Keras raises
    "Output ... is not connected to inputs" if you try). Instead we build
    two small models entirely within the submodel's own graph (valid),
    then bridge them to the outer model's remaining layers by calling
    those layers directly inside one GradientTape.
    """
    submodel = _find_nested_submodel(model)

    if submodel is None:
        # No nested backbone - conv layer should be directly on `model`.
        try:
            conv_layer = model.get_layer(last_conv_layer)
        except ValueError:
            conv_layer = next(l for l in reversed(model.layers) if len(l.output_shape) == 4)
        conv_extractor = tf.keras.Model(model.inputs, conv_layer.output)

        def forward_from_conv(conv_out):
            return model(conv_out, training=False)  # not reachable in our current architecture

        return conv_extractor, forward_from_conv

    try:
        conv_layer = submodel.get_layer(last_conv_layer)
    except ValueError:
        conv_layer = next(l for l in reversed(submodel.layers) if len(l.output_shape) == 4)

    conv_extractor = tf.keras.Model(submodel.input, conv_layer.output)
    tail = tf.keras.Model(conv_layer.output, submodel.output)

    post_layers = []
    seen_submodel = False
    for layer in model.layers:
        if layer is submodel:
            seen_submodel = True
            continue
        if seen_submodel:
            post_layers.append(layer)

    def forward_from_conv(conv_out):
        x = tail(conv_out, training=False)
        for layer in post_layers:
            try:
                x = layer(x, training=False)
            except TypeError:
                x = layer(x)
        return x

    return conv_extractor, forward_from_conv


def make_gradcam_overlay(model, raw_bytes: bytes, last_conv_layer: str = settings.last_conv_layer) -> bytes:
    """Return a PNG (as bytes) of the original image blended with a
    Grad-CAM heatmap for the predicted class."""
    global tf
    import tensorflow as tf

    # Same raw-[0,255] rule as app/inference.py: EfficientNetB0 has its own
    # internal Rescaling(1/255) + Normalization as its first two layers, so we
    # must NOT pre-divide by 255 here (see inference.py for the full story).
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB").resize(settings.img_size)
    arr = np.asarray(img, dtype="float32")
    batch = tf.convert_to_tensor(np.expand_dims(arr, axis=0))

    conv_extractor, forward_from_conv = _build_grad_pipeline(model, last_conv_layer)

    with tf.GradientTape() as tape:
        conv_outputs = conv_extractor(batch, training=False)
        tape.watch(conv_outputs)
        predictions = forward_from_conv(conv_outputs)
        class_channel = predictions[:, 0] if predictions.shape[-1] == 1 else predictions[:, 1]

    grads = tape.gradient(class_channel, conv_outputs)
    if grads is None:
        raise RuntimeError(
            "Could not compute Grad-CAM gradients - the conv layer output isn't "
            "connected to the prediction in the way this code expects."
        )
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    heatmap_img = Image.fromarray(np.uint8(255 * heatmap)).resize(settings.img_size)
    heatmap_rgba = _apply_colormap(np.asarray(heatmap_img))

    base = img.convert("RGBA")
    overlay = Image.fromarray(heatmap_rgba, mode="RGBA")
    blended = Image.alpha_composite(base, overlay)

    out = io.BytesIO()
    blended.convert("RGB").save(out, format="PNG")
    return out.getvalue()


def _apply_colormap(gray: np.ndarray, alpha: int = 110) -> np.ndarray:
    """Cheap red/yellow 'hot' colormap so we don't need matplotlib as a
    runtime dependency just for this."""
    h, w = gray.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., 0] = 255                       # R
    rgba[..., 1] = gray                      # G ramps up with intensity -> red to yellow
    rgba[..., 2] = 0                         # B
    rgba[..., 3] = (gray.astype(np.float32) / 255.0 * alpha).astype(np.uint8)
    return rgba
