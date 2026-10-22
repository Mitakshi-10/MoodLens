"""Transfer learning and fine-tuning with MobileNetV2 (sections 10 and 11).

FER-2013 is 48x48 grayscale, but MobileNetV2 was pretrained on 96x96 RGB
ImageNet crops. The grayscale plane is replicated into three channels and the
image is upsampled to 96x96 so that the pretrained filters operate at their
native scale.

Two stages, exactly as the specification lays out:

    E3  freeze the backbone, train only the new classifier head
    E4  unfreeze the last FINE_TUNE_UNFREEZE_LAST_N blocks, train at a much
        smaller learning rate
"""

from __future__ import annotations

import numpy as np
from keras import layers, models, optimizers
from keras.applications import MobileNetV2

from . import config
from .augmentation import preprocessing_block


def build_transfer_model(
    augment: bool = True,
    fine_tune: bool = False,
    weights: str | None = "imagenet",
    seed: int = config.SEED,
    compile_model: bool = True,
    learning_rate: float | None = None,
    name: str = "mobilenet_v2_transfer",
) -> models.Model:
    """Build the MobileNetV2 classifier.

    ``fine_tune=False`` freezes the whole backbone (section 10).
    ``fine_tune=True`` re-enables the trailing blocks only (section 11).
    """
    spec = config.preprocessing("mobilenet")

    backbone = MobileNetV2(
        input_shape=(spec["size"], spec["size"], spec["channels"]),
        include_top=False,
        weights=weights,
    )

    inputs = layers.Input(shape=(spec["size"], spec["size"], spec["channels"]), name="input")
    x = inputs
    for layer in preprocessing_block(augment, "mobilenet", seed=seed):
        x = layer(x)

    if fine_tune:
        backbone.trainable = True
        for layer in backbone.layers[:-config.FINE_TUNE_UNFREEZE_LAST_N]:
            layer.trainable = False
    else:
        backbone.trainable = False

    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x)
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="emotion")(x)

    model = models.Model(inputs, outputs, name=name)

    if compile_model:
        # Fine-tuning needs a much smaller learning rate than the frozen stage.
        lr = learning_rate
        if lr is None:
            lr = config.FINE_TUNE_LEARNING_RATE if fine_tune else config.LEARNING_RATE
        model.compile(
            optimizer=optimizers.Adam(learning_rate=lr),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    model.trainable_frozen_blocks = sum(1 for layer in backbone.layers if not layer.trainable)
    return model


def describe(model: models.Model) -> str:
    total = model.count_params()
    trainable = sum(
        int(np.prod(layer.shape)) for layer in model.trainable_weights if layer.shape
    )
    return (
        f"{model.name}: {total:,} total parameters, {trainable:,} trainable "
        f"({100 * trainable / max(total, 1):.1f}%)"
    )
