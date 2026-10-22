"""Custom CNN for 48x48 grayscale facial expression classification.

This follows the architecture in specification section 6 layer for layer:

    Input 48x48x1
      Conv2D 32  -> ReLU -> BatchNorm -> MaxPool -> Dropout
      Conv2D 64  -> ReLU -> BatchNorm -> MaxPool -> Dropout
      Conv2D 128 -> ReLU                -> MaxPool
      Flatten -> Dense 128 -> Dropout -> Dense 7 -> Softmax

Two deliberate refinements over the diagram, both documented in the report:
``padding="same"`` keeps the spatial resolution intact through the first two
blocks at 48x48, and the network is built with the Keras functional API so the
preprocessing front-end can be attached as named layers.
"""

from __future__ import annotations

from keras import layers, models, optimizers

from . import config
from .augmentation import preprocessing_block


def conv_block(
    filters: int,
    dropout: float | None = None,
    batch_norm: bool = True,
    name: str = "",
) -> list[layers.Layer]:
    """Conv -> ReLU -> (BatchNorm) -> MaxPool -> (Dropout)."""
    block: list[layers.Layer] = [
        layers.Conv2D(filters, (3, 3), padding="same", name=f"{name}conv"),
        layers.ReLU(name=f"{name}relu"),
    ]
    if batch_norm:
        block.append(layers.BatchNormalization(name=f"{name}bn"))
    block.append(layers.MaxPooling2D((2, 2), name=f"{name}pool"))
    if dropout is not None:
        block.append(layers.Dropout(dropout, name=f"{name}drop"))
    return block


def build_cnn(
    augment: bool = False,
    seed: int = config.SEED,
    compile_model: bool = True,
    learning_rate: float = config.LEARNING_RATE,
    name: str = "custom_cnn",
) -> models.Model:
    """Build the custom CNN described in section 6."""
    spec = config.preprocessing("cnn")
    inputs = layers.Input(shape=(spec["size"], spec["size"], spec["channels"]), name="input")
    x = inputs
    for layer in preprocessing_block(augment, "cnn", seed=seed):
        x = layer(x)

    for layer in conv_block(32, dropout=0.25, batch_norm=True, name="block1_"):
        x = layer(x)
    for layer in conv_block(64, dropout=0.25, batch_norm=True, name="block2_"):
        x = layer(x)
    for layer in conv_block(128, dropout=None, batch_norm=False, name="block3_"):
        x = layer(x)

    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(128, activation="relu", name="dense")(x)
    x = layers.Dropout(0.5, name="dropout_dense")(x)
    # Section 2: the final layer is a 7-neuron softmax over the emotion classes.
    outputs = layers.Dense(config.NUM_CLASSES, activation="softmax", name="emotion")(x)

    model = models.Model(inputs, outputs, name=name)
    if compile_model:
        model.compile(
            optimizer=optimizers.Adam(learning_rate=learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
    return model
