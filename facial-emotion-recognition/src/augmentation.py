"""Data augmentation and normalisation front-ends (specification sections 4 and 5).

Augmentation is modelled as Keras layers prepended to the network rather than
applied to the dataset. Three reasons:

1. The transform is stored inside the saved ``.keras`` file, so the webcam and
   Streamlit code paths cannot accidentally diverge from training.
2. It only ever runs on the training split, because the block is only attached
   when ``augment=True``.
3. Keras layers are vectorised, which is considerably faster on CPU than the
   per-image PIL transforms in ``ImageDataGenerator``.

Section 5 lists rotation, horizontal flip, zoom, translation and brightness;
all five are applied, each with a fixed seed so runs stay reproducible.
"""

from __future__ import annotations

from keras import layers

from . import config


def augmentation_layers(seed: int = config.SEED) -> list[layers.Layer]:
    """Return the section-5 augmentation stack.

    Expects float32 images in the 0-255 range (see ``normalisation`` below,
    which casts uint8 input without altering the value range).
    """
    return [
        layers.RandomFlip(config.FLIP_MODE, seed=seed),
        layers.RandomRotation(
            config.ROTATION_FACTOR, fill_mode="reflect", seed=seed + 1
        ),
        layers.RandomZoom(config.ZOOM_FACTOR, fill_mode="reflect", seed=seed + 2),
        layers.RandomTranslation(
            config.TRANSLATION_FACTOR,
            config.TRANSLATION_FACTOR,
            fill_mode="reflect",
            seed=seed + 3,
        ),
        layers.RandomBrightness(config.BRIGHTNESS_FACTOR, value_range=(0, 255), seed=seed + 4),
    ]


def normalisation(model_kind: str) -> list[layers.Layer]:
    """Return the layers that convert 0-255 pixel values into model-ready floats.

    ``Rescaling(1.0)`` keeps values in the 0-255 range while guaranteeing a
    float32 input dtype, which is what ``RandomBrightness`` expects. It is a
    no-op for the float32 batches ``image_dataset_from_directory`` already
    produces, and does the cast for integer webcam frames.
    """
    spec = config.preprocessing(model_kind)
    layers_out: list[layers.Layer] = [layers.Rescaling(1.0)]

    if spec["scale"] == "zero_one":
        # Section 4.2: x_normalized = x / 255
        layers_out.append(layers.Rescaling(1.0 / 255.0))
    elif spec["scale"] == "mobilenet":
        # MobileNetV2's ImageNet preprocessing: x / 127.5 - 1  ->  [-1, 1]
        layers_out.append(layers.Rescaling(2.0 / 255.0, offset=-1.0))
    else:
        raise ValueError(f"Unknown scale {spec['scale']!r}")

    return layers_out


def preprocessing_block(
    augment: bool, model_kind: str, seed: int = config.SEED
) -> list[layers.Layer]:
    """Full preprocessing front-end: optional augmentation, then normalisation."""
    block: list[layers.Layer] = []
    if augment:
        block.extend(augmentation_layers(seed=seed))
    block.extend(normalisation(model_kind))
    return block
