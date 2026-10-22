"""tf.data pipelines over the ImageFolder tree.

Datasets emit images as float32 in the 0-255 range with one-hot categorical
labels. All rescaling and normalisation happens inside the model (see
src/augmentation.py), which means the same saved ``.keras`` file carries its
own preprocessing and can be fed a raw image crop by the webcam application
with no extra steps.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras

from . import config

AUTOTUNE = tf.data.AUTOTUNE


def build_dataset(
    split: str,
    model_kind: str,
    training: bool,
    seed: int = config.SEED,
    cache: bool = False,
) -> tf.data.Dataset:
    """Return a batched dataset for ``split`` shaped for ``model_kind``.

    Section 5 is honoured here: augmentation is never applied to the validation
    or test splits because those must represent unseen data.
    """
    if split not in config.SPLITS:
        raise ValueError(f"split must be one of {config.SPLITS}, got {split!r}")

    spec = config.preprocessing(model_kind)
    directory = config.DATA_DIR / split
    if not directory.is_dir():
        raise FileNotFoundError(
            f"Missing {directory}. Run `python -m src.data_prep` first."
        )

    dataset = keras.utils.image_dataset_from_directory(
        str(directory),
        # Alphabetical folder sorting would order the classes as
        # angry, disgust, fear, happy, neutral, sad, surprise, which scrambles
        # Sad/Neutral/Surprise relative to the FER-2013 labels 4/5/6. Passing
        # class_names pins the canonical order explicitly.
        class_names=list(config.EMOTION_SLUGS),
        labels="inferred",
        label_mode="categorical",
        color_mode=spec["color_mode"],
        image_size=(spec["size"], spec["size"]),
        batch_size=config.BATCH_SIZE,
        shuffle=training,
        seed=seed if training else None,
    )

    # Belt-and-braces: never trust the inferred order silently.
    inferred = [str(name) for name in dataset.class_names]
    if inferred != list(config.EMOTION_SLUGS):
        raise ValueError(
            f"Class order mismatch for {split!r}.\n"
            f"  inferred: {inferred}\n  expected: {list(config.EMOTION_SLUGS)}"
        )

    if cache:
        # Caches the decoded uint8 batches, which are small for the custom CNN
        # (66 MB for the grayscale train split).
        dataset = dataset.cache()

    return dataset.prefetch(AUTOTUNE)


def count_examples(dataset: tf.data.Dataset) -> int:
    return int(tf.data.experimental.cardinality(dataset))
