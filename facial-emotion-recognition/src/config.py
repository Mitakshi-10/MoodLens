"""Central configuration for the facial expression recognition project.

This module is the single source of truth for emotion labels, image sizes,
hyper-parameters and the experiment matrix. Training, evaluation and live
inference all import from here so those three paths can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --------------------------------------------------------------------------
# Dataset / classes (specification sections 1 and 2)
# --------------------------------------------------------------------------

EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
NUM_CLASSES = len(EMOTIONS)
EMOTION_SLUGS = [name.lower() for name in EMOTIONS]

SPLITS = ("train", "validation", "test")

RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

FER_CSV = RAW_DIR / "fer2013.csv"
MANIFEST_CSV = DATA_DIR / "manifest.csv"
DISTRIBUTION_FIG = RESULTS_DIR / "class_distribution.png"

# Official fer2013.csv layout. PublicTest is used for validation and
# PrivateTest is held out for the final evaluation, which is how FER-2013 was
# designed to be used. data_prep asserts these numbers so a corrupted or
# substituted dataset fails loudly instead of producing meaningless metrics.
USAGE_TO_SPLIT = {
    "Training": "train",
    "PublicTest": "validation",
    "PrivateTest": "test",
}

EXPECTED_SPLIT_SIZES = {"train": 28709, "validation": 3589, "test": 3589}

EXPECTED_COUNTS = {
    "train": [3995, 436, 4097, 7215, 4830, 3171, 4965],
    "validation": [467, 56, 496, 895, 653, 415, 607],
    "test": [491, 55, 528, 879, 594, 416, 626],
}

# --------------------------------------------------------------------------
# Image geometry
# --------------------------------------------------------------------------

IMG_SIZE = 48
IMG_CHANNELS = 1
TL_IMG_SIZE = 96
TL_IMG_CHANNELS = 3

MODEL_KINDS = ("cnn", "mobilenet")


def preprocessing(model_kind: str) -> dict:
    """Return the preprocessing contract for a model kind.

    Every consumer (dataset pipeline, evaluator, webcam app, Streamlit) derives
    its resizing and normalisation from this function, which guarantees that
    inference sees exactly what training saw.
    """
    if model_kind == "cnn":
        return {
            "size": IMG_SIZE,
            "channels": IMG_CHANNELS,
            "color_mode": "grayscale",
            "scale": "zero_one",  # section 4.2: x / 255
        }
    if model_kind == "mobilenet":
        return {
            "size": TL_IMG_SIZE,
            "channels": TL_IMG_CHANNELS,
            "color_mode": "rgb",
            # ImageNet-pretrained MobileNetV2 requires [-1, 1] inputs, so the
            # /255 rule from section 4.2 is deliberately not applied here.
            "scale": "mobilenet",
        }
    raise ValueError(f"Unknown model_kind {model_kind!r}; expected one of {MODEL_KINDS}")


# --------------------------------------------------------------------------
# Training hyper-parameters (sections 8 and 9)
# --------------------------------------------------------------------------

SEED = 42
BATCH_SIZE = 64
EPOCHS = 35
FT_EPOCHS = 20
LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5

EARLY_STOPPING_PATIENCE = 8
EARLY_STOPPING_MONITOR = "val_accuracy"
LR_PLATEAU_PATIENCE = 3
LR_PLATEAU_FACTOR = 0.5
MIN_LR = 1e-6

# Number of trainable samples as a fraction, used by the --fast profile.
FAST_TRAIN_FRACTION = 0.15
FAST_VALIDATION_FRACTION = 0.25

# --------------------------------------------------------------------------
# Augmentation (section 5)
# --------------------------------------------------------------------------

FLIP_MODE = "horizontal"
ROTATION_FACTOR = 0.20  # roughly +/- 11.5 degrees
ZOOM_FACTOR = 0.10
TRANSLATION_FACTOR = 0.10
BRIGHTNESS_FACTOR = 0.20

# --------------------------------------------------------------------------
# Experiment matrix (section 12)
# --------------------------------------------------------------------------

EXPERIMENTS = {
    "E1": {
        "label": "Basic CNN",
        "model": "cnn",
        "augment": False,
        "transfer": False,
        "fine_tune": False,
        "epochs": EPOCHS,
    },
    "E2": {
        "label": "CNN + Augmentation",
        "model": "cnn",
        "augment": True,
        "transfer": False,
        "fine_tune": False,
        "epochs": EPOCHS,
    },
    "E3": {
        "label": "Transfer Learning",
        "model": "mobilenet",
        "augment": True,
        "transfer": True,
        "fine_tune": False,
        "epochs": EPOCHS,
    },
    "E4": {
        "label": "Transfer Learning + Fine-Tuning",
        "model": "mobilenet",
        "augment": True,
        "transfer": True,
        "fine_tune": True,
        "epochs": FT_EPOCHS,
    },
}

BEST_EXPERIMENT = "E4"

# MobileNetV2 blocks kept trainable during fine-tuning (section 11).
FINE_TUNE_UNFREEZE_LAST_N = 30


def get_experiment(name: str) -> dict:
    key = name.upper()
    if key not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {name!r}; expected one of {sorted(EXPERIMENTS)}")
    return dict(EXPERIMENTS[key], name=key)


def class_weights() -> dict[int, float]:
    """Inverse-frequency class weights for the training split.

    FER-2013 is heavily imbalanced (436 Disgust vs 7215 Happy in train), so
    unweighted training biases the model away from the rare classes.
    """
    counts = EXPECTED_COUNTS["train"]
    total = sum(counts)
    return {i: total / (NUM_CLASSES * c) for i, c in enumerate(counts)}


def model_path(experiment: str) -> Path:
    return MODELS_DIR / f"{experiment.upper()}.best.keras"


def history_csv(experiment: str) -> Path:
    return RESULTS_DIR / f"{experiment.upper()}_history.csv"


def metrics_json(experiment: str) -> Path:
    return RESULTS_DIR / f"{experiment.upper()}_metrics.json"


def ensure_dirs() -> None:
    for path in (DATA_DIR, MODELS_DIR, RESULTS_DIR):
        path.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for slug in EMOTION_SLUGS:
            (DATA_DIR / split / slug).mkdir(parents=True, exist_ok=True)
