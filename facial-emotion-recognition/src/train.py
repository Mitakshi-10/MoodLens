"""Training harness for the section-12 experiment matrix.

One experiment per invocation so long runs are independent, resumable and
skippable:

    python -m src.train --experiment E1
    python -m src.train --experiment E4 --epochs 20
    python -m src.train --experiment E2 --fast        # short iteration run

Training configuration follows sections 8 and 9: Adam, categorical
cross-entropy, batch size 64, with early stopping and learning-rate decay.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from keras import callbacks, utils

from . import config
from .cnn_model import build_cnn
from .data import build_dataset
from .transfer_model import build_transfer_model, describe


def resolve_epochs(experiment: dict, override: int | None) -> int:
    if override is not None:
        return override
    return experiment["epochs"]


def build_for_experiment(experiment: dict, compile_model: bool = True):
    """Instantiate the right architecture with the right training flags."""
    if experiment["model"] == "cnn":
        return build_cnn(
            augment=experiment["augment"],
            compile_model=compile_model,
            name=f"{experiment['name'].lower()}_cnn",
        )
    if experiment["model"] == "mobilenet":
        return build_transfer_model(
            augment=experiment["augment"],
            fine_tune=experiment["fine_tune"],
            compile_model=compile_model,
            name=f"{experiment['name'].lower()}_mobilenetv2",
        )
    raise ValueError(f"Unknown model kind {experiment['model']!r}")


def make_callbacks(experiment_name: str, restore_best: bool = True) -> list[callbacks.Callback]:
    """Checkpointing, early stopping and LR decay (section 9)."""
    return [
        callbacks.ModelCheckpoint(
            filepath=str(config.model_path(experiment_name)),
            monitor=config.EARLY_STOPPING_MONITOR,
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor=config.EARLY_STOPPING_MONITOR,
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=restore_best,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=config.LR_PLATEAU_FACTOR,
            patience=config.LR_PLATEAU_PATIENCE,
            min_lr=config.MIN_LR,
            verbose=1,
        ),
        callbacks.CSVLogger(str(config.history_csv(experiment_name)), append=False),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train one experiment.")
    parser.add_argument(
        "--experiment", required=True, choices=sorted(config.EXPERIMENTS),
        help="experiment id from section 12",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override the default count")
    parser.add_argument("--fast", action="store_true", help="2-epoch smoke run")
    parser.add_argument("--class-weight", action="store_true", default=True)
    parser.add_argument("--no-class-weight", dest="class_weight", action="store_false")
    parser.add_argument("--cache", action="store_true", help="cache decoded batches in RAM")
    parser.add_argument("--skip-if-exists", action="store_true")
    parser.add_argument("--no-fine-tune", action="store_true", help="for E4, train frozen only")
    args = parser.parse_args(argv)

    experiment = config.get_experiment(args.experiment)
    if args.no_fine_tune:
        experiment["fine_tune"] = False

    model_file = config.model_path(experiment["name"])
    if args.skip_if_exists and model_file.exists():
        print(f"{model_file.name} already exists; skipping. Use --force to retrain.")
        return 0

    utils.set_random_seed(config.SEED)
    config.ensure_dirs()

    epochs = 2 if args.fast else resolve_epochs(experiment, args.epochs)
    print("=" * 68)
    print(f"Experiment {experiment['name']}: {experiment['label']}")
    print(
        f"  model={experiment['model']}  augment={experiment['augment']}  "
        f"transfer={experiment['transfer']}  fine_tune={experiment['fine_tune']}"
    )
    print(f"  epochs={epochs}  batch_size={config.BATCH_SIZE}  seed={config.SEED}")
    print("=" * 68)

    train_ds = build_dataset(
        "train", experiment["model"], training=True, cache=args.cache
    )
    val_ds = build_dataset("validation", experiment["model"], training=False)

    if args.fast:
        train_ds = train_ds.take(60)
        val_ds = val_ds.take(8)

    model = build_for_experiment(experiment)
    print(describe(model))
    model.summary(print_fn=lambda line: None)

    class_weight = config.class_weights() if args.class_weight else None
    if class_weight:
        print(f"  class_weight=balanced ({ {k: round(v, 2) for k, v in class_weight.items()} })")

    started = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=make_callbacks(experiment["name"]),
        class_weight=class_weight,
        verbose=2,
    )
    elapsed = time.time() - started

    best_epoch = int(max(range(len(history.history["val_accuracy"])),
                        key=lambda i: history.history["val_accuracy"][i]))
    summary = {
        "experiment": experiment["name"],
        "label": experiment["label"],
        "epochs_completed": len(history.history["loss"]),
        "epochs_requested": epochs,
        "seconds": round(elapsed, 1),
        "best_epoch": best_epoch + 1,
        "best_val_accuracy": float(max(history.history["val_accuracy"])),
        "best_val_loss": float(min(history.history["val_loss"])),
        "final_train_accuracy": float(history.history["accuracy"][-1]),
        "config": {
            "model": experiment["model"],
            "augment": experiment["augment"],
            "transfer": experiment["transfer"],
            "fine_tune": experiment["fine_tune"],
            "batch_size": config.BATCH_SIZE,
            "class_weight": args.class_weight,
            "seed": config.SEED,
        },
    }

    summary_file = config.RESULTS_DIR / f"{experiment['name']}_train_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 68)
    print(f"Finished in {elapsed / 60:.1f} min")
    print(f"Best val accuracy : {summary['best_val_accuracy'] * 100:.2f}% (epoch {best_epoch + 1})")
    print(f"Best val loss     : {summary['best_val_loss']:.4f}")
    print(f"Saved model       : {model_file.relative_to(config.BASE_DIR)}")
    print(f"History           : {config.history_csv(experiment['name']).relative_to(config.BASE_DIR)}")
    print(f"Summary           : {summary_file.relative_to(config.BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
