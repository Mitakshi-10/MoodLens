"""Model evaluation (specification sections 13, 14 and 15).

Produces, for one experiment:

    results/<exp>_metrics.json              headline numbers
    results/<exp>_classification_report.csv per-class precision/recall/F1
    results/confusion_matrix_<exp>.png      counts and row-normalised (section 14)
    results/accuracy_curve_<exp>.png        training vs validation (section 15)
    results/loss_curve_<exp>.png            training vs validation (section 15)
    results/calibration_<exp>.png           reliability + confidence spread
    results/error_analysis/<exp>_errors.png most confident mistakes

    python -m src.evaluate --experiment E1
    python -m src.evaluate --all
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config
from .data import build_dataset

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn import metrics as skmetrics  # noqa: E402

sns.set_theme(style="whitegrid")

ERROR_SHEET_DIR = config.RESULTS_DIR / "error_analysis"


def load_model(experiment: str):
    """Load a trained model for inference (no optimizer state needed)."""
    from keras import models

    path = config.model_path(experiment)
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not found. Train it first:\n"
            f"  python -m src.train --experiment {experiment.upper()}"
        )
    return models.load_model(path, compile=False)


def predict_split(model, model_kind: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (images_uint8, y_true_onehot, probabilities) for a split."""
    dataset = build_dataset(split, model_kind, training=False)
    images, labels, probabilities = [], [], []
    for batch_images, batch_labels in dataset:
        images.append(batch_images.numpy())
        labels.append(batch_labels.numpy())
        probabilities.append(model.predict(batch_images, verbose=0))
    return (
        np.concatenate(images),
        np.concatenate(labels),
        np.concatenate(probabilities),
    )


def calibration_curve(y_true_idx, y_pred_idx, confidence, bins=10):
    """Standard expected calibration error over equal-width confidence bins.

    Returns (ece, rows) where each row describes one bin, for the reliability
    diagram plotted by plot_calibration.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    rows = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (confidence > low) & (confidence <= high)
        if not mask.any():
            rows.append({"low": float(low), "high": float(high), "count": 0,
                         "confidence": 0.0, "accuracy": 0.0})
            continue
        bucket_acc = float(np.mean(y_pred_idx[mask] == y_true_idx[mask]))
        bucket_conf = float(confidence[mask].mean())
        ece += float(mask.mean()) * abs(bucket_acc - bucket_conf)
        rows.append({"low": float(low), "high": float(high), "count": int(mask.sum()),
                     "confidence": bucket_conf, "accuracy": bucket_acc})
    return ece, rows


def plot_confusion_matrix(cm: np.ndarray, title: str, path, normalize: bool = False) -> None:
    matrix = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalize else cm

    fig, ax = plt.subplots(figsize=(7.2, 6))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=config.EMOTIONS,
        yticklabels=config.EMOTIONS,
        ax=ax,
        square=True,
        cbar=False,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_curves(history: pd.DataFrame, experiment: str) -> None:
    if history.empty or "epoch" not in history.columns:
        return
    epochs = history["epoch"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["accuracy"], label="Training accuracy", linewidth=2)
    ax.plot(epochs, history["val_accuracy"], label="Validation accuracy", linewidth=2)
    best = history.loc[history["val_accuracy"].idxmax()]
    ax.axvline(best["epoch"], color="grey", linestyle="--", alpha=0.6,
               label=f"Best val epoch ({int(best['epoch'])})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{experiment} - Accuracy")
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"accuracy_curve_{experiment}.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, history["loss"], label="Training loss", linewidth=2)
    ax.plot(epochs, history["val_loss"], label="Validation loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Categorical cross-entropy loss")
    ax.set_title(f"{experiment} - Loss")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"loss_curve_{experiment}.png", dpi=150)
    plt.close(fig)


def plot_calibration(rows, ece: float, confidences: np.ndarray, experiment: str) -> None:
    populated = [r for r in rows if r["count"] > 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

    ax = axes[0]
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.plot([r["confidence"] for r in populated], [r["accuracy"] for r in populated],
            "o-", color="#C44E52", linewidth=2, label="Model")
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.hist(confidences, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(confidences.mean(), color="#C44E52", linestyle="--",
               label=f"mean = {confidences.mean():.3f}")
    ax.set_xlabel("Top-1 confidence")
    ax.set_ylabel("Number of test images")
    ax.set_title("Confidence distribution")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"calibration_{experiment}.png", dpi=150)
    plt.close(fig)


def plot_errors(
    images: np.ndarray,
    y_true_idx: np.ndarray,
    y_pred_idx: np.ndarray,
    probabilities: np.ndarray,
    experiment: str,
    count: int = 24,
) -> list[dict]:
    """Contact sheet of the most confident mistakes, answering section 25 Q9."""
    ERROR_SHEET_DIR.mkdir(parents=True, exist_ok=True)
    confidence = probabilities.max(axis=1)
    wrong = np.where(y_true_idx != y_pred_idx)[0]
    if wrong.size == 0:
        return []
    order = wrong[np.argsort(-confidence[wrong])][:count]

    cols = 8
    rows = int(np.ceil(len(order) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.75))
    for ax in axes.ravel():
        ax.axis("off")

    records = []
    for slot, idx in enumerate(order):
        ax = axes.ravel()[slot]
        img = images[idx]
        if img.ndim == 3:
            img = img[:, :, 0] if img.shape[2] == 1 else img.mean(axis=2)
        ax.imshow(np.clip(img, 0, 255).astype(np.uint8), cmap="gray")
        ax.set_title(
            f"{config.EMOTIONS[y_true_idx[idx]]}\n-> {config.EMOTIONS[y_pred_idx[idx]]} "
            f"{confidence[idx] * 100:.0f}%",
            fontsize=6.5,
            color="#C44E52",
        )
        records.append({
            "index": int(idx),
            "true": config.EMOTIONS[y_true_idx[idx]],
            "predicted": config.EMOTIONS[y_pred_idx[idx]],
            "confidence": float(confidence[idx]),
        })

    fig.suptitle(
        f"{experiment} - {len(order)} most confident mistakes on the held-out test split",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ERROR_SHEET_DIR / f"{experiment}_errors.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return records


def evaluate_experiment(experiment_name: str) -> dict:
    experiment = config.get_experiment(experiment_name)
    model_kind = experiment["model"]

    print("=" * 68)
    print(f"Evaluating {experiment['name']}: {experiment['label']}")
    print("=" * 68)
    model = load_model(experiment["name"])

    results = {}
    for split in ("validation", "test"):
        images, y_true, probs = predict_split(model, model_kind, split)
        y_true_idx = y_true.argmax(axis=1)
        y_pred_idx = probs.argmax(axis=1)

        accuracy = float((y_true_idx == y_pred_idx).mean())
        macro = skmetrics.f1_score(y_true_idx, y_pred_idx, average="macro", zero_division=0)
        weighted = skmetrics.f1_score(y_true_idx, y_pred_idx, average="weighted", zero_division=0)
        cm = skmetrics.confusion_matrix(y_true_idx, y_pred_idx,
                                        labels=list(range(config.NUM_CLASSES)))

        report = skmetrics.classification_report(
            y_true_idx,
            y_pred_idx,
            target_names=config.EMOTIONS,
            output_dict=True,
            zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(
            config.RESULTS_DIR / f"{experiment['name']}_{split}_classification_report.csv"
        )

        results[split] = {
            "accuracy": accuracy,
            "macro_f1": float(macro),
            "weighted_f1": float(weighted),
            "top1_accuracy": accuracy,
            "per_class": {
                name: {
                    "precision": float(report[name]["precision"]),
                    "recall": float(report[name]["recall"]),
                    "f1": float(report[name]["f1-score"]),
                    "support": int(report[name]["support"]),
                }
                for name in config.EMOTIONS
            },
            "confusion_matrix": cm.tolist(),
        }

        print(f"  {split:11s} accuracy={accuracy * 100:6.2f}%  macro-F1={macro * 100:6.2f}%  "
              f"weighted-F1={weighted * 100:6.2f}%")

        plot_confusion_matrix(
            cm, f"{experiment['name']} - Confusion matrix ({split}, counts)",
            config.RESULTS_DIR / f"confusion_matrix_{experiment['name']}_{split}.png",
        )
        if split == "test":
            plot_confusion_matrix(
                cm, f"{experiment['name']} - Confusion matrix (test, row-normalised)",
                config.RESULTS_DIR / f"confusion_matrix_{experiment['name']}_test_normalised.png",
                normalize=True,
            )

            confidence = probs.max(axis=1)
            ece, rows = calibration_curve(y_true_idx, y_pred_idx, confidence)
            plot_calibration(rows, ece, confidence, experiment["name"])
            errors = plot_errors(images, y_true_idx, y_pred_idx, probs, experiment["name"])

            results["test"]["expected_calibration_error"] = ece
            results["test"]["mean_confidence"] = float(confidence.mean())
            results["test"]["confidence_when_correct"] = float(
                confidence[y_true_idx == y_pred_idx].mean()
            )
            results["test"]["confidence_when_wrong"] = float(
                confidence[y_true_idx != y_pred_idx].mean()
            )
            results["test"]["top3_accuracy"] = float(
                np.mean([y_true_idx[i] in np.argsort(-probs[i])[:3] for i in range(len(probs))])
            )
            results["test"]["most_confident_errors"] = errors
            print(f"  test       ECE={ece:.4f}  mean confidence={confidence.mean():.4f}  "
                  f"top-3 accuracy={results['test']['top3_accuracy'] * 100:.2f}%")

    history_file = config.history_csv(experiment["name"])
    history = pd.read_csv(history_file) if history_file.exists() else pd.DataFrame()
    if not history.empty:
        plot_curves(history, experiment["name"])
        results["epochs_trained"] = int(history["epoch"].max()) + 1

    train_summary = config.RESULTS_DIR / f"{experiment['name']}_train_summary.json"
    if train_summary.exists():
        results["training"] = json.loads(train_summary.read_text(encoding="utf-8"))

    results["experiment"] = experiment["name"]
    results["label"] = experiment["label"]
    results["config"] = experiment

    out = config.metrics_json(experiment["name"])
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"  metrics -> {out.relative_to(config.BASE_DIR)}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate trained experiments.")
    parser.add_argument("--experiment", choices=sorted(config.EXPERIMENTS))
    parser.add_argument("--all", action="store_true", help="evaluate every trained model")
    args = parser.parse_args(argv)

    if args.all:
        targets = [e for e in sorted(config.EXPERIMENTS) if config.model_path(e).exists()]
        if not targets:
            print("No trained models found. Run run_all_experiments.ps1 first.")
            return 1
    elif args.experiment:
        targets = [args.experiment]
    else:
        parser.error("pass --experiment or --all")

    for target in targets:
        try:
            evaluate_experiment(target)
        except FileNotFoundError as exc:
            print(f"Skipping {target}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())