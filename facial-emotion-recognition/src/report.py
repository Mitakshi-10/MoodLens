"""Aggregate experiment results into the section-12 comparison table.

Reads every ``results/<exp>_metrics.json`` produced by src.evaluate and writes:

    results/comparison_table.csv
    results/comparison_table.md
    results/accuracy_comparison.png
    results/confusion_matrices.png   (one panel per experiment)

Specification section 12 asks for the table to be filled with measured values
rather than assumed ones, which is exactly what this module does: any
experiment that has not been evaluated is reported as missing instead of being
invented.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from . import config

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

sns.set_theme(style="whitegrid")


def load_metrics() -> dict[str, dict]:
    metrics = {}
    for name in config.EXPERIMENTS:
        path = config.metrics_json(name)
        if path.exists():
            metrics[name] = json.loads(path.read_text(encoding="utf-8"))
    return metrics


def build_table(metrics: dict[str, dict]) -> pd.DataFrame:
    rows = []
    for name, cfg in config.EXPERIMENTS.items():
        entry = metrics.get(name)
        row = {
            "Experiment": name,
            "Description": cfg["label"],
            "Augmentation": "Yes" if cfg["augment"] else "No",
            "Transfer Learning": "Yes" if cfg["transfer"] else "No",
            "Fine-Tuning": "Yes" if cfg["fine_tune"] else "No",
        }
        if entry and "test" in entry:
            test = entry["test"]
            row["Test Accuracy"] = test["accuracy"]
            row["Macro F1"] = test["macro_f1"]
            row["Weighted F1"] = test["weighted_f1"]
            row["Top-3 Accuracy"] = test.get("top3_accuracy", float("nan"))
            row["Val Accuracy"] = entry.get("validation", {}).get("accuracy", float("nan"))
            row["Val Loss"] = float(entry.get("training", {}).get("best_val_loss", float("nan")))
            row["Epochs"] = entry.get("epochs_trained", float("nan"))
            row["Minutes"] = entry.get("training", {}).get("seconds", float("nan")) / 60.0
        else:
            for key in ("Test Accuracy", "Macro F1", "Weighted F1", "Top-3 Accuracy",
                        "Val Accuracy", "Val Loss", "Epochs", "Minutes"):
                row[key] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(table: pd.DataFrame, metrics: dict[str, dict]) -> str:
    """Render the section-12 table, keeping the specification's own column layout."""
    lines = [
        "# Experiment comparison",
        "",
        "Measured on the held-out FER-2013 PrivateTest split (3,589 images), "
        "which was never used for training or model selection.",
        "",
        "| Experiment | Augmentation | Transfer Learning | Fine-Tuning | Accuracy | F1 |",
        "|---|---|---|---|---:|---:|",
    ]
    for _, row in table.iterrows():
        if np.isnan(row["Test Accuracy"]):
            acc = f1 = "not evaluated"
        else:
            acc = f"{row['Test Accuracy'] * 100:.2f}%"
            f1 = f"{row['Macro F1'] * 100:.2f}%"
        lines.append(
            f"| {row['Experiment']} | {row['Augmentation']} | "
            f"{row['Transfer Learning']} | {row['Fine-Tuning']} | {acc} | {f1} |"
        )

    lines += [
        "",
        "## Extended metrics",
        "",
        "| Experiment | Description | Val Acc | Test Acc | Macro F1 | Weighted F1 | "
        "Top-3 Acc | ECE | Epochs | Minutes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value, pct=True):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "n/a"
        return f"{value * 100:.2f}%" if pct else f"{value:.2f}"

    for _, row in table.iterrows():
        entry = metrics.get(row["Experiment"], {})
        ece = entry.get("test", {}).get("expected_calibration_error", float("nan"))
        minutes = fmt(row["Minutes"], pct=False)
        epochs = "n/a" if np.isnan(row["Epochs"]) else str(int(row["Epochs"]))
        lines.append(
            f"| {row['Experiment']} | {row['Description']} | "
            f"{fmt(row['Val Accuracy'])} | {fmt(row['Test Accuracy'])} | "
            f"{fmt(row['Macro F1'])} | {fmt(row['Weighted F1'])} | "
            f"{fmt(row['Top-3 Accuracy'])} | {fmt(ece, pct=False)} | "
            f"{epochs} | {minutes} |"
        )

    per_class_rows = []
    for name, entry in metrics.items():
        per_class = entry.get("test", {}).get("per_class")
        if per_class:
            for emotion, values in per_class.items():
                per_class_rows.append({"Experiment": name, "Emotion": emotion, **values})
    if per_class_rows:
        frame = pd.DataFrame(per_class_rows)
        pivot = frame.pivot(index="Emotion", columns="Experiment", values="f1")
        order = [e for e in config.EMOTIONS if e in pivot.index]
        pivot = pivot.reindex(order)
        pivot = (pivot * 100).round(2)
        lines += [
            "",
            "## Per-class F1 on the test split (%)",
            "",
            "Disgust has only 55 test images, so its scores carry a large standard "
            "error and should not be over-interpreted.",
            "",
            pivot.to_markdown(),
            "",
            "### Test support per class",
            "",
            frame[frame["Experiment"] == frame["Experiment"].iloc[0]]
            .set_index("Emotion")
            .reindex(order)["support"]
            .to_frame("support")
            .to_markdown(),
        ]

    confusion_paths = [
        config.RESULTS_DIR / f"confusion_matrix_{name}_test.png"
        for name in metrics
        if (config.RESULTS_DIR / f"confusion_matrix_{name}_test.png").exists()
    ]
    if confusion_paths:
        lines += [
            "",
            "## Figures",
            "",
            *[f"- `results/{p.name}`" for p in confusion_paths],
            "- `results/accuracy_comparison.png`",
            "- `results/confusion_matrices.png`",
        ]
    return "\n".join(lines)


def plot_accuracy_comparison(table: pd.DataFrame, metrics: dict[str, dict]) -> None:
    evaluated = table.dropna(subset=["Test Accuracy"])
    if evaluated.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(evaluated))
    width = 0.38
    ax.bar(x - width / 2, evaluated["Val Accuracy"], width, label="Validation (PublicTest)",
           color="#A8C4E0")
    ax.bar(x + width / 2, evaluated["Test Accuracy"], width, label="Test (PrivateTest)",
           color="#4C72B0")
    ax.set_xticks(x)
    ax.set_xticklabels(evaluated["Experiment"])
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title("Held-out accuracy by experiment")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    for i, row in enumerate(evaluated.itertuples()):
        ax.text(i + width / 2, row.Test_Accuracy + 0.015, f"{row.Test_Accuracy * 100:.1f}%",
                ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix_grid(metrics: dict[str, dict]) -> None:
    available = {
        name: np.array(entry["test"]["confusion_matrix"])
        for name, entry in metrics.items()
        if entry.get("test", {}).get("confusion_matrix")
    }
    if not available:
        return

    n = len(available)
    cols = min(n, 2)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5.4 * rows), squeeze=False)

    for ax, (name, cm) in zip(axes.ravel(), available.items()):
        normalised = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(
            normalised, annot=cm, fmt="d", cmap="Blues", cbar=False, square=True,
            xticklabels=config.EMOTIONS, yticklabels=config.EMOTIONS, ax=ax,
        )
        ax.set_title(f"{name} - test split (row-normalised)")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    for ax in axes.ravel()[n:]:
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "confusion_matrices.png", dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the comparison report.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    metrics = load_metrics()
    if not metrics:
        print("No metrics found. Run `python -m src.evaluate --all` first.")
        return 1

    table = build_table(metrics)
    csv_path = config.RESULTS_DIR / "comparison_table.csv"
    md_path = config.RESULTS_DIR / "comparison_table.md"
    table.to_csv(csv_path, index=False)
    md_path.write_text(to_markdown(table, metrics), encoding="utf-8")

    plot_accuracy_comparison(table, metrics)
    plot_confusion_matrix_grid(metrics)

    if not args.quiet:
        display = table[["Experiment", "Description", "Test Accuracy", "Macro F1",
                         "Top-3 Accuracy", "Epochs"]].copy()
        for column in ("Test Accuracy", "Macro F1", "Top-3 Accuracy"):
            display[column] = (display[column] * 100).round(2).astype(object).where(
                display[column].notna(), "n/a"
            )
        print(display.to_string(index=False))
        print(f"\nWrote {csv_path.relative_to(config.BASE_DIR)}")
        print(f"Wrote {md_path.relative_to(config.BASE_DIR)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())