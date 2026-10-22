"""Build the section-1 ImageFolder tree from the official fer2013.csv.

The original train/validation/test separation is preserved rather than
re-shuffled (specification section 1): ``Training`` becomes ``train``,
``PublicTest`` becomes ``validation`` and ``PrivateTest`` becomes ``test``.

Every split size and per-class count is asserted against the known official
values, so a truncated or substituted download aborts the pipeline instead of
silently producing meaningless metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from . import config

CSV_URL = (
    "https://huggingface.co/datasets/chitradrishti/fer2013/resolve/main/fer2013.csv.zip"
)


def download_csv(force: bool = False) -> Path:
    """Ensure data/raw/fer2013.csv exists, downloading the archive if needed."""
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    if config.FER_CSV.exists() and not force:
        return config.FER_CSV

    import urllib.request
    import zipfile

    archive = config.RAW_DIR / "fer2013.csv.zip"
    if not archive.exists() or force:
        print(f"Downloading {CSV_URL}")
        urllib.request.urlretrieve(CSV_URL, archive)
    print(f"Extracting {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(config.RAW_DIR)
    if not config.FER_CSV.exists():
        raise FileNotFoundError("fer2013.csv missing after extraction")
    return config.FER_CSV


def load_frames(csv_path: Path) -> pd.DataFrame:
    """Read the CSV and turn the space separated pixel string into a 48x48 array."""
    df = pd.read_csv(csv_path)
    required = {"emotion", "pixels", "Usage"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fer2013.csv is missing columns: {sorted(missing)}")

    df["split"] = df["Usage"].map(config.USAGE_TO_SPLIT)
    if df["split"].isna().any():
        unknown = sorted(df.loc[df["split"].isna(), "Usage"].unique())
        raise ValueError(f"Unrecognised Usage values: {unknown}")

    bad = df["emotion"].between(0, config.NUM_CLASSES - 1).all()
    if not bad:
        raise ValueError("emotion column contains labels outside 0..6")
    return df


def verify(df: pd.DataFrame) -> None:
    """Assert the dataset matches the official FER-2013 layout."""
    total = len(df)
    if total != sum(config.EXPECTED_SPLIT_SIZES.values()):
        raise ValueError(f"Expected 35887 rows, found {total}")

    for split, expected_total in config.EXPECTED_SPLIT_SIZES.items():
        subset = df[df["split"] == split]
        if len(subset) != expected_total:
            raise ValueError(
                f"Split {split!r}: expected {expected_total} rows, found {len(subset)}"
            )
        counts = (
            subset["emotion"].value_counts().reindex(range(config.NUM_CLASSES), fill_value=0)
        ).tolist()
        expected_counts = config.EXPECTED_COUNTS[split]
        if counts != expected_counts:
            raise ValueError(
                f"Split {split!r} per-class counts do not match the official dataset.\n"
                f"  expected: {expected_counts}\n  found:    {counts}"
            )
    print(f"Verified {total} images across {len(config.SPLITS)} splits against official counts.")


def _decode(pixels: str) -> np.ndarray:
    arr = np.fromstring(pixels, dtype=np.uint8, sep=" ")
    if arr.size != config.IMG_SIZE * config.IMG_SIZE:
        raise ValueError(f"Expected 2304 pixels, found {arr.size}")
    return arr.reshape(config.IMG_SIZE, config.IMG_SIZE)


def write_tree(df: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Write every image to data/processed/<split>/<emotion>/ as a 48x48 PNG."""
    config.ensure_dirs()

    records = []
    counters = {s: np.zeros(config.NUM_CLASSES, dtype=int) for s in config.SPLITS}
    stem_lengths = set()

    for row in df.itertuples(index=False):
        split = row.split
        label = int(row.emotion)
        slug = config.EMOTION_SLUGS[label]
        out_dir = config.DATA_DIR / split / slug
        filename = f"{split}_{label}_{counters[split][label]:05d}.png"
        out_path = out_dir / filename

        if force or not out_path.exists():
            array = _decode(row.pixels)
            image = Image.fromarray(array, mode="L")
            image.save(out_path, optimize=True)
            stem_lengths.add(len(image.getbands()))

        counters[split][label] += 1
        records.append(
            {
                "path": str(out_path.relative_to(config.BASE_DIR)).replace("\\", "/"),
                "split": split,
                "usage": row.Usage,
                "label": label,
                "emotion": config.EMOTIONS[label],
            }
        )

    if stem_lengths and stem_lengths != {1}:
        raise ValueError(f"Expected single-channel images, got band counts {stem_lengths}")

    manifest = pd.DataFrame(records)
    manifest.to_csv(config.MANIFEST_CSV, index=False)

    print("\nClass distribution")
    table = manifest.groupby(["split", "emotion"]).size().unstack(fill_value=0)
    table = table.reindex(columns=config.EMOTIONS, fill_value=0)
    print(table.to_string())
    print(f"\nTotal images written: {len(manifest)}")
    print(f"Manifest: {config.MANIFEST_CSV.relative_to(config.BASE_DIR)}")
    return manifest


def plot_distribution(manifest: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    counts = (
        manifest[manifest["split"] == "train"]["emotion"]
        .value_counts()
        .reindex(config.EMOTIONS)
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(config.EMOTIONS, counts.values, color="#4C72B0")
    ax.bar_label(bars, fmt="%d", padding=2, fontsize=9)
    ax.set_title("FER-2013 training split class distribution (28,709 images)")
    ax.set_ylabel("Images")
    ax.set_ylim(0, counts.max() * 1.15)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.DISTRIBUTION_FIG, dpi=150)
    plt.close(fig)
    print(f"Saved {config.DISTRIBUTION_FIG.relative_to(config.BASE_DIR)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare the FER-2013 dataset.")
    parser.add_argument("--force", action="store_true", help="rewrite images and re-download")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args(argv)

    config.ensure_dirs()
    csv_path = download_csv(force=args.force)
    df = load_frames(csv_path)
    verify(df)
    manifest = write_tree(df, force=args.force)
    if not args.skip_plot:
        plot_distribution(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
