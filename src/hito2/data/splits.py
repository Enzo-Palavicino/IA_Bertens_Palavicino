"""Deterministic train/val/test split utilities for Hito 2."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.hito2.config import (
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SUMMARY_PATH,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
)
from src.hito2.data.manifest import load_manifest_dataframe, validate_manifest_dataframe
from src.report_utils import save_dataframe, save_json

SPLIT_NAMES = ("train", "val", "test")
SPLIT_COLUMN = "split"


def validate_split_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> np.ndarray:
    """Validate and return split ratios as a numpy array."""
    ratios = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)

    if np.any(ratios <= 0):
        raise ValueError("All split ratios must be strictly positive.")
    if not np.isclose(ratios.sum(), 1.0):
        raise ValueError(
            "Split ratios must sum to 1.0. "
            f"Received train={train_ratio}, val={val_ratio}, test={test_ratio}."
        )

    return ratios


def _minimum_split_counts(num_samples: int) -> np.ndarray:
    """
    Return the minimum feasible split allocation for one class.

    Classes with fewer than three samples cannot appear in all three splits, so
    the fallback is conservative: singletons stay in train and pairs are split
    between train and test.
    """
    if num_samples <= 0:
        raise ValueError("Each class must contain at least one sample.")
    if num_samples == 1:
        return np.array([1, 0, 0], dtype=np.int64)
    if num_samples == 2:
        return np.array([1, 0, 1], dtype=np.int64)
    return np.array([1, 1, 1], dtype=np.int64)


def _allocate_split_counts(num_samples: int, ratios: np.ndarray) -> tuple[int, int, int]:
    """Allocate deterministic per-class counts while honoring feasibility limits."""
    desired = ratios * num_samples
    counts = np.floor(desired).astype(np.int64)
    minimums = _minimum_split_counts(num_samples)
    counts = np.maximum(counts, minimums)

    while counts.sum() < num_samples:
        deficits = desired - counts
        split_idx = int(np.argmax(deficits))
        counts[split_idx] += 1

    while counts.sum() > num_samples:
        removable = np.where(counts > minimums)[0]
        if len(removable) == 0:
            raise ValueError(f"Could not allocate splits for class with {num_samples} samples.")
        surpluses = counts[removable] - desired[removable]
        split_idx = int(removable[np.argmax(surpluses)])
        counts[split_idx] -= 1

    return tuple(int(value) for value in counts)


def _stable_class_seed(class_name: str, random_state: int) -> int:
    """Derive a deterministic per-class seed from the class name and global seed."""
    seed_material = f"{random_state}:{class_name}".encode("utf-8")
    return int(hashlib.md5(seed_material).hexdigest()[:8], 16)


def _shuffled_group_indices(group_df: pd.DataFrame, random_state: int) -> np.ndarray:
    """Shuffle one class group without coupling its order to other classes."""
    class_name = str(group_df["class_name"].iloc[0])
    class_rng = np.random.default_rng(_stable_class_seed(class_name, random_state))
    indices = group_df.index.to_numpy(copy=True)
    return indices[class_rng.permutation(len(indices))]


def create_stratified_splits_dataframe(
    manifest_df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    Create deterministic train/val/test splits with per-class balancing.

    For classes with three or more samples, every split receives at least one
    sample. Rare classes with one or two images use a documented fallback,
    because perfect three-way stratification is not feasible.
    """
    validate_manifest_dataframe(manifest_df)
    ratios = validate_split_ratios(train_ratio, val_ratio, test_ratio)

    split_assignments: dict[int, str] = {}

    grouped = manifest_df.groupby(["class_id", "class_name"], sort=True, dropna=False)
    for (_, _), group_df in grouped:
        shuffled_indices = _shuffled_group_indices(group_df, random_state=random_state)
        train_count, val_count, test_count = _allocate_split_counts(len(group_df), ratios=ratios)

        boundaries = {
            "train": train_count,
            "val": train_count + val_count,
            "test": train_count + val_count + test_count,
        }

        for index in shuffled_indices[: boundaries["train"]]:
            split_assignments[int(index)] = "train"
        for index in shuffled_indices[boundaries["train"] : boundaries["val"]]:
            split_assignments[int(index)] = "val"
        for index in shuffled_indices[boundaries["val"] : boundaries["test"]]:
            split_assignments[int(index)] = "test"

    split_df = manifest_df.copy()
    split_df[SPLIT_COLUMN] = [split_assignments[int(index)] for index in split_df.index]
    return split_df


def load_split_dataframe(split_path: str | Path) -> pd.DataFrame:
    """Load a split CSV and validate the expected schema."""
    split_df = load_manifest_dataframe(split_path)
    if SPLIT_COLUMN not in split_df.columns:
        raise ValueError(f"Split dataframe is missing the `{SPLIT_COLUMN}` column.")
    return split_df


def save_split_dataframe(
    split_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_SPLIT_PATH,
) -> Path:
    """Persist a split dataframe as CSV."""
    if SPLIT_COLUMN not in split_df.columns:
        raise ValueError(f"Split dataframe is missing the `{SPLIT_COLUMN}` column.")
    return save_dataframe(split_df, output_path, index=False)


def summarize_split_dataframe(
    split_df: pd.DataFrame,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    val_ratio: float = DEFAULT_VAL_RATIO,
    test_ratio: float = DEFAULT_TEST_RATIO,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Return a compact summary of the split distribution and fallback classes."""
    if SPLIT_COLUMN not in split_df.columns:
        raise ValueError(f"Split dataframe is missing the `{SPLIT_COLUMN}` column.")

    class_split_counts = (
        split_df.groupby(["class_name", SPLIT_COLUMN], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=SPLIT_NAMES, fill_value=0)
    )

    class_counts = split_df.groupby("class_name", sort=True).size()
    fallback_classes = class_counts[class_counts < 3].index.tolist()

    return {
        "random_state": int(random_state),
        "target_ratios": {
            "train": float(train_ratio),
            "val": float(val_ratio),
            "test": float(test_ratio),
        },
        "num_images": int(len(split_df)),
        "num_classes": int(split_df["class_id"].nunique()),
        "split_counts": {
            split_name: int((split_df[SPLIT_COLUMN] == split_name).sum())
            for split_name in SPLIT_NAMES
        },
        "fallback_classes": fallback_classes,
        "classes_missing_from_val": class_split_counts.index[class_split_counts["val"] == 0].tolist(),
        "classes_missing_from_test": class_split_counts.index[class_split_counts["test"] == 0].tolist(),
        "class_split_counts": class_split_counts.reset_index().to_dict(orient="records"),
    }


def save_split_summary(
    split_summary: dict[str, object],
    output_path: str | Path = DEFAULT_SPLIT_SUMMARY_PATH,
) -> Path:
    """Persist the split summary as JSON."""
    return save_json(split_summary, output_path)
