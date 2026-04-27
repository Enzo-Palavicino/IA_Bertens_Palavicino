"""Manifest generation utilities for the Hito 2 multiclass dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from src.config import QUERIES_DIR, ROOT_DIR
from src.data_loader import is_valid_image_file
from src.hito2.config import DEFAULT_MANIFEST_PATH, as_repo_relative
from src.report_utils import save_dataframe

MANIFEST_COLUMNS = ("image_path", "class_name", "class_id")


def _iter_query_class_directories(dataset_dir: Path = QUERIES_DIR) -> Iterable[Path]:
    """Yield query class directories in deterministic order."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Query dataset directory not found: {dataset_dir}")

    yield from sorted(path for path in dataset_dir.iterdir() if path.is_dir())


def _list_valid_images(class_dir: Path) -> list[Path]:
    """Return valid image files from one class directory in deterministic order."""
    return [
        image_path
        for image_path in sorted(class_dir.iterdir())
        if image_path.is_file() and is_valid_image_file(image_path)
    ]


def discover_query_class_names(dataset_dir: Path = QUERIES_DIR) -> list[str]:
    """Return query class names that contain at least one valid image."""
    class_names: list[str] = []

    for class_dir in _iter_query_class_directories(dataset_dir=dataset_dir):
        if _list_valid_images(class_dir):
            class_names.append(class_dir.name)

    return class_names


def build_class_to_id_mapping(class_names: Sequence[str]) -> dict[str, int]:
    """
    Build a stable class-to-id mapping.

    The mapping is always derived from the lexicographically sorted class names,
    which keeps ids reproducible across runs as long as the available class set
    remains unchanged.
    """
    return {class_name: class_id for class_id, class_name in enumerate(sorted(set(class_names)))}


def build_query_manifest_dataframe(
    dataset_dir: Path = QUERIES_DIR,
    root_dir: Path = ROOT_DIR,
) -> pd.DataFrame:
    """
    Build a manifest dataframe with one row per valid query image.

    Each immediate subdirectory under `dataset_dir` is treated as one class. Any
    directory without valid images is skipped explicitly.
    """
    class_to_images: dict[str, list[Path]] = {}

    for class_dir in _iter_query_class_directories(dataset_dir=dataset_dir):
        valid_images = _list_valid_images(class_dir)
        if valid_images:
            class_to_images[class_dir.name] = valid_images

    class_to_id = build_class_to_id_mapping(class_to_images.keys())
    records: list[dict[str, str | int]] = []

    for class_name in sorted(class_to_images):
        class_id = class_to_id[class_name]
        for image_path in class_to_images[class_name]:
            records.append(
                {
                    "image_path": as_repo_relative(image_path, root_dir=root_dir),
                    "class_name": class_name,
                    "class_id": class_id,
                }
            )

    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def load_manifest_dataframe(manifest_path: str | Path) -> pd.DataFrame:
    """Load a manifest CSV and validate the required columns."""
    manifest_df = pd.read_csv(manifest_path)
    validate_manifest_dataframe(manifest_df)
    return manifest_df


def validate_manifest_dataframe(manifest_df: pd.DataFrame) -> None:
    """Validate that a dataframe contains the Hito 2 manifest schema."""
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in manifest_df.columns]
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {missing_columns}")


def save_query_manifest(
    manifest_df: pd.DataFrame,
    output_path: str | Path = DEFAULT_MANIFEST_PATH,
) -> Path:
    """Persist the manifest CSV to disk."""
    validate_manifest_dataframe(manifest_df)
    return save_dataframe(manifest_df, output_path, index=False)


def summarize_manifest_dataframe(manifest_df: pd.DataFrame) -> dict[str, object]:
    """Return a compact summary of the generated manifest."""
    validate_manifest_dataframe(manifest_df)

    class_counts = (
        manifest_df.groupby(["class_id", "class_name"], sort=True)
        .size()
        .reset_index(name="num_images")
        .sort_values(by=["class_id"], kind="stable")
    )

    small_classes = class_counts[class_counts["num_images"] < 3]["class_name"].tolist()

    return {
        "num_images": int(len(manifest_df)),
        "num_classes": int(class_counts["class_id"].nunique()),
        "min_images_per_class": int(class_counts["num_images"].min()),
        "max_images_per_class": int(class_counts["num_images"].max()),
        "classes_with_fewer_than_three_images": small_classes,
        "class_counts": class_counts.to_dict(orient="records"),
    }
