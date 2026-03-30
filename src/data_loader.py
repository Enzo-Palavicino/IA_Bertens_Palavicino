"""Reusable utilities for loading DocExplore query images."""

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.config import DEFAULT_TARGET_CLASSES, QUERIES_DIR, VALID_IMAGE_EXTENSIONS


def list_available_classes(dataset_dir: Path = QUERIES_DIR) -> list[str]:
    """Return class names detected from query subdirectories."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    return sorted(path.name for path in dataset_dir.iterdir() if path.is_dir())


def is_valid_image_file(path: Path) -> bool:
    """Check extension and basic PIL loading to filter corrupt files early."""
    if not path.is_file() or path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
        return False

    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except (UnidentifiedImageError, OSError, ValueError):
        return False


def _iter_valid_image_paths(class_dir: Path) -> Iterable[Path]:
    """Yield valid images from one class directory in a stable order."""
    for image_path in sorted(class_dir.iterdir()):
        if is_valid_image_file(image_path):
            yield image_path


def encode_string_labels(
    labels: Sequence[str],
    class_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """
    Encode string labels into integers.

    If class_names is provided, that explicit order is preserved. Otherwise the
    mapping uses sorted unique labels for deterministic behavior.
    """
    if class_names is None:
        class_name_list = sorted(set(labels))
    else:
        class_name_list = list(class_names)

    label_to_id = {label: idx for idx, label in enumerate(class_name_list)}
    encoded = np.array([label_to_id[label] for label in labels], dtype=np.int64)
    return encoded, class_name_list


def load_binary_class_dataset(
    selected_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    dataset_dir: Path = QUERIES_DIR,
) -> tuple[list[Path], np.ndarray, list[str]]:
    """
    Load image paths and numeric labels for two selected classes.

    This function only returns metadata for now. Future stages can extend it to
    load image tensors, embeddings or cached features without changing callers.
    """
    if len(selected_classes) != 2:
        raise ValueError("Exactly two classes must be provided for the binary dataset.")

    available_classes = set(list_available_classes(dataset_dir))
    missing_classes = [class_name for class_name in selected_classes if class_name not in available_classes]
    if missing_classes:
        raise ValueError(f"Classes not found in dataset: {missing_classes}")

    image_paths: list[Path] = []
    string_labels: list[str] = []

    for class_name in selected_classes:
        class_dir = dataset_dir / class_name
        for image_path in _iter_valid_image_paths(class_dir):
            image_paths.append(image_path)
            string_labels.append(class_name)

    y, class_names = encode_string_labels(string_labels, class_names=selected_classes)
    return image_paths, y, class_names
