"""Basic image preprocessing helpers for future experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split


def load_image_rgb(path: str | Path) -> Image.Image:
    """Open an image from disk and convert it to RGB."""
    with Image.open(path) as image:
        return image.convert("RGB")


def resize_image(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize an image to a fixed (width, height) size."""
    return image.resize(size)


def flatten_image(image: Image.Image) -> np.ndarray:
    """Convert an image into a 1D numpy array for pixel-based baselines."""
    return np.asarray(image, dtype=np.float32).reshape(-1)


def image_path_to_flattened_array(path: str | Path, size: tuple[int, int]) -> np.ndarray:
    """Load, resize and flatten a single image path into a feature vector."""
    image = load_image_rgb(path)
    image = resize_image(image, size=size)
    return flatten_image(image)


def build_flattened_feature_matrix(
    image_paths: list[str | Path],
    size: tuple[int, int],
) -> np.ndarray:
    """Convert a list of image paths into a 2D feature matrix."""
    features = [image_path_to_flattened_array(path, size=size) for path in image_paths]
    return np.vstack(features).astype(np.float32)


def split_data(
    X: Any,
    y: Any,
    test_size: float,
    random_state: int,
    stratify: Any = True,
):
    """
    Thin wrapper around train_test_split with optional stratification.

    If stratify=True, y is used automatically. If stratify=False, no
    stratification is applied. A custom array-like value can also be passed.
    """
    if stratify is True:
        stratify_input = y
    elif stratify is False:
        stratify_input = None
    else:
        stratify_input = stratify

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_input,
    )
