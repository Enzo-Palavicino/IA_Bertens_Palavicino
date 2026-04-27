"""Experiment-level helpers for Hito 2 reproducibility and reporting."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from src.report_utils import make_serializable


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device(device_name: str) -> torch.device:
    """Resolve `auto`, `cpu` or `cuda` into a concrete torch device."""
    normalized = device_name.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but no CUDA device is available.")
    return torch.device(normalized)


def summarize_split_sizes(split_bundles: Mapping[str, Any]) -> dict[str, int]:
    """Return the number of samples per split."""
    return {split_name: int(len(bundle.dataframe)) for split_name, bundle in split_bundles.items()}


def summarize_split_sizes_from_datasets(**datasets: Any) -> dict[str, int]:
    """Return the number of samples for named dataset objects."""
    return {split_name: int(len(dataset)) for split_name, dataset in datasets.items()}


def summarize_class_distribution(train_dataframe: pd.DataFrame) -> list[dict[str, int | str]]:
    """Return the class counts observed in the training split only."""
    counts = (
        train_dataframe.groupby(["class_id", "class_name"], sort=True)
        .size()
        .reset_index(name="num_images")
        .sort_values("class_id", kind="stable")
    )
    return counts.to_dict(orient="records")


def build_clip_mlp_summary(
    experiment_name: str,
    class_names: Sequence[str],
    config: Mapping[str, Any],
    split_sizes: Mapping[str, int],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
    notes: Sequence[str] | None = None,
) -> str:
    """Create a compact markdown summary for the CLIP+MLP experiment."""
    lines = [
        f"# {experiment_name}",
        "",
        "## Configuracion",
        f"- **model_id**: `{config['model_id']}`",
        f"- **device**: `{config['device']}`",
        f"- **hidden_sizes**: `{make_serializable(config['hidden_sizes'])}`",
        f"- **dropout**: `{config['dropout']}`",
        f"- **learning_rate**: `{config['learning_rate']}`",
        f"- **batch_size**: `{config['batch_size']}`",
        f"- **epochs**: `{config['epochs']}`",
        f"- **patience**: `{config['patience']}`",
        f"- **class_weights**: `{config['use_class_weights']}`",
        "",
        "## Dataset",
        f"- **num_classes**: `{len(class_names)}`",
        f"- **train_size**: `{split_sizes['train']}`",
        f"- **val_size**: `{split_sizes['val']}`",
        f"- **test_size**: `{split_sizes['test']}`",
        "",
        "## Metricas Val",
    ]

    for key, value in val_metrics.items():
        lines.append(f"- **{key}**: `{make_serializable(value)}`")

    lines.extend(["", "## Metricas Test"])
    for key, value in test_metrics.items():
        lines.append(f"- **{key}**: `{make_serializable(value)}`")

    lines.extend(["", "## Artefactos"])
    for key, value in artifact_paths.items():
        lines.append(f"- **{key}**: `{make_serializable(value)}`")

    if notes:
        lines.extend(["", "## Notas"])
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)


def build_cnn_summary(
    experiment_name: str,
    class_names: Sequence[str],
    config: Mapping[str, Any],
    split_sizes: Mapping[str, int],
    val_metrics: Mapping[str, Any],
    test_metrics: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
    notes: Sequence[str] | None = None,
) -> str:
    """Create a compact markdown summary for the CNN experiment."""
    lines = [
        f"# {experiment_name}",
        "",
        "## Configuracion",
        f"- **image_size**: `{make_serializable(config['image_size'])}`",
        f"- **device**: `{config['device']}`",
        f"- **conv_channels**: `{make_serializable(config['conv_channels'])}`",
        f"- **classifier_hidden_dim**: `{config['classifier_hidden_dim']}`",
        f"- **dropout**: `{config['dropout']}`",
        f"- **learning_rate**: `{config['learning_rate']}`",
        f"- **batch_size**: `{config['batch_size']}`",
        f"- **epochs**: `{config['epochs']}`",
        f"- **patience**: `{config['patience']}`",
        f"- **class_weights**: `{config['use_class_weights']}`",
        "",
        "## Dataset",
        f"- **num_classes**: `{len(class_names)}`",
        f"- **train_size**: `{split_sizes['train']}`",
        f"- **val_size**: `{split_sizes['val']}`",
        f"- **test_size**: `{split_sizes['test']}`",
        "",
        "## Metricas Val",
    ]

    for key, value in val_metrics.items():
        lines.append(f"- **{key}**: `{make_serializable(value)}`")

    lines.extend(["", "## Metricas Test"])
    for key, value in test_metrics.items():
        lines.append(f"- **{key}**: `{make_serializable(value)}`")

    lines.extend(["", "## Artefactos"])
    for key, value in artifact_paths.items():
        lines.append(f"- **{key}**: `{make_serializable(value)}`")

    if notes:
        lines.extend(["", "## Notas"])
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)
