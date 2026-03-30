"""Helpers for saving tables, JSON files and markdown summaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def make_serializable(value: Any) -> Any:
    """Recursively convert common Python, numpy and pathlib objects to JSON-safe values."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): make_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save_json(data: Any, path: str | Path) -> Path:
    """Save structured data as pretty-printed JSON."""
    output_path = Path(path)
    ensure_parent_dir(output_path)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(make_serializable(data), file, indent=2, ensure_ascii=False)
    return output_path


def save_dataframe(dataframe: pd.DataFrame, path: str | Path, index: bool = False) -> Path:
    """Save a dataframe as CSV or Markdown according to the file suffix."""
    output_path = Path(path)
    ensure_parent_dir(output_path)

    if output_path.suffix == ".csv":
        dataframe.to_csv(output_path, index=index)
    elif output_path.suffix == ".md":
        output_path.write_text(dataframe.to_markdown(index=index), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported table format for path: {output_path}")

    return output_path


def save_text(text: str, path: str | Path) -> Path:
    """Write plain text or markdown content to disk."""
    output_path = Path(path)
    ensure_parent_dir(output_path)
    output_path.write_text(text, encoding="utf-8")
    return output_path


def format_metrics_table(metrics: Mapping[str, Any], decimals: int = 4) -> str:
    """Format a metrics dictionary as a compact markdown list."""
    lines: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, float):
            rendered = f"{value:.{decimals}f}"
        else:
            rendered = str(value)
        lines.append(f"- **{key}**: {rendered}")
    return "\n".join(lines)


def build_experiment_summary(
    experiment_name: str,
    class_names: list[str],
    metrics: Mapping[str, Any],
    artifact_paths: Mapping[str, Any] | None = None,
    notes: list[str] | None = None,
) -> str:
    """Create a markdown summary suitable for reports and notebooks."""
    lines = [
        f"# {experiment_name}",
        "",
        "## Clases",
        f"- Clase 0: `{class_names[0]}`",
        f"- Clase 1: `{class_names[1]}`",
        "",
        "## Metricas",
        format_metrics_table(metrics),
    ]

    if artifact_paths:
        lines.extend(["", "## Artefactos"])
        for key, value in artifact_paths.items():
            lines.append(f"- **{key}**: `{make_serializable(value)}`")

    if notes:
        lines.extend(["", "## Notas"])
        for note in notes:
            lines.append(f"- {note}")

    return "\n".join(lines)
