"""Reusable evaluation helpers for binary classification experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.config import DEFAULT_EXAMPLES_PER_GROUP, THRESHOLD_VALUES
from src.report_utils import save_dataframe, save_json

EXAMPLE_TYPE_ORDER = ("TP", "FP", "TN", "FN")


def calculate_binary_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> dict[str, float | int]:
    """Calculate standard binary classification metrics plus confusion-matrix counts."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "num_samples": int(len(y_true_arr)),
    }


def generate_classification_report_dict(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict:
    """Generate a sklearn classification report as a serializable dictionary."""
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )


def save_metrics_json(metrics: dict, output_path: str | Path) -> Path:
    """Persist metrics to JSON."""
    return save_json(metrics, output_path)


def save_confusion_matrix_figure(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a confusion-matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title(title)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def build_threshold_table(
    y_true: Sequence[int],
    positive_probabilities: Sequence[float],
    thresholds: Sequence[float] = THRESHOLD_VALUES,
) -> pd.DataFrame:
    """Compute precision/recall/f1/accuracy for a list of thresholds."""
    y_true_arr = np.asarray(y_true)
    probabilities = np.asarray(positive_probabilities)
    rows: list[dict[str, float]] = []

    for threshold in thresholds:
        y_pred_threshold = (probabilities >= threshold).astype(int)
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true_arr, y_pred_threshold)),
                "precision": float(precision_score(y_true_arr, y_pred_threshold, zero_division=0)),
                "recall": float(recall_score(y_true_arr, y_pred_threshold, zero_division=0)),
                "f1": float(f1_score(y_true_arr, y_pred_threshold, zero_division=0)),
            }
        )

    return pd.DataFrame(rows)


def save_threshold_table(threshold_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Persist threshold analysis as CSV."""
    return save_dataframe(threshold_df, output_path, index=False)


def plot_threshold_curves(
    threshold_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> Path:
    """Plot threshold versus precision, recall and F1."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    for metric_name in ("precision", "recall", "f1"):
        sns.lineplot(
            data=threshold_df,
            x="threshold",
            y=metric_name,
            marker="o",
            label=metric_name,
        )

    plt.title(title)
    plt.ylim(0.0, 1.02)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path


def select_prediction_examples(
    image_paths: Sequence[str | Path],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    positive_probabilities: Sequence[float],
    class_names: Sequence[str],
    max_examples_per_type: int = DEFAULT_EXAMPLES_PER_GROUP,
) -> pd.DataFrame:
    """Select representative TP, FP, TN and FN examples with their paths and scores."""
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    probabilities = np.asarray(positive_probabilities)
    path_strings = [str(Path(path).resolve()) for path in image_paths]

    selection_rules = {
        "TP": ((y_true_arr == 1) & (y_pred_arr == 1), np.argsort(-probabilities)),
        "FP": ((y_true_arr == 0) & (y_pred_arr == 1), np.argsort(-probabilities)),
        "TN": ((y_true_arr == 0) & (y_pred_arr == 0), np.argsort(probabilities)),
        "FN": ((y_true_arr == 1) & (y_pred_arr == 0), np.argsort(probabilities)),
    }

    records: list[dict[str, str | int | float]] = []

    for example_type, (mask, ranked_indices) in selection_rules.items():
        selected_indices = [idx for idx in ranked_indices if mask[idx]][:max_examples_per_type]
        for idx in selected_indices:
            records.append(
                {
                    "example_type": example_type,
                    "image_path": path_strings[idx],
                    "true_label": int(y_true_arr[idx]),
                    "pred_label": int(y_pred_arr[idx]),
                    "true_class_name": class_names[int(y_true_arr[idx])],
                    "pred_class_name": class_names[int(y_pred_arr[idx])],
                    "positive_probability": float(probabilities[idx]),
                }
            )

    return pd.DataFrame(records)


def save_examples_table(examples_df: pd.DataFrame, output_path: str | Path) -> Path:
    """Save selected TP/FP/TN/FN examples as CSV."""
    return save_dataframe(examples_df, output_path, index=False)


def save_example_figure(
    examples_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
    max_examples_per_type: int = 4,
) -> Path | None:
    """Generate a compact figure with TP, FP, TN and FN examples."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if examples_df.empty:
        return None

    subsets = {
        example_type: examples_df[examples_df["example_type"] == example_type].head(max_examples_per_type)
        for example_type in EXAMPLE_TYPE_ORDER
    }
    num_rows = max((len(subset) for subset in subsets.values()), default=0)
    if num_rows == 0:
        return None

    fig, axes = plt.subplots(
        num_rows,
        len(EXAMPLE_TYPE_ORDER),
        figsize=(4 * len(EXAMPLE_TYPE_ORDER), 3 * num_rows),
    )

    axes_array = np.array(axes, dtype=object)
    if axes_array.ndim == 1:
        axes_array = axes_array.reshape(1, -1)

    for col_idx, example_type in enumerate(EXAMPLE_TYPE_ORDER):
        subset = subsets[example_type].reset_index(drop=True)
        for row_idx in range(num_rows):
            ax = axes_array[row_idx, col_idx]
            if row_idx >= len(subset):
                ax.axis("off")
                if row_idx == 0:
                    ax.set_title(f"{example_type}\n(sin ejemplos)")
                continue

            row = subset.iloc[row_idx]
            with Image.open(row["image_path"]) as image:
                ax.imshow(image.convert("RGB"))
            ax.set_title(
                f"{example_type}\nreal={row['true_class_name']}\npred={row['pred_class_name']}\np={row['positive_probability']:.2f}"
            )
            ax.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path
