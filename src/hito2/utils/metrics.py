"""Multiclass metrics, reports and prediction tables for Hito 2."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

from src.config import OUTPUTS_DIR

HITO2_CACHE_DIR = OUTPUTS_DIR / ".hito2_cache"
HITO2_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str((HITO2_CACHE_DIR / "matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(HITO2_CACHE_DIR.resolve()))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_top_k_accuracy(y_true: Sequence[int], probabilities: np.ndarray, k: int = 3) -> float:
    """Compute top-k accuracy from a probability matrix."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    probabilities_arr = np.asarray(probabilities, dtype=np.float64)

    if probabilities_arr.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got shape {probabilities_arr.shape}.")

    k = max(1, min(int(k), probabilities_arr.shape[1]))
    top_k_indices = np.argsort(-probabilities_arr, axis=1)[:, :k]
    hits = [int(label) in top_k_indices[row_idx] for row_idx, label in enumerate(y_true_arr)]
    return float(np.mean(hits))


def calculate_multiclass_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: np.ndarray,
    top_k: int = 3,
) -> dict[str, float | int]:
    """Calculate the main multiclass metrics requested for Hito 2."""
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)

    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_precision": float(precision_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true_arr, y_pred_arr, average="weighted", zero_division=0)),
        f"top_{min(max(1, top_k), probabilities.shape[1])}_accuracy": float(
            compute_top_k_accuracy(y_true_arr, probabilities=probabilities, k=top_k)
        ),
        "num_samples": int(len(y_true_arr)),
    }


def build_classification_report_dict(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> dict[str, object]:
    """Return a sklearn classification report with a fixed full class list."""
    labels = list(range(len(class_names)))
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=list(class_names),
        output_dict=True,
        zero_division=0,
    )


def build_predictions_dataframe(
    metadata_df: pd.DataFrame,
    y_true: Sequence[int],
    probabilities: np.ndarray,
    class_names: Sequence[str],
    top_k: int = 3,
) -> pd.DataFrame:
    """Build a prediction table with top-k class names and probabilities."""
    probabilities_arr = np.asarray(probabilities, dtype=np.float64)
    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = probabilities_arr.argmax(axis=1)
    top_k = max(1, min(int(top_k), probabilities_arr.shape[1]))
    top_k_indices = np.argsort(-probabilities_arr, axis=1)[:, :top_k]

    predictions_df = metadata_df.copy().reset_index(drop=True)
    predictions_df["true_label"] = y_true_arr
    predictions_df["true_class_name"] = [class_names[int(label)] for label in y_true_arr]
    predictions_df["pred_label"] = y_pred_arr
    predictions_df["pred_class_name"] = [class_names[int(label)] for label in y_pred_arr]
    predictions_df["pred_probability"] = probabilities_arr[np.arange(len(predictions_df)), y_pred_arr]

    for rank in range(top_k):
        indices = top_k_indices[:, rank]
        predictions_df[f"top_{rank + 1}_class_id"] = indices
        predictions_df[f"top_{rank + 1}_class_name"] = [class_names[int(label)] for label in indices]
        predictions_df[f"top_{rank + 1}_probability"] = probabilities_arr[np.arange(len(predictions_df)), indices]

    return predictions_df


def build_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> np.ndarray:
    """Build a fixed-size confusion matrix over the full class list."""
    labels = list(range(len(class_names)))
    return confusion_matrix(y_true, y_pred, labels=labels)


def save_confusion_matrix_figure(
    confusion: np.ndarray,
    class_names: Sequence[str],
    output_path: str | Path,
    title: str,
) -> Path:
    """Save a multiclass confusion matrix heatmap."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_classes = len(class_names)
    figure_size = max(10, int(num_classes * 0.5))
    annotate = num_classes <= 15

    plt.figure(figsize=(figure_size, figure_size))
    sns.heatmap(
        confusion,
        annot=annotate,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel("Prediccion")
    plt.ylabel("Real")
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    return output_path
