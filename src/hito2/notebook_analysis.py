"""Helpers for the Hito 2 comparison notebook."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


@dataclass
class ExperimentBundle:
    """Container with the main artifacts required for notebook analysis."""

    model_key: str
    display_name: str
    experiment_dir: Path
    config: dict[str, Any] = field(default_factory=dict)
    val_metrics: dict[str, Any] = field(default_factory=dict)
    test_metrics: dict[str, Any] = field(default_factory=dict)
    history_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    val_report: dict[str, Any] = field(default_factory=dict)
    test_report: dict[str, Any] = field(default_factory=dict)
    val_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)
    val_confusion_path: Path | None = None
    test_confusion_path: Path | None = None
    missing_artifacts: list[str] = field(default_factory=list)

    @property
    def class_names(self) -> list[str]:
        """Return the configured class names or infer them from predictions."""
        class_names = self.config.get("class_names")
        if class_names:
            return list(class_names)

        if not self.test_predictions.empty and "true_class_name" in self.test_predictions:
            return sorted(self.test_predictions["true_class_name"].dropna().astype(str).unique().tolist())

        return []

    @property
    def best_epoch(self) -> int | None:
        """Return the best epoch when available."""
        value = self.config.get("best_epoch")
        return int(value) if value is not None else None


def _read_json_if_exists(path: Path) -> tuple[dict[str, Any], bool]:
    """Read a JSON file when present, otherwise return an empty mapping."""
    if not path.exists():
        return {}, False
    return json.loads(path.read_text(encoding="utf-8")), True


def _read_csv_if_exists(path: Path) -> tuple[pd.DataFrame, bool]:
    """Read a CSV file when present, otherwise return an empty dataframe."""
    if not path.exists():
        return pd.DataFrame(), False
    return pd.read_csv(path), True


def load_experiment_bundle(model_key: str, experiment_dir: str | Path, display_name: str | None = None) -> ExperimentBundle:
    """Load the notebook artifact bundle for one experiment directory."""
    experiment_dir = Path(experiment_dir).resolve()
    metrics_dir = experiment_dir / "metrics"
    models_dir = experiment_dir / "models"
    predictions_dir = experiment_dir / "predictions"
    figures_dir = experiment_dir / "figures"

    display_name = display_name or model_key
    missing_artifacts: list[str] = []

    config, exists = _read_json_if_exists(models_dir / "experiment_config.json")
    if not exists:
        missing_artifacts.append("models/experiment_config.json")

    val_metrics, exists = _read_json_if_exists(metrics_dir / "val_metrics.json")
    if not exists:
        missing_artifacts.append("metrics/val_metrics.json")

    test_metrics, exists = _read_json_if_exists(metrics_dir / "test_metrics.json")
    if not exists:
        missing_artifacts.append("metrics/test_metrics.json")

    val_report, exists = _read_json_if_exists(metrics_dir / "val_classification_report.json")
    if not exists:
        missing_artifacts.append("metrics/val_classification_report.json")

    test_report, exists = _read_json_if_exists(metrics_dir / "test_classification_report.json")
    if not exists:
        missing_artifacts.append("metrics/test_classification_report.json")

    history_df, exists = _read_csv_if_exists(metrics_dir / "training_history.csv")
    if not exists:
        missing_artifacts.append("metrics/training_history.csv")

    val_predictions, exists = _read_csv_if_exists(predictions_dir / "val_predictions.csv")
    if not exists:
        missing_artifacts.append("predictions/val_predictions.csv")

    test_predictions, exists = _read_csv_if_exists(predictions_dir / "test_predictions.csv")
    if not exists:
        missing_artifacts.append("predictions/test_predictions.csv")

    val_confusion_path = figures_dir / "val_confusion_matrix.png"
    if not val_confusion_path.exists():
        val_confusion_path = None
        missing_artifacts.append("figures/val_confusion_matrix.png")

    test_confusion_path = figures_dir / "test_confusion_matrix.png"
    if not test_confusion_path.exists():
        test_confusion_path = None
        missing_artifacts.append("figures/test_confusion_matrix.png")

    return ExperimentBundle(
        model_key=model_key,
        display_name=display_name,
        experiment_dir=experiment_dir,
        config=config,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        history_df=history_df,
        val_report=val_report,
        test_report=test_report,
        val_predictions=val_predictions,
        test_predictions=test_predictions,
        val_confusion_path=val_confusion_path,
        test_confusion_path=test_confusion_path,
        missing_artifacts=missing_artifacts,
    )


def load_shared_artifacts(
    manifest_path: str | Path,
    split_path: str | Path,
    split_summary_path: str | Path,
    comparison_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load shared Hito 2 artifacts used across notebook sections."""
    manifest_df, _ = _read_csv_if_exists(Path(manifest_path))
    split_df, _ = _read_csv_if_exists(Path(split_path))
    split_summary, _ = _read_json_if_exists(Path(split_summary_path))

    comparison_df = pd.DataFrame()
    if comparison_path is not None:
        comparison_df, _ = _read_csv_if_exists(Path(comparison_path))

    return {
        "manifest_df": manifest_df,
        "split_df": split_df,
        "split_summary": split_summary,
        "comparison_df": comparison_df,
    }


def build_dataset_overview(manifest_df: pd.DataFrame, split_df: pd.DataFrame) -> dict[str, Any]:
    """Build a compact dataset overview for markdown or tabular rendering."""
    overview = {
        "num_images": int(len(manifest_df)) if not manifest_df.empty else 0,
        "num_classes": int(manifest_df["class_name"].nunique()) if "class_name" in manifest_df else 0,
        "split_counts": {},
        "rare_classes": [],
    }

    if not split_df.empty and "split" in split_df:
        overview["split_counts"] = split_df["split"].value_counts().sort_index().to_dict()

    if not manifest_df.empty and "class_name" in manifest_df:
        class_counts = manifest_df["class_name"].value_counts().sort_values(ascending=True)
        rare_df = class_counts[class_counts <= 5].rename("num_images").reset_index()
        rare_df.columns = ["class_name", "num_images"]
        overview["rare_classes"] = rare_df.to_dict(orient="records")

    return overview


def build_class_distribution_dataframe(manifest_df: pd.DataFrame, split_df: pd.DataFrame) -> pd.DataFrame:
    """Return a per-class distribution table including split counts."""
    if manifest_df.empty:
        return pd.DataFrame(columns=["class_name", "class_id", "total_images", "train", "val", "test"])

    manifest_counts = (
        manifest_df.groupby(["class_name", "class_id"])
        .size()
        .rename("total_images")
        .reset_index()
        .sort_values(["total_images", "class_name"], ascending=[False, True])
    )

    if split_df.empty:
        manifest_counts["train"] = 0
        manifest_counts["val"] = 0
        manifest_counts["test"] = 0
        return manifest_counts

    split_counts = (
        split_df.groupby(["class_name", "split"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for split_name in ("train", "val", "test"):
        if split_name not in split_counts:
            split_counts[split_name] = 0

    merged = manifest_counts.merge(split_counts, on="class_name", how="left").fillna(0)
    for column in ("train", "val", "test"):
        merged[column] = merged[column].astype(int)
    return merged


def build_metrics_comparison_dataframe(experiments: list[ExperimentBundle]) -> pd.DataFrame:
    """Flatten the main experiment metrics into a notebook-friendly table."""
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        row: dict[str, Any] = {
            "model": experiment.display_name,
            "best_epoch": experiment.best_epoch,
            "val_accuracy": experiment.val_metrics.get("accuracy"),
            "test_accuracy": experiment.test_metrics.get("accuracy"),
            "val_macro_precision": experiment.val_metrics.get("macro_precision"),
            "test_macro_precision": experiment.test_metrics.get("macro_precision"),
            "val_macro_recall": experiment.val_metrics.get("macro_recall"),
            "test_macro_recall": experiment.test_metrics.get("macro_recall"),
            "val_macro_f1": experiment.val_metrics.get("macro_f1"),
            "test_macro_f1": experiment.test_metrics.get("macro_f1"),
            "val_weighted_f1": experiment.val_metrics.get("weighted_f1"),
            "test_weighted_f1": experiment.test_metrics.get("weighted_f1"),
            "val_top_3_accuracy": experiment.val_metrics.get("top_3_accuracy"),
            "test_top_3_accuracy": experiment.test_metrics.get("top_3_accuracy"),
            "val_loss": experiment.val_metrics.get("loss"),
            "test_loss": experiment.test_metrics.get("loss"),
        }

        if not experiment.history_df.empty:
            row["epochs_ran"] = int(len(experiment.history_df))
            row["best_val_loss"] = float(experiment.history_df["val_loss"].min())
            row["final_val_loss"] = float(experiment.history_df["val_loss"].iloc[-1])
            row["final_train_loss"] = float(experiment.history_df["train_loss"].iloc[-1])
            row["val_loss_std"] = float(experiment.history_df["val_loss"].astype(float).std(ddof=0))
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.set_index("model")
    return df


def classification_report_to_dataframe(report: dict[str, Any], model_name: str, split_name: str) -> pd.DataFrame:
    """Convert a sklearn report dict into a tidy per-class dataframe."""
    rows: list[dict[str, Any]] = []
    for class_name, metrics in report.items():
        if class_name in {"macro avg", "weighted avg", "micro avg", "samples avg"}:
            continue
        if not isinstance(metrics, dict):
            continue
        if "precision" not in metrics or "recall" not in metrics or "f1-score" not in metrics:
            continue
        rows.append(
            {
                "class_name": class_name,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1-score"]),
                "support": int(metrics["support"]),
                "model": model_name,
                "split": split_name,
            }
        )

    return pd.DataFrame(rows).sort_values(["support", "f1_score", "class_name"], ascending=[False, False, True])


def build_per_class_comparison_dataframe(
    experiments: list[ExperimentBundle],
    split_name: str = "test",
) -> pd.DataFrame:
    """Merge the per-class reports from several experiments into a comparison table."""
    if split_name not in {"val", "test"}:
        raise ValueError(f"Unsupported split name: {split_name}")

    merged: pd.DataFrame | None = None
    for experiment in experiments:
        report = experiment.val_report if split_name == "val" else experiment.test_report
        report_df = classification_report_to_dataframe(report, experiment.display_name, split_name)
        if report_df.empty:
            continue

        model_df = report_df.rename(
            columns={
                "precision": f"{experiment.model_key}_precision",
                "recall": f"{experiment.model_key}_recall",
                "f1_score": f"{experiment.model_key}_f1",
                "support": f"{experiment.model_key}_support",
            }
        )[["class_name", f"{experiment.model_key}_precision", f"{experiment.model_key}_recall", f"{experiment.model_key}_f1", f"{experiment.model_key}_support"]]

        if merged is None:
            merged = model_df
        else:
            merged = merged.merge(model_df, on="class_name", how="outer")

    if merged is None:
        return pd.DataFrame()

    support_columns = [column for column in merged.columns if column.endswith("_support")]
    if support_columns:
        merged["support"] = merged[support_columns].bfill(axis=1).iloc[:, 0].fillna(0).astype(int)

    metric_prefixes = [experiment.model_key for experiment in experiments]
    if len(metric_prefixes) >= 2:
        first_key = metric_prefixes[0]
        second_key = metric_prefixes[1]
        if f"{first_key}_f1" in merged and f"{second_key}_f1" in merged:
            merged["f1_gap"] = merged[f"{first_key}_f1"] - merged[f"{second_key}_f1"]

    sort_columns = ["support"]
    ascending = [False]
    if "f1_gap" in merged:
        sort_columns.append("f1_gap")
        ascending.append(False)
    return merged.sort_values(sort_columns, ascending=ascending)


def build_confusion_matrix_from_predictions(predictions_df: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Compute a confusion matrix directly from a predictions table."""
    if predictions_df.empty:
        return np.zeros((len(class_names), len(class_names)), dtype=int)

    labels = list(range(len(class_names)))
    return confusion_matrix(
        predictions_df["true_label"].astype(int),
        predictions_df["pred_label"].astype(int),
        labels=labels,
    )


def build_top_confusions_dataframe(predictions_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the most frequent confusion pairs from a predictions table."""
    if predictions_df.empty:
        return pd.DataFrame(columns=["true_class_name", "pred_class_name", "count"])

    errors_df = predictions_df[predictions_df["true_class_name"] != predictions_df["pred_class_name"]].copy()
    if errors_df.empty:
        return pd.DataFrame(columns=["true_class_name", "pred_class_name", "count"])

    confusion_df = (
        errors_df.groupby(["true_class_name", "pred_class_name"])
        .size()
        .rename("count")
        .reset_index()
        .sort_values(["count", "true_class_name", "pred_class_name"], ascending=[False, True, True])
        .head(top_n)
    )
    return confusion_df


def build_top_k_gain_dataframe(experiments: list[ExperimentBundle]) -> pd.DataFrame:
    """Compare top-1 and top-3 accuracy for val and test."""
    rows: list[dict[str, Any]] = []
    for experiment in experiments:
        val_top1 = experiment.val_metrics.get("accuracy")
        val_top3 = experiment.val_metrics.get("top_3_accuracy")
        test_top1 = experiment.test_metrics.get("accuracy")
        test_top3 = experiment.test_metrics.get("top_3_accuracy")
        rows.append(
            {
                "model": experiment.display_name,
                "val_top_1": val_top1,
                "val_top_3": val_top3,
                "val_gain": None if val_top1 is None or val_top3 is None else float(val_top3) - float(val_top1),
                "test_top_1": test_top1,
                "test_top_3": test_top3,
                "test_gain": None if test_top1 is None or test_top3 is None else float(test_top3) - float(test_top1),
            }
        )
    return pd.DataFrame(rows).set_index("model")


def select_prediction_examples(
    predictions_df: pd.DataFrame,
    correct: bool,
    n: int = 4,
) -> pd.DataFrame:
    """Select a compact and reasonably diverse subset of prediction examples."""
    if predictions_df.empty:
        return predictions_df

    working_df = predictions_df.copy()
    working_df["is_correct"] = working_df["true_label"] == working_df["pred_label"]
    filtered_df = working_df[working_df["is_correct"] == bool(correct)].copy()
    if filtered_df.empty:
        return filtered_df

    filtered_df["display_score"] = filtered_df["pred_probability"].astype(float)
    filtered_df = filtered_df.sort_values("display_score", ascending=False)

    selected_rows: list[pd.Series] = []
    seen_true_classes: set[str] = set()
    for _, row in filtered_df.iterrows():
        true_class_name = str(row["true_class_name"])
        if true_class_name in seen_true_classes:
            continue
        selected_rows.append(row)
        seen_true_classes.add(true_class_name)
        if len(selected_rows) >= n:
            break

    if len(selected_rows) < n:
        remaining = filtered_df.loc[~filtered_df.index.isin([row.name for row in selected_rows])]
        for _, row in remaining.head(n - len(selected_rows)).iterrows():
            selected_rows.append(row)

    return pd.DataFrame(selected_rows)
