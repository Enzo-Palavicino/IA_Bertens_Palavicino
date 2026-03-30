"""Pixel baseline: resized RGB images + flatten + Logistic Regression."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import (
    BASELINE_EXPERIMENT_NAME,
    DEFAULT_TARGET_CLASSES,
    FIGURES_DIR,
    LOGREG_MAX_ITER,
    METRICS_DIR,
    MODELS_DIR,
    PIXEL_BASELINE_IMAGE_SIZE,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_project_directories,
)
from src.data_loader import load_binary_class_dataset
from src.evaluate import (
    build_threshold_table,
    calculate_binary_metrics,
    generate_classification_report_dict,
    plot_threshold_curves,
    save_confusion_matrix_figure,
    save_example_figure,
    save_examples_table,
    save_metrics_json,
    select_prediction_examples,
)
from src.preprocessing import image_path_to_flattened_array, split_data
from src.report_utils import build_experiment_summary, save_dataframe, save_json, save_text


def build_pixel_feature_matrix(
    image_paths: Sequence[str | Path],
    image_size: tuple[int, int],
) -> pd.DataFrame:
    """Convert image paths into a 2D feature matrix based on resized RGB pixels."""
    rows = [
        image_path_to_flattened_array(path, size=image_size)
        for path in tqdm(image_paths, desc="Construyendo baseline de pixeles", leave=False)
    ]
    return pd.DataFrame(rows, dtype="float32")


def _prediction_dataframe(
    image_paths: Sequence[str | Path],
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[float],
    class_names: Sequence[str],
) -> pd.DataFrame:
    """Build a prediction table for later inspection and reporting."""
    return pd.DataFrame(
        {
            "image_path": [str(Path(path).resolve()) for path in image_paths],
            "true_label": y_true,
            "pred_label": y_pred,
            "true_class_name": [class_names[int(label)] for label in y_true],
            "pred_class_name": [class_names[int(label)] for label in y_pred],
            "positive_probability": probabilities,
        }
    )


def run_baseline_experiment(
    selected_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    image_size: tuple[int, int] = PIXEL_BASELINE_IMAGE_SIZE,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    max_iter: int = LOGREG_MAX_ITER,
) -> dict:
    """Run the pixel baseline end-to-end and persist all requested artifacts."""
    ensure_project_directories()

    image_paths, y, class_names = load_binary_class_dataset(selected_classes=selected_classes)
    train_paths, test_paths, y_train, y_test = split_data(
        X=image_paths,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    X_train = build_pixel_feature_matrix(train_paths, image_size=image_size)
    X_test = build_pixel_feature_matrix(test_paths, image_size=image_size)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    classifier.fit(X_train_scaled, y_train)

    y_pred = classifier.predict(X_test_scaled)
    positive_probabilities = classifier.predict_proba(X_test_scaled)[:, 1]

    metrics = calculate_binary_metrics(y_test, y_pred)
    metrics.update(
        {
            "experiment_name": BASELINE_EXPERIMENT_NAME,
            "positive_class_name": class_names[1],
            "negative_class_name": class_names[0],
            "train_size": int(len(train_paths)),
            "test_size": int(len(test_paths)),
            "image_width": int(image_size[0]),
            "image_height": int(image_size[1]),
        }
    )
    classification_report_dict = generate_classification_report_dict(y_test, y_pred, class_names)
    threshold_df = build_threshold_table(y_test, positive_probabilities)
    predictions_df = _prediction_dataframe(test_paths, y_test, y_pred, positive_probabilities, class_names)
    examples_df = select_prediction_examples(
        image_paths=test_paths,
        y_true=y_test,
        y_pred=y_pred,
        positive_probabilities=positive_probabilities,
        class_names=class_names,
    )

    metrics_path = METRICS_DIR / f"{BASELINE_EXPERIMENT_NAME}_metrics.json"
    report_path = METRICS_DIR / f"{BASELINE_EXPERIMENT_NAME}_classification_report.json"
    threshold_path = METRICS_DIR / f"{BASELINE_EXPERIMENT_NAME}_thresholds.csv"
    predictions_path = METRICS_DIR / f"{BASELINE_EXPERIMENT_NAME}_predictions.csv"
    examples_path = METRICS_DIR / f"{BASELINE_EXPERIMENT_NAME}_examples.csv"
    summary_path = METRICS_DIR / f"{BASELINE_EXPERIMENT_NAME}_summary.md"

    confusion_path = FIGURES_DIR / f"{BASELINE_EXPERIMENT_NAME}_confusion_matrix.png"
    threshold_figure_path = FIGURES_DIR / f"{BASELINE_EXPERIMENT_NAME}_threshold_curves.png"
    examples_figure_path = FIGURES_DIR / f"{BASELINE_EXPERIMENT_NAME}_examples.png"

    model_path = MODELS_DIR / f"{BASELINE_EXPERIMENT_NAME}_model.joblib"

    save_metrics_json(metrics, metrics_path)
    save_json(classification_report_dict, report_path)
    save_dataframe(threshold_df, threshold_path, index=False)
    save_dataframe(predictions_df, predictions_path, index=False)
    save_examples_table(examples_df, examples_path)

    save_confusion_matrix_figure(
        y_true=y_test,
        y_pred=y_pred,
        class_names=class_names,
        output_path=confusion_path,
        title=f"Matriz de confusion - {BASELINE_EXPERIMENT_NAME}",
    )
    plot_threshold_curves(
        threshold_df=threshold_df,
        output_path=threshold_figure_path,
        title=f"Threshold analysis - {BASELINE_EXPERIMENT_NAME}",
    )
    example_figure_output = save_example_figure(
        examples_df=examples_df,
        output_path=examples_figure_path,
        title=f"Ejemplos TP / FP / TN / FN - {BASELINE_EXPERIMENT_NAME}",
    )

    joblib.dump(
        {
            "scaler": scaler,
            "classifier": classifier,
            "class_names": list(class_names),
            "image_size": image_size,
            "selected_classes": list(selected_classes),
            "random_state": random_state,
        },
        model_path,
    )

    artifact_paths = {
        "metrics_json": metrics_path,
        "classification_report_json": report_path,
        "threshold_table_csv": threshold_path,
        "predictions_csv": predictions_path,
        "examples_csv": examples_path,
        "confusion_matrix_png": confusion_path,
        "threshold_curves_png": threshold_figure_path,
        "examples_png": example_figure_output,
        "model_joblib": model_path,
    }
    summary_text = build_experiment_summary(
        experiment_name=BASELINE_EXPERIMENT_NAME,
        class_names=list(class_names),
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes=[
            "Baseline de pixeles usando RGB redimensionado, flatten, StandardScaler y LogisticRegression.",
        ],
    )
    save_text(summary_text, summary_path)

    return {
        "experiment_name": BASELINE_EXPERIMENT_NAME,
        "class_names": list(class_names),
        "metrics": metrics,
        "classification_report": classification_report_dict,
        "artifact_paths": {key: str(value) if value is not None else None for key, value in artifact_paths.items()},
        "summary_path": str(summary_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for running only the baseline experiment."""
    parser = argparse.ArgumentParser(description="Run the pixel baseline for Hito 1.")
    parser.add_argument("--class-a", default=DEFAULT_TARGET_CLASSES[0], help="Negative class name.")
    parser.add_argument("--class-b", default=DEFAULT_TARGET_CLASSES[1], help="Positive class name.")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=PIXEL_BASELINE_IMAGE_SIZE,
        help="Resize used before flattening.",
    )
    return parser


def main() -> None:
    """Entry point for direct CLI execution."""
    args = build_parser().parse_args()
    result = run_baseline_experiment(
        selected_classes=(args.class_a, args.class_b),
        image_size=tuple(args.image_size),
    )
    print(f"{result['experiment_name']} listo.")
    print(result["metrics"])


if __name__ == "__main__":
    main()
