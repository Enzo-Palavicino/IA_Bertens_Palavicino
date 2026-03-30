"""CLIP embeddings + Logistic Regression pipeline for Hito 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.clip_embeddings import extract_clip_embeddings, get_embedding_cache_paths, load_clip_components
from src.config import (
    CLIP_BATCH_SIZE,
    CLIP_EXPERIMENT_NAME,
    CLIP_MODEL_ID,
    DEFAULT_TARGET_CLASSES,
    FIGURES_DIR,
    LOGREG_MAX_ITER,
    METRICS_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    ensure_project_directories,
)
from src.data_loader import decode_numeric_labels, load_binary_class_dataset
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
from src.preprocessing import split_data
from src.report_utils import build_experiment_summary, save_dataframe, save_json, save_text


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


def run_clip_experiment(
    selected_classes: Sequence[str] = DEFAULT_TARGET_CLASSES,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    max_iter: int = LOGREG_MAX_ITER,
    model_id: str = CLIP_MODEL_ID,
    batch_size: int = CLIP_BATCH_SIZE,
    reuse_embeddings: bool = True,
) -> dict:
    """Run CLIP embeddings + Logistic Regression end-to-end."""
    ensure_project_directories()

    image_paths, y, class_names = load_binary_class_dataset(selected_classes=selected_classes)
    train_paths, test_paths, y_train, y_test = split_data(
        X=image_paths,
        y=y,
        test_size=test_size,
        random_state=random_state,
        stratify=True,
    )

    train_labels = decode_numeric_labels(y_train, class_names)
    test_labels = decode_numeric_labels(y_test, class_names)

    train_cache = get_embedding_cache_paths(train_paths, CLIP_EXPERIMENT_NAME, "train", model_id=model_id)
    test_cache = get_embedding_cache_paths(test_paths, CLIP_EXPERIMENT_NAME, "test", model_id=model_id)

    cache_ready = reuse_embeddings and all(
        path_set["embeddings"].exists() and path_set["metadata"].exists() and path_set["info"].exists()
        for path_set in (train_cache, test_cache)
    )

    processor = model = device = None
    if not cache_ready:
        processor, model, device = load_clip_components(model_id=model_id)

    X_train_embeddings, _, train_cache_info = extract_clip_embeddings(
        image_paths=train_paths,
        labels=train_labels,
        experiment_name=CLIP_EXPERIMENT_NAME,
        split_name="train",
        batch_size=batch_size,
        model_id=model_id,
        reuse_cache=reuse_embeddings,
        processor=processor,
        model=model,
        device=device,
    )
    X_test_embeddings, _, test_cache_info = extract_clip_embeddings(
        image_paths=test_paths,
        labels=test_labels,
        experiment_name=CLIP_EXPERIMENT_NAME,
        split_name="test",
        batch_size=batch_size,
        model_id=model_id,
        reuse_cache=reuse_embeddings,
        processor=processor,
        model=model,
        device=device,
    )

    classifier = LogisticRegression(max_iter=max_iter, random_state=random_state)
    classifier.fit(X_train_embeddings, y_train)

    y_pred = classifier.predict(X_test_embeddings)
    positive_probabilities = classifier.predict_proba(X_test_embeddings)[:, 1]

    metrics = calculate_binary_metrics(y_test, y_pred)
    metrics.update(
        {
            "experiment_name": CLIP_EXPERIMENT_NAME,
            "positive_class_name": class_names[1],
            "negative_class_name": class_names[0],
            "train_size": int(len(train_paths)),
            "test_size": int(len(test_paths)),
            "clip_model_id": model_id,
            "embedding_dim": int(X_train_embeddings.shape[1]),
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

    metrics_path = METRICS_DIR / f"{CLIP_EXPERIMENT_NAME}_metrics.json"
    report_path = METRICS_DIR / f"{CLIP_EXPERIMENT_NAME}_classification_report.json"
    threshold_path = METRICS_DIR / f"{CLIP_EXPERIMENT_NAME}_thresholds.csv"
    predictions_path = METRICS_DIR / f"{CLIP_EXPERIMENT_NAME}_predictions.csv"
    examples_path = METRICS_DIR / f"{CLIP_EXPERIMENT_NAME}_examples.csv"
    summary_path = METRICS_DIR / f"{CLIP_EXPERIMENT_NAME}_summary.md"

    confusion_path = FIGURES_DIR / f"{CLIP_EXPERIMENT_NAME}_confusion_matrix.png"
    threshold_figure_path = FIGURES_DIR / f"{CLIP_EXPERIMENT_NAME}_threshold_curves.png"
    examples_figure_path = FIGURES_DIR / f"{CLIP_EXPERIMENT_NAME}_examples.png"

    model_path = MODELS_DIR / f"{CLIP_EXPERIMENT_NAME}_model.joblib"

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
        title=f"Matriz de confusion - {CLIP_EXPERIMENT_NAME}",
    )
    plot_threshold_curves(
        threshold_df=threshold_df,
        output_path=threshold_figure_path,
        title=f"Threshold analysis - {CLIP_EXPERIMENT_NAME}",
    )
    example_figure_output = save_example_figure(
        examples_df=examples_df,
        output_path=examples_figure_path,
        title=f"Ejemplos TP / FP / TN / FN - {CLIP_EXPERIMENT_NAME}",
    )

    joblib.dump(
        {
            "classifier": classifier,
            "class_names": list(class_names),
            "selected_classes": list(selected_classes),
            "model_id": model_id,
            "embedding_cache": {
                "train": train_cache_info,
                "test": test_cache_info,
            },
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
        "train_embedding_cache": train_cache_info["cache_dir"],
        "test_embedding_cache": test_cache_info["cache_dir"],
    }
    summary_text = build_experiment_summary(
        experiment_name=CLIP_EXPERIMENT_NAME,
        class_names=list(class_names),
        metrics=metrics,
        artifact_paths=artifact_paths,
        notes=[
            "Embeddings extraidos con CLIP y normalizados con L2 antes de entrenar LogisticRegression.",
            "Los embeddings se guardan en data/processed/clip_embeddings para evitar recomputarlos.",
        ],
    )
    save_text(summary_text, summary_path)

    return {
        "experiment_name": CLIP_EXPERIMENT_NAME,
        "class_names": list(class_names),
        "metrics": metrics,
        "classification_report": classification_report_dict,
        "artifact_paths": {key: str(value) if value is not None else None for key, value in artifact_paths.items()},
        "summary_path": str(summary_path),
    }


def build_parser() -> argparse.ArgumentParser:
    """Create a CLI parser for running only the CLIP experiment."""
    parser = argparse.ArgumentParser(description="Run CLIP + Logistic Regression for Hito 1.")
    parser.add_argument("--class-a", default=DEFAULT_TARGET_CLASSES[0], help="Negative class name.")
    parser.add_argument("--class-b", default=DEFAULT_TARGET_CLASSES[1], help="Positive class name.")
    parser.add_argument("--batch-size", type=int, default=CLIP_BATCH_SIZE, help="CLIP batch size.")
    parser.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore cached embeddings and recompute them.",
    )
    return parser


def main() -> None:
    """Entry point for direct CLI execution."""
    args = build_parser().parse_args()
    result = run_clip_experiment(
        selected_classes=(args.class_a, args.class_b),
        batch_size=args.batch_size,
        reuse_embeddings=not args.force_recompute_embeddings,
    )
    print(f"{result['experiment_name']} listo.")
    print(result["metrics"])


if __name__ == "__main__":
    main()
