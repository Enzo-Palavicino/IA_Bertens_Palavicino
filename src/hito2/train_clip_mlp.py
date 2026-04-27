"""CLI entrypoint for the Hito 2 CLIP encoder + MLP multiclass experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch import load as torch_load

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hito2.config import (
    DEFAULT_CLIP_EMBEDDING_BATCH_SIZE,
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_CLIP_MLP_BATCH_SIZE,
    DEFAULT_CLIP_MLP_DROPOUT,
    DEFAULT_CLIP_MLP_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_CLIP_MLP_EARLY_STOPPING_PATIENCE,
    DEFAULT_CLIP_MLP_EPOCHS,
    DEFAULT_CLIP_MLP_HIDDEN_SIZES,
    DEFAULT_CLIP_MLP_LR,
    DEFAULT_CLIP_MLP_USE_CLASS_WEIGHTS,
    DEFAULT_CLIP_MLP_WEIGHT_DECAY,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_PATH,
    HITO2_CLIP_MLP_FIGURES_DIR,
    HITO2_CLIP_MLP_METRICS_DIR,
    HITO2_CLIP_MLP_MODELS_DIR,
    HITO2_CLIP_MLP_PREDICTIONS_DIR,
    ensure_hito2_output_directories,
)
from src.hito2.data.clip_embeddings import get_class_names_from_split_dataframe, prepare_clip_embedding_splits
from src.hito2.data.splits import load_split_dataframe
from src.hito2.models.clip_mlp import ClipMLPClassifier
from src.hito2.trainers.clip_mlp_trainer import TrainingConfig, evaluate_clip_mlp_split, train_clip_mlp
from src.hito2.utils.experiment import (
    build_clip_mlp_summary,
    resolve_device,
    set_global_seed,
    summarize_class_distribution,
    summarize_split_sizes,
)
from src.hito2.utils.metrics import save_confusion_matrix_figure
from src.report_utils import save_dataframe, save_json, save_text

EXPERIMENT_NAME = "hito2_clip_mlp"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for the Hito 2 CLIP+MLP experiment."""
    parser = argparse.ArgumentParser(description="Run the Hito 2 CLIP encoder + MLP multiclass experiment.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH, help="CSV with train/val/test splits.")
    parser.add_argument("--model-id", default=DEFAULT_CLIP_MODEL_ID, help="Hugging Face CLIP model id.")
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=DEFAULT_CLIP_EMBEDDING_BATCH_SIZE,
        help="Batch size used while extracting CLIP embeddings.",
    )
    parser.add_argument(
        "--hidden-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CLIP_MLP_HIDDEN_SIZES),
        help="Hidden layer sizes for the MLP head.",
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_CLIP_MLP_DROPOUT, help="Dropout after each hidden layer.")
    parser.add_argument("--lr", type=float, default=DEFAULT_CLIP_MLP_LR, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CLIP_MLP_WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CLIP_MLP_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CLIP_MLP_EPOCHS, help="Maximum number of training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CLIP_MLP_EARLY_STOPPING_PATIENCE,
        help="Early stopping patience measured in epochs without validation-loss improvement.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_CLIP_MLP_EARLY_STOPPING_MIN_DELTA,
        help="Minimum validation-loss improvement required to reset patience.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE, help="Global random seed.")
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device to use: `auto`, `cpu`, `cuda`, or a specific device string.",
    )
    parser.add_argument(
        "--disable-class-weights",
        action="store_true",
        help="Disable inverse-frequency class weights in CrossEntropyLoss.",
    )
    parser.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore cached embeddings and recompute them from the raw images.",
    )
    return parser


def main() -> None:
    """Run the full CLIP+MLP experiment for Hito 2."""
    args = build_parser().parse_args()

    ensure_hito2_output_directories()
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    split_df = load_split_dataframe(args.split_path)
    class_names = get_class_names_from_split_dataframe(split_df)
    split_bundles, _ = prepare_clip_embedding_splits(
        split_path=args.split_path,
        model_id=args.model_id,
        batch_size=args.embedding_batch_size,
        reuse_cache=not args.force_recompute_embeddings,
    )

    train_bundle = split_bundles["train"]
    val_bundle = split_bundles["val"]
    test_bundle = split_bundles["test"]

    input_dim = int(train_bundle.embeddings.shape[1])
    num_classes = len(class_names)
    model = ClipMLPClassifier(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_sizes=tuple(args.hidden_sizes),
        dropout=args.dropout,
    ).to(device)

    checkpoint_path = HITO2_CLIP_MLP_MODELS_DIR / "best_model.pt"
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        use_class_weights=DEFAULT_CLIP_MLP_USE_CLASS_WEIGHTS and not args.disable_class_weights,
        seed=args.seed,
    )

    training_result = train_clip_mlp(
        model=model,
        train_bundle=train_bundle,
        val_bundle=val_bundle,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        device=device,
        config=training_config,
    )

    checkpoint = torch_load(checkpoint_path, map_location=device)
    training_result.model.load_state_dict(checkpoint["model_state_dict"])

    val_result = evaluate_clip_mlp_split(
        model=training_result.model,
        split_bundle=val_bundle,
        class_names=class_names,
        batch_size=args.batch_size,
        device=device,
        class_weights=training_result.class_weights,
    )
    test_result = evaluate_clip_mlp_split(
        model=training_result.model,
        split_bundle=test_bundle,
        class_names=class_names,
        batch_size=args.batch_size,
        device=device,
        class_weights=training_result.class_weights,
    )

    history_path = save_dataframe(training_result.history_df, HITO2_CLIP_MLP_METRICS_DIR / "training_history.csv", index=False)
    val_metrics_path = save_json(val_result.metrics, HITO2_CLIP_MLP_METRICS_DIR / "val_metrics.json")
    test_metrics_path = save_json(test_result.metrics, HITO2_CLIP_MLP_METRICS_DIR / "test_metrics.json")
    val_report_path = save_json(
        val_result.classification_report,
        HITO2_CLIP_MLP_METRICS_DIR / "val_classification_report.json",
    )
    test_report_path = save_json(
        test_result.classification_report,
        HITO2_CLIP_MLP_METRICS_DIR / "test_classification_report.json",
    )
    val_predictions_path = save_dataframe(
        val_result.predictions_df,
        HITO2_CLIP_MLP_PREDICTIONS_DIR / "val_predictions.csv",
        index=False,
    )
    test_predictions_path = save_dataframe(
        test_result.predictions_df,
        HITO2_CLIP_MLP_PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )
    val_confusion_path = save_confusion_matrix_figure(
        confusion=val_result.confusion_matrix,
        class_names=class_names,
        output_path=HITO2_CLIP_MLP_FIGURES_DIR / "val_confusion_matrix.png",
        title="CLIP+MLP confusion matrix - val",
    )
    test_confusion_path = save_confusion_matrix_figure(
        confusion=test_result.confusion_matrix,
        class_names=class_names,
        output_path=HITO2_CLIP_MLP_FIGURES_DIR / "test_confusion_matrix.png",
        title="CLIP+MLP confusion matrix - test",
    )

    class_distribution = summarize_class_distribution(train_bundle.dataframe)
    split_sizes = summarize_split_sizes(split_bundles)
    embedding_cache_paths = {
        split_name: {key: str(path) for key, path in bundle.cache_paths.items()}
        for split_name, bundle in split_bundles.items()
    }
    experiment_config = {
        "experiment_name": EXPERIMENT_NAME,
        "split_path": str(Path(args.split_path).resolve()),
        "model_id": args.model_id,
        "embedding_batch_size": int(args.embedding_batch_size),
        "hidden_sizes": list(args.hidden_sizes),
        "dropout": float(args.dropout),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "seed": int(args.seed),
        "device": str(device),
        "use_class_weights": bool(training_config.use_class_weights),
        "force_recompute_embeddings": bool(args.force_recompute_embeddings),
        "input_dim": input_dim,
        "num_classes": num_classes,
        "class_names": class_names,
        "split_sizes": split_sizes,
        "train_class_distribution": class_distribution,
        "class_weights": training_result.class_weights,
        "embedding_cache_paths": embedding_cache_paths,
        "best_epoch": int(training_result.best_epoch),
        "best_val_loss": float(training_result.best_val_loss),
        "stopped_early": bool(training_result.stopped_early),
    }
    config_path = save_json(experiment_config, HITO2_CLIP_MLP_MODELS_DIR / "experiment_config.json")

    summary_notes = [
        "Los splits train/val/test se leen directamente desde `query_splits.csv`; no se regeneran en este script.",
        "Los embeddings CLIP se cachean por split y por conjunto exacto de rutas, evitando recalculo salvo que cambie el split o se use `--force-recompute-embeddings`.",
        "El desbalance de clases se maneja por defecto con pesos inversos a la frecuencia calculados solo sobre train, para evitar leakage desde val/test.",
        "La validacion no contiene las clases `obj_3`, `obj_36`, `obj_37` y `obj_42` porque solo tienen dos ejemplos totales; esa limitacion afecta la lectura de metricas macro en val.",
        "El encoder CLIP permanece congelado: este experimento solo entrena la cabeza MLP sobre embeddings precomputados.",
    ]
    artifact_paths = {
        "best_checkpoint": checkpoint_path,
        "experiment_config_json": config_path,
        "training_history_csv": history_path,
        "val_metrics_json": val_metrics_path,
        "test_metrics_json": test_metrics_path,
        "val_report_json": val_report_path,
        "test_report_json": test_report_path,
        "val_predictions_csv": val_predictions_path,
        "test_predictions_csv": test_predictions_path,
        "val_confusion_png": val_confusion_path,
        "test_confusion_png": test_confusion_path,
    }
    summary_text = build_clip_mlp_summary(
        experiment_name=EXPERIMENT_NAME,
        class_names=class_names,
        config=experiment_config,
        split_sizes=split_sizes,
        val_metrics=val_result.metrics,
        test_metrics=test_result.metrics,
        artifact_paths=artifact_paths,
        notes=summary_notes,
    )
    summary_path = save_text(summary_text, HITO2_CLIP_MLP_METRICS_DIR / "experiment_summary.md")

    print(f"Experimento listo: {EXPERIMENT_NAME}")
    print(f"Best epoch: {training_result.best_epoch}")
    print("Val metrics:", val_result.metrics)
    print("Test metrics:", test_result.metrics)
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
