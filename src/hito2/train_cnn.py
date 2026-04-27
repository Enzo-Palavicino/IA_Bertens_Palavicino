"""CLI entrypoint for the Hito 2 CNN multiclass experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from torch import load as torch_load

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hito2.config import (
    DEFAULT_CNN_BATCH_SIZE,
    DEFAULT_CNN_CHANNELS,
    DEFAULT_CNN_CLASSIFIER_HIDDEN_DIM,
    DEFAULT_CNN_DROPOUT,
    DEFAULT_CNN_EARLY_STOPPING_MIN_DELTA,
    DEFAULT_CNN_EARLY_STOPPING_PATIENCE,
    DEFAULT_CNN_EPOCHS,
    DEFAULT_CNN_IMAGE_SIZE,
    DEFAULT_CNN_LR,
    DEFAULT_CNN_NUM_WORKERS,
    DEFAULT_CNN_USE_CLASS_WEIGHTS,
    DEFAULT_CNN_WEIGHT_DECAY,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_PATH,
    HITO2_CNN_FIGURES_DIR,
    HITO2_CNN_METRICS_DIR,
    HITO2_CNN_MODELS_DIR,
    HITO2_CNN_PREDICTIONS_DIR,
    ensure_hito2_output_directories,
)
from src.hito2.data.clip_embeddings import get_class_names_from_split_dataframe
from src.hito2.data.datasets import DocExploreCNNDataset
from src.hito2.data.splits import load_split_dataframe
from src.hito2.models.cnn import SimpleCNNClassifier
from src.hito2.trainers.cnn_trainer import CNNTrainingConfig, evaluate_cnn_split, train_cnn
from src.hito2.utils.experiment import (
    build_cnn_summary,
    resolve_device,
    set_global_seed,
    summarize_class_distribution,
    summarize_split_sizes_from_datasets,
)
from src.hito2.utils.metrics import save_confusion_matrix_figure
from src.report_utils import save_dataframe, save_json, save_text

EXPERIMENT_NAME = "hito2_cnn"


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI for the Hito 2 CNN experiment."""
    parser = argparse.ArgumentParser(description="Run the Hito 2 CNN multiclass experiment.")
    parser.add_argument("--split-path", type=Path, default=DEFAULT_SPLIT_PATH, help="CSV with train/val/test splits.")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=DEFAULT_CNN_IMAGE_SIZE,
        help="Direct resize used before the CNN receives each image.",
    )
    parser.add_argument(
        "--conv-channels",
        nargs="+",
        type=int,
        default=list(DEFAULT_CNN_CHANNELS),
        help="Output channels for each convolutional block.",
    )
    parser.add_argument(
        "--classifier-hidden-dim",
        type=int,
        default=DEFAULT_CNN_CLASSIFIER_HIDDEN_DIM,
        help="Hidden dimension of the final MLP classifier head.",
    )
    parser.add_argument("--dropout", type=float, default=DEFAULT_CNN_DROPOUT, help="Dropout in the classifier head.")
    parser.add_argument("--lr", type=float, default=DEFAULT_CNN_LR, help="Adam learning rate.")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_CNN_WEIGHT_DECAY, help="Adam weight decay.")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CNN_BATCH_SIZE, help="Training batch size.")
    parser.add_argument("--epochs", type=int, default=DEFAULT_CNN_EPOCHS, help="Maximum number of training epochs.")
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_CNN_EARLY_STOPPING_PATIENCE,
        help="Early stopping patience measured in epochs without validation-loss improvement.",
    )
    parser.add_argument(
        "--min-delta",
        type=float,
        default=DEFAULT_CNN_EARLY_STOPPING_MIN_DELTA,
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
        "--num-workers",
        type=int,
        default=DEFAULT_CNN_NUM_WORKERS,
        help="Number of DataLoader workers.",
    )
    return parser


def main() -> None:
    """Run the full CNN experiment for Hito 2."""
    args = build_parser().parse_args()

    ensure_hito2_output_directories()
    set_global_seed(args.seed)
    device = resolve_device(args.device)

    split_df = load_split_dataframe(args.split_path)
    class_names = get_class_names_from_split_dataframe(split_df)
    image_size = tuple(args.image_size)

    train_dataset = DocExploreCNNDataset(
        manifest_path=args.split_path,
        split="train",
        image_size=image_size,
    )
    val_dataset = DocExploreCNNDataset(
        manifest_path=args.split_path,
        split="val",
        image_size=image_size,
    )
    test_dataset = DocExploreCNNDataset(
        manifest_path=args.split_path,
        split="test",
        image_size=image_size,
    )

    model = SimpleCNNClassifier(
        num_classes=len(class_names),
        conv_channels=tuple(args.conv_channels),
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
    ).to(device)

    checkpoint_path = HITO2_CNN_MODELS_DIR / "best_model.pt"
    training_config = CNNTrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        min_delta=args.min_delta,
        use_class_weights=DEFAULT_CNN_USE_CLASS_WEIGHTS and not args.disable_class_weights,
        seed=args.seed,
        num_workers=args.num_workers,
    )

    training_result = train_cnn(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        class_names=class_names,
        checkpoint_path=checkpoint_path,
        device=device,
        config=training_config,
    )

    checkpoint = torch_load(checkpoint_path, map_location=device)
    training_result.model.load_state_dict(checkpoint["model_state_dict"])

    val_result = evaluate_cnn_split(
        model=training_result.model,
        dataset=val_dataset,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        class_weights=training_result.class_weights,
    )
    test_result = evaluate_cnn_split(
        model=training_result.model,
        dataset=test_dataset,
        class_names=class_names,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        class_weights=training_result.class_weights,
    )

    history_path = save_dataframe(training_result.history_df, HITO2_CNN_METRICS_DIR / "training_history.csv", index=False)
    val_metrics_path = save_json(val_result.metrics, HITO2_CNN_METRICS_DIR / "val_metrics.json")
    test_metrics_path = save_json(test_result.metrics, HITO2_CNN_METRICS_DIR / "test_metrics.json")
    val_report_path = save_json(
        val_result.classification_report,
        HITO2_CNN_METRICS_DIR / "val_classification_report.json",
    )
    test_report_path = save_json(
        test_result.classification_report,
        HITO2_CNN_METRICS_DIR / "test_classification_report.json",
    )
    val_predictions_path = save_dataframe(
        val_result.predictions_df,
        HITO2_CNN_PREDICTIONS_DIR / "val_predictions.csv",
        index=False,
    )
    test_predictions_path = save_dataframe(
        test_result.predictions_df,
        HITO2_CNN_PREDICTIONS_DIR / "test_predictions.csv",
        index=False,
    )
    val_confusion_path = save_confusion_matrix_figure(
        confusion=val_result.confusion_matrix,
        class_names=class_names,
        output_path=HITO2_CNN_FIGURES_DIR / "val_confusion_matrix.png",
        title="CNN confusion matrix - val",
    )
    test_confusion_path = save_confusion_matrix_figure(
        confusion=test_result.confusion_matrix,
        class_names=class_names,
        output_path=HITO2_CNN_FIGURES_DIR / "test_confusion_matrix.png",
        title="CNN confusion matrix - test",
    )

    split_sizes = summarize_split_sizes_from_datasets(
        train=train_dataset,
        val=val_dataset,
        test=test_dataset,
    )
    class_distribution = summarize_class_distribution(train_dataset.dataframe)
    experiment_config = {
        "experiment_name": EXPERIMENT_NAME,
        "split_path": str(Path(args.split_path).resolve()),
        "image_size": list(image_size),
        "conv_channels": list(args.conv_channels),
        "classifier_hidden_dim": int(args.classifier_hidden_dim),
        "dropout": float(args.dropout),
        "learning_rate": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "batch_size": int(args.batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "min_delta": float(args.min_delta),
        "seed": int(args.seed),
        "device": str(device),
        "num_workers": int(args.num_workers),
        "use_class_weights": bool(training_config.use_class_weights),
        "num_classes": len(class_names),
        "class_names": class_names,
        "split_sizes": split_sizes,
        "train_class_distribution": class_distribution,
        "class_weights": training_result.class_weights,
        "best_epoch": int(training_result.best_epoch),
        "best_val_loss": float(training_result.best_val_loss),
        "stopped_early": bool(training_result.stopped_early),
    }
    config_path = save_json(experiment_config, HITO2_CNN_MODELS_DIR / "experiment_config.json")

    summary_notes = [
        "Los splits train/val/test se leen directamente desde `query_splits.csv`; son exactamente los mismos usados por CLIP+MLP.",
        "El preprocesamiento usa resize directo a un tamano fijo sin padding de entrada para mantener un baseline simple y evitar bordes artificiales; la contracara es posible distorsion de aspecto.",
        "No se usan augmentations en este baseline para que la comparacion contra CLIP+MLP dependa sobre todo del modelo y no de una politica de regularizacion extra.",
        "El desbalance de clases se maneja por defecto con pesos inversos a la frecuencia calculados solo sobre train, para evitar leakage desde val/test.",
        "La validacion no contiene las clases `obj_3`, `obj_36`, `obj_37` y `obj_42` porque solo tienen dos ejemplos totales; esa limitacion afecta la interpretacion de metricas macro en val.",
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
    summary_text = build_cnn_summary(
        experiment_name=EXPERIMENT_NAME,
        class_names=class_names,
        config=experiment_config,
        split_sizes=split_sizes,
        val_metrics=val_result.metrics,
        test_metrics=test_result.metrics,
        artifact_paths=artifact_paths,
        notes=summary_notes,
    )
    summary_path = save_text(summary_text, HITO2_CNN_METRICS_DIR / "experiment_summary.md")

    print(f"Experimento listo: {EXPERIMENT_NAME}")
    print(f"Best epoch: {training_result.best_epoch}")
    print("Val metrics:", val_result.metrics)
    print("Test metrics:", test_result.metrics)
    print(f"Resumen guardado en: {summary_path}")


if __name__ == "__main__":
    main()
