"""Main entry point for running Hito 1 experiments."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.baseline_pixels import run_baseline_experiment
from src.config import (
    CLIP_BATCH_SIZE,
    DEFAULT_TARGET_CLASSES,
    METRICS_DIR,
    PIXEL_BASELINE_IMAGE_SIZE,
    ensure_project_directories,
)
from src.report_utils import save_dataframe, save_text
from src.train_clip_logreg import run_clip_experiment


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the Hito 1 runner."""
    parser = argparse.ArgumentParser(description="Run the Hito 1 DocExplore experiments.")
    parser.add_argument(
        "--experiment",
        choices=("baseline", "clip", "both"),
        default="both",
        help="Which experiment(s) to run.",
    )
    parser.add_argument("--class-a", default=DEFAULT_TARGET_CLASSES[0], help="Negative class.")
    parser.add_argument("--class-b", default=DEFAULT_TARGET_CLASSES[1], help="Positive class.")
    parser.add_argument(
        "--image-size",
        nargs=2,
        type=int,
        metavar=("WIDTH", "HEIGHT"),
        default=PIXEL_BASELINE_IMAGE_SIZE,
        help="Resize used by the pixel baseline.",
    )
    parser.add_argument("--clip-batch-size", type=int, default=CLIP_BATCH_SIZE, help="CLIP batch size.")
    parser.add_argument(
        "--force-recompute-embeddings",
        action="store_true",
        help="Ignore cached CLIP embeddings.",
    )
    return parser


def _build_comparison_dataframe(results: list[dict]) -> pd.DataFrame:
    """Create a flat comparison table across executed experiments."""
    rows = []
    for result in results:
        metrics = result["metrics"]
        rows.append(
            {
                "experiment_name": result["experiment_name"],
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "train_size": metrics["train_size"],
                "test_size": metrics["test_size"],
                "positive_class_name": metrics["positive_class_name"],
            }
        )
    return pd.DataFrame(rows).sort_values(by="f1", ascending=False).reset_index(drop=True)


def _console_summary(comparison_df: pd.DataFrame) -> str:
    """Format a compact text summary for stdout and markdown export."""
    lines = ["Resumen final de Hito 1", ""]
    for _, row in comparison_df.iterrows():
        lines.append(
            f"- {row['experiment_name']}: accuracy={row['accuracy']:.4f}, "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, f1={row['f1']:.4f}"
        )

    if not comparison_df.empty:
        best = comparison_df.iloc[0]
        lines.extend(
            [
                "",
                f"Mejor F1: {best['experiment_name']} ({best['f1']:.4f})",
            ]
        )

    return "\n".join(lines)


def main() -> None:
    """Run the selected Hito 1 experiments and print a comparison summary."""
    args = build_parser().parse_args()
    ensure_project_directories()

    selected_classes = (args.class_a, args.class_b)
    results: list[dict] = []

    if args.experiment in {"baseline", "both"}:
        results.append(
            run_baseline_experiment(
                selected_classes=selected_classes,
                image_size=tuple(args.image_size),
            )
        )

    if args.experiment in {"clip", "both"}:
        results.append(
            run_clip_experiment(
                selected_classes=selected_classes,
                batch_size=args.clip_batch_size,
                reuse_embeddings=not args.force_recompute_embeddings,
            )
        )

    comparison_df = _build_comparison_dataframe(results)
    comparison_csv_path = METRICS_DIR / "hito1_experiment_comparison.csv"
    comparison_md_path = METRICS_DIR / "hito1_experiment_comparison.md"

    save_dataframe(comparison_df, comparison_csv_path, index=False)
    save_text(_console_summary(comparison_df), comparison_md_path)

    print(_console_summary(comparison_df))
    print(f"\nComparacion guardada en: {comparison_csv_path}")


if __name__ == "__main__":
    main()
