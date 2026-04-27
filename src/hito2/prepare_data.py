"""CLI entrypoint to build the Hito 2 manifest and deterministic splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.hito2.config import (
    DEFAULT_MANIFEST_PATH,
    DEFAULT_QUERY_DATASET_DIR,
    DEFAULT_RANDOM_STATE,
    DEFAULT_SPLIT_PATH,
    DEFAULT_SPLIT_SUMMARY_PATH,
    DEFAULT_TEST_RATIO,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    ensure_hito2_output_directories,
)
from src.hito2.data.manifest import (
    build_query_manifest_dataframe,
    save_query_manifest,
    summarize_manifest_dataframe,
)
from src.hito2.data.splits import (
    create_stratified_splits_dataframe,
    save_split_dataframe,
    save_split_summary,
    summarize_split_dataframe,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for Hito 2 dataset preparation."""
    parser = argparse.ArgumentParser(description="Prepare the Hito 2 multiclass query dataset.")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_QUERY_DATASET_DIR,
        help="Directory that contains one subdirectory per query class.",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Output CSV path for the manifest file.",
    )
    parser.add_argument(
        "--splits-output",
        type=Path,
        default=DEFAULT_SPLIT_PATH,
        help="Output CSV path for the split assignments.",
    )
    parser.add_argument(
        "--splits-summary-output",
        type=Path,
        default=DEFAULT_SPLIT_SUMMARY_PATH,
        help="Output JSON path for the split summary.",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO, help="Train split ratio.")
    parser.add_argument("--val-ratio", type=float, default=DEFAULT_VAL_RATIO, help="Validation split ratio.")
    parser.add_argument("--test-ratio", type=float, default=DEFAULT_TEST_RATIO, help="Test split ratio.")
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RANDOM_STATE,
        help="Random seed used for deterministic per-class shuffling.",
    )
    parser.add_argument(
        "--skip-splits",
        action="store_true",
        help="Generate only the base manifest without the split CSV.",
    )
    return parser


def _print_manifest_summary(summary: dict[str, object], manifest_path: Path) -> None:
    """Print a compact manifest summary for interactive runs."""
    print(f"Manifest guardado en: {manifest_path}")
    print(f"Total de imagenes: {summary['num_images']}")
    print(f"Total de clases: {summary['num_classes']}")
    print(
        "Rango de imagenes por clase: "
        f"{summary['min_images_per_class']} - {summary['max_images_per_class']}"
    )

    small_classes = summary["classes_with_fewer_than_three_images"]
    if small_classes:
        print("Clases con menos de 3 imagenes:", ", ".join(small_classes))


def _print_split_summary(summary: dict[str, object], split_path: Path, split_summary_path: Path) -> None:
    """Print a compact split summary for interactive runs."""
    print(f"Splits guardados en: {split_path}")
    print(f"Resumen de splits guardado en: {split_summary_path}")
    print("Cantidad por split:", summary["split_counts"])

    fallback_classes = summary["fallback_classes"]
    if fallback_classes:
        print(
            "Clases con fallback por baja frecuencia (<3 imagenes):",
            ", ".join(fallback_classes),
        )


def main() -> None:
    """Generate the Hito 2 manifest and, optionally, deterministic splits."""
    args = build_parser().parse_args()
    ensure_hito2_output_directories()

    manifest_df = build_query_manifest_dataframe(dataset_dir=args.dataset_dir)
    manifest_path = save_query_manifest(manifest_df, output_path=args.manifest_output)
    manifest_summary = summarize_manifest_dataframe(manifest_df)
    _print_manifest_summary(manifest_summary, manifest_path=manifest_path)

    if args.skip_splits:
        return

    split_df = create_stratified_splits_dataframe(
        manifest_df=manifest_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    split_path = save_split_dataframe(split_df, output_path=args.splits_output)
    split_summary = summarize_split_dataframe(
        split_df=split_df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    split_summary_path = save_split_summary(split_summary, output_path=args.splits_summary_output)
    _print_split_summary(split_summary, split_path=split_path, split_summary_path=split_summary_path)


if __name__ == "__main__":
    main()
