"""Initial dataset inspection for DocExplore query and page images."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FIGURES_DIR, METRICS_DIR, PAGES_DIR, QUERIES_DIR, ensure_output_directories
from src.data_loader import is_valid_image_file

CLASS_COUNTS_CSV = METRICS_DIR / "class_counts.csv"
CLASS_DISTRIBUTION_PLOT = FIGURES_DIR / "class_distribution.png"
TOP_CLASSES_TO_PLOT = 15


def inspect_query_dataset(queries_dir: Path = QUERIES_DIR) -> pd.DataFrame:
    """Count valid images per class and return a sorted dataframe."""
    if not queries_dir.exists():
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

    class_records: list[dict[str, int | str]] = []

    for class_dir in sorted(path for path in queries_dir.iterdir() if path.is_dir()):
        valid_count = sum(1 for file_path in class_dir.iterdir() if is_valid_image_file(file_path))
        class_records.append({"class_name": class_dir.name, "valid_images": valid_count})

    counts_df = pd.DataFrame(class_records).sort_values(
        by="valid_images",
        ascending=False,
        kind="stable",
    )
    counts_df.reset_index(drop=True, inplace=True)
    return counts_df


def count_valid_page_images(pages_dir: Path = PAGES_DIR) -> int:
    """Count valid full-page images available in the raw page directory."""
    if not pages_dir.exists():
        raise FileNotFoundError(f"Pages directory not found: {pages_dir}")

    return sum(1 for file_path in pages_dir.iterdir() if is_valid_image_file(file_path))


def save_class_counts(counts_df: pd.DataFrame, output_path: Path = CLASS_COUNTS_CSV) -> None:
    """Persist class counts to CSV."""
    counts_df.to_csv(output_path, index=False)


def plot_class_distribution(
    counts_df: pd.DataFrame,
    output_path: Path = CLASS_DISTRIBUTION_PLOT,
    top_n: int = TOP_CLASSES_TO_PLOT,
) -> None:
    """Save a bar chart for the classes with the most valid images."""
    plot_df = counts_df.head(top_n).sort_values(by="valid_images", ascending=True)

    plt.figure(figsize=(10, 8))
    sns.barplot(data=plot_df, x="valid_images", y="class_name", palette="Blues_r")
    plt.title(f"Top {len(plot_df)} classes by valid images")
    plt.xlabel("Valid images")
    plt.ylabel("Class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


def print_summary(counts_df: pd.DataFrame, valid_page_count: int) -> None:
    """Print the dataset summary requested for the milestone."""
    total_classes = len(counts_df)
    top_10 = counts_df.head(10)
    suggested_classes = counts_df.head(2)["class_name"].tolist()

    print(f"Cantidad total de clases: {total_classes}")
    print("\nTop 10 clases con mas imagenes validas:")
    for _, row in top_10.iterrows():
        print(f"- {row['class_name']}: {row['valid_images']}")

    if len(suggested_classes) == 2:
        print(
            "\nSugerencia automatica de 2 clases con mas imagenes: "
            f"{suggested_classes[0]}, {suggested_classes[1]}"
        )
    else:
        print("\nNo hay suficientes clases para sugerir dos clases.")

    print(f"\nCantidad de paginas completas validas: {valid_page_count}")
    print(f"CSV guardado en: {CLASS_COUNTS_CSV}")
    print(f"Grafico guardado en: {CLASS_DISTRIBUTION_PLOT}")


def main() -> None:
    """Run the initial dataset inspection."""
    ensure_output_directories()

    counts_df = inspect_query_dataset()
    valid_page_count = count_valid_page_images()

    save_class_counts(counts_df)
    plot_class_distribution(counts_df)
    print_summary(counts_df, valid_page_count)


if __name__ == "__main__":
    main()
