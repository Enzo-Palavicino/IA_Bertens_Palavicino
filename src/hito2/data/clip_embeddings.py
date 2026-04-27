"""CLIP embedding preparation for the Hito 2 multiclass experiment."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.clip_embeddings import extract_clip_embeddings, get_embedding_cache_paths, load_clip_components
from src.config import ROOT_DIR
from src.hito2.config import (
    DEFAULT_CLIP_EMBEDDING_BATCH_SIZE,
    DEFAULT_CLIP_MODEL_ID,
    DEFAULT_SPLIT_PATH,
    HITO2_CLIP_EMBEDDING_EXPERIMENT_NAME,
)
from src.hito2.data.splits import SPLIT_NAMES, load_split_dataframe
from src.report_utils import save_dataframe, save_json

HITO2_SPLIT_METADATA_FILENAME = "hito2_split_metadata.csv"
HITO2_SPLIT_INFO_FILENAME = "hito2_split_info.json"


@dataclass(frozen=True)
class ClipEmbeddingSplitBundle:
    """In-memory view of one split after CLIP feature extraction."""

    split_name: str
    embeddings: np.ndarray
    dataframe: pd.DataFrame
    cache_paths: dict[str, Path]

    @property
    def labels(self) -> np.ndarray:
        """Return integer class labels aligned with the embedding rows."""
        return self.dataframe["class_id"].to_numpy(dtype=np.int64, copy=True)


def get_class_names_from_split_dataframe(split_df: pd.DataFrame) -> list[str]:
    """Return class names ordered by the stable class id mapping."""
    class_frame = split_df[["class_id", "class_name"]].drop_duplicates().sort_values("class_id", kind="stable")
    return class_frame["class_name"].tolist()


def _filter_split_dataframe(split_df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return one split subset while preserving the CSV row order."""
    filtered_df = split_df[split_df["split"] == split_name].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"Split `{split_name}` has no rows in the provided split dataframe.")
    return filtered_df


def _resolve_image_paths(split_df: pd.DataFrame, root_dir: Path) -> list[Path]:
    """Resolve repo-relative manifest paths into absolute filesystem paths."""
    return [(root_dir / Path(image_path)).resolve() for image_path in split_df["image_path"].tolist()]


def _as_path_mapping(path_mapping: dict[str, str | Path]) -> dict[str, Path]:
    """Convert a string/path mapping into Path objects."""
    return {key: Path(value) for key, value in path_mapping.items()}


def _cache_exists(image_paths: list[Path], split_name: str, model_id: str) -> bool:
    """Check whether all cache files already exist for one split."""
    cache_paths = get_embedding_cache_paths(
        image_paths=image_paths,
        experiment_name=HITO2_CLIP_EMBEDDING_EXPERIMENT_NAME,
        split_name=split_name,
        model_id=model_id,
    )
    return all(path.exists() for path in cache_paths.values() if path != cache_paths["cache_dir"])


def _persist_hito2_cache_metadata(
    split_df: pd.DataFrame,
    split_name: str,
    resolved_image_paths: list[Path],
    cache_dir: Path,
) -> None:
    """Save Hito 2-specific metadata next to the cached embeddings."""
    metadata_df = split_df.copy()
    metadata_df["resolved_image_path"] = [str(path) for path in resolved_image_paths]
    metadata_df["embedding_index"] = np.arange(len(metadata_df), dtype=np.int64)

    save_dataframe(metadata_df, cache_dir / HITO2_SPLIT_METADATA_FILENAME, index=False)
    save_json(
        {
            "split_name": split_name,
            "num_images": int(len(metadata_df)),
            "num_classes_present": int(metadata_df["class_id"].nunique()),
            "class_names_present": (
                metadata_df[["class_id", "class_name"]]
                .drop_duplicates()
                .sort_values("class_id", kind="stable")["class_name"]
                .tolist()
            ),
        },
        cache_dir / HITO2_SPLIT_INFO_FILENAME,
    )


def prepare_clip_embedding_splits(
    split_path: str | Path = DEFAULT_SPLIT_PATH,
    root_dir: str | Path = ROOT_DIR,
    model_id: str = DEFAULT_CLIP_MODEL_ID,
    batch_size: int = DEFAULT_CLIP_EMBEDDING_BATCH_SIZE,
    reuse_cache: bool = True,
) -> tuple[dict[str, ClipEmbeddingSplitBundle], list[str]]:
    """
    Extract or load cached CLIP embeddings for train/val/test.

    The cache key depends on the exact ordered image paths per split, so if the
    split CSV changes the embeddings are recomputed automatically.
    """
    split_df = load_split_dataframe(split_path)
    class_names = get_class_names_from_split_dataframe(split_df)
    root_path = Path(root_dir).resolve()

    split_frames = {split_name: _filter_split_dataframe(split_df, split_name) for split_name in SPLIT_NAMES}
    split_image_paths = {
        split_name: _resolve_image_paths(split_frame, root_dir=root_path)
        for split_name, split_frame in split_frames.items()
    }

    needs_model = (not reuse_cache) or any(
        not _cache_exists(image_paths=image_paths, split_name=split_name, model_id=model_id)
        for split_name, image_paths in split_image_paths.items()
    )

    processor = None
    model = None
    device = None
    if needs_model:
        try:
            processor, model, device = load_clip_components(model_id=model_id)
        except OSError as error:
            raise OSError(
                "No se pudieron cargar los componentes CLIP. "
                "Si el modelo no esta cacheado localmente, la primera ejecucion necesita acceso a Hugging Face "
                f"para descargar `{model_id}`."
            ) from error

    split_bundles: dict[str, ClipEmbeddingSplitBundle] = {}
    for split_name in SPLIT_NAMES:
        split_frame = split_frames[split_name]
        resolved_paths = split_image_paths[split_name]

        embeddings, _, cache_paths_raw = extract_clip_embeddings(
            image_paths=resolved_paths,
            labels=split_frame["class_name"].tolist(),
            experiment_name=HITO2_CLIP_EMBEDDING_EXPERIMENT_NAME,
            split_name=split_name,
            batch_size=batch_size,
            model_id=model_id,
            reuse_cache=reuse_cache,
            processor=processor,
            model=model,
            device=device,
        )
        cache_paths = _as_path_mapping(cache_paths_raw)
        _persist_hito2_cache_metadata(
            split_df=split_frame,
            split_name=split_name,
            resolved_image_paths=resolved_paths,
            cache_dir=cache_paths["cache_dir"],
        )
        split_bundles[split_name] = ClipEmbeddingSplitBundle(
            split_name=split_name,
            embeddings=embeddings.astype(np.float32, copy=False),
            dataframe=split_frame.copy(),
            cache_paths=cache_paths,
        )

    return split_bundles, class_names
