"""CLIP embedding extraction utilities with on-disk caching."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from src.config import CLIP_BATCH_SIZE, CLIP_EMBEDDINGS_DIR, CLIP_MODEL_ID, ensure_project_directories
from src.preprocessing import load_image_rgb
from src.report_utils import save_json


def get_device() -> torch.device:
    """Return the preferred torch device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_clip_components(model_id: str = CLIP_MODEL_ID) -> tuple[CLIPProcessor, CLIPModel, torch.device]:
    """Load the CLIP processor and model on the best available device."""
    device = get_device()
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id).to(device)
    model.eval()
    return processor, model, device


def _project_if_needed(features: torch.Tensor, model: CLIPModel) -> torch.Tensor:
    """Apply CLIP visual projection only when the feature size requires it."""
    projection = getattr(model, "visual_projection", None)
    if projection is None:
        return features

    in_features = getattr(projection, "in_features", None)
    if in_features is None or features.shape[-1] != in_features:
        return features

    return projection(features)


def _as_image_feature_tensor(output: object, model: CLIPModel) -> torch.Tensor:
    """
    Convert several possible CLIP outputs into a 2D image-feature tensor.

    Different `transformers` versions can return either a tensor directly or a
    structured object such as `BaseModelOutputWithPooling`. This helper makes
    the downstream normalization logic version-agnostic.
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, (tuple, list)) and output:
        return _as_image_feature_tensor(output[0], model)

    pooler_output = getattr(output, "pooler_output", None)
    if pooler_output is not None:
        return _project_if_needed(pooler_output, model)

    last_hidden_state = getattr(output, "last_hidden_state", None)
    if last_hidden_state is not None:
        cls_embedding = last_hidden_state[:, 0, :]
        return _project_if_needed(cls_embedding, model)

    raise TypeError(f"Unsupported CLIP image feature output type: {type(output)!r}")


def _extract_image_feature_tensor(model: CLIPModel, pixel_values: torch.Tensor) -> torch.Tensor:
    """Extract a normalized-ready 2D tensor of CLIP image embeddings."""
    output = model.get_image_features(pixel_values=pixel_values)
    image_features = _as_image_feature_tensor(output, model)

    if image_features.ndim != 2:
        raise ValueError(f"Expected a 2D image feature tensor, got shape {tuple(image_features.shape)}")

    return image_features


def _cache_key(image_paths: Sequence[str | Path], model_id: str) -> str:
    """Build a deterministic cache key from the model id and image paths."""
    digest = hashlib.md5()
    digest.update(model_id.encode("utf-8"))
    for path in image_paths:
        digest.update(str(Path(path).resolve()).encode("utf-8"))
    return digest.hexdigest()[:12]


def get_embedding_cache_paths(
    image_paths: Sequence[str | Path],
    experiment_name: str,
    split_name: str,
    model_id: str = CLIP_MODEL_ID,
) -> dict[str, Path]:
    """Return all paths used by one cached embedding block."""
    cache_dir = CLIP_EMBEDDINGS_DIR / experiment_name / f"{split_name}_{_cache_key(image_paths, model_id)}"
    return {
        "cache_dir": cache_dir,
        "embeddings": cache_dir / "embeddings.npy",
        "metadata": cache_dir / "metadata.csv",
        "info": cache_dir / "info.json",
    }


def extract_clip_embeddings(
    image_paths: Sequence[str | Path],
    labels: Sequence[str] | None,
    experiment_name: str,
    split_name: str,
    batch_size: int = CLIP_BATCH_SIZE,
    model_id: str = CLIP_MODEL_ID,
    reuse_cache: bool = True,
    processor: CLIPProcessor | None = None,
    model: CLIPModel | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, str]]:
    """
    Extract CLIP image embeddings and cache them under data/processed/.

    If cached files already exist and reuse_cache=True, embeddings are loaded
    directly from disk without re-running the CLIP encoder.
    """
    ensure_project_directories()
    cache_paths = get_embedding_cache_paths(
        image_paths=image_paths,
        experiment_name=experiment_name,
        split_name=split_name,
        model_id=model_id,
    )

    embeddings_path = cache_paths["embeddings"]
    metadata_path = cache_paths["metadata"]
    info_path = cache_paths["info"]

    if reuse_cache and embeddings_path.exists() and metadata_path.exists() and info_path.exists():
        embeddings = np.load(embeddings_path)
        metadata = pd.read_csv(metadata_path)
        return embeddings, metadata, {key: str(value) for key, value in cache_paths.items()}

    if processor is None or model is None or device is None:
        processor, model, device = load_clip_components(model_id=model_id)

    cache_paths["cache_dir"].mkdir(parents=True, exist_ok=True)

    embeddings_batches: list[np.ndarray] = []
    path_list = [Path(path) for path in image_paths]

    for start_idx in tqdm(
        range(0, len(path_list), batch_size),
        desc=f"Extrayendo embeddings CLIP ({split_name})",
        leave=False,
    ):
        batch_paths = path_list[start_idx : start_idx + batch_size]
        batch_images = [load_image_rgb(path) for path in batch_paths]
        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            image_features = _extract_image_feature_tensor(model=model, pixel_values=pixel_values)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        embeddings_batches.append(image_features.cpu().numpy().astype(np.float32))

    embeddings = np.vstack(embeddings_batches)
    metadata_dict: dict[str, list[str]] = {
        "image_path": [str(path.resolve()) for path in path_list],
        "split": [split_name] * len(path_list),
    }
    if labels is not None:
        metadata_dict["label"] = list(labels)
    metadata = pd.DataFrame(metadata_dict)

    np.save(embeddings_path, embeddings)
    metadata.to_csv(metadata_path, index=False)
    save_json(
        {
            "experiment_name": experiment_name,
            "split_name": split_name,
            "model_id": model_id,
            "batch_size": batch_size,
            "device": str(device),
            "num_images": len(path_list),
            "embedding_dim": int(embeddings.shape[1]),
            "cache_key": _cache_key(image_paths, model_id),
        },
        info_path,
    )

    return embeddings, metadata, {key: str(value) for key, value in cache_paths.items()}
