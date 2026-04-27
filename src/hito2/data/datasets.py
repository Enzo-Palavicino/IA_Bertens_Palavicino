"""Reusable PyTorch datasets for Hito 2 CLIP and CNN experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from src.config import ROOT_DIR
from src.hito2.config import DEFAULT_CNN_IMAGE_SIZE
from src.hito2.data.manifest import load_manifest_dataframe, validate_manifest_dataframe
from src.preprocessing import load_image_rgb

try:
    _BILINEAR_RESAMPLE = Image.Resampling.BILINEAR
except AttributeError:
    _BILINEAR_RESAMPLE = Image.BILINEAR


def _resolve_image_path(image_path: str | Path, root_dir: Path) -> Path:
    """Resolve a manifest path against the repository root when needed."""
    path = Path(image_path)
    if path.is_absolute():
        return path
    return root_dir / path


def _ensure_split_column(dataframe: pd.DataFrame, split: str | None) -> None:
    """Validate that the split column exists when a split filter is requested."""
    if split is not None and "split" not in dataframe.columns:
        raise ValueError("A split filter was requested but the dataframe has no `split` column.")


def _load_dataset_dataframe(
    manifest_path: str | Path | None,
    dataframe: pd.DataFrame | None,
    split: str | None,
) -> pd.DataFrame:
    """Load or copy the source dataframe and apply an optional split filter."""
    if manifest_path is None and dataframe is None:
        raise ValueError("Either `manifest_path` or `dataframe` must be provided.")

    if dataframe is None:
        loaded_df = load_manifest_dataframe(manifest_path)
    else:
        loaded_df = dataframe.copy()
        validate_manifest_dataframe(loaded_df)

    _ensure_split_column(loaded_df, split=split)
    if split is not None:
        loaded_df = loaded_df[loaded_df["split"] == split].reset_index(drop=True)

    return loaded_df.reset_index(drop=True)


def _squeeze_processor_batch(processed: Mapping[str, Any]) -> dict[str, Any]:
    """Remove the batch dimension added by Hugging Face image processors."""
    squeezed: dict[str, Any] = {}
    for key, value in processed.items():
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == 1:
            squeezed[key] = value.squeeze(0)
        else:
            squeezed[key] = value
    return squeezed


def _pil_image_to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an RGB PIL image into a float tensor in CHW format."""
    image_array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(image_array.transpose(2, 0, 1)) / 255.0


class _BaseDocExploreDataset(Dataset):
    """Common dataframe-backed behavior shared by all Hito 2 datasets."""

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        split: str | None = None,
        root_dir: str | Path = ROOT_DIR,
    ) -> None:
        self.dataframe = _load_dataset_dataframe(
            manifest_path=manifest_path,
            dataframe=dataframe,
            split=split,
        )
        self.root_dir = Path(root_dir).resolve()
        self.split = split

    def __len__(self) -> int:
        return len(self.dataframe)

    @property
    def num_classes(self) -> int:
        """Return the number of classes represented in the current view."""
        return int(self.dataframe["class_id"].nunique())

    def get_class_names(self) -> list[str]:
        """Return class names ordered by class id."""
        ordered = self.dataframe[["class_id", "class_name"]].drop_duplicates().sort_values("class_id", kind="stable")
        return ordered["class_name"].tolist()

    def _row_to_metadata(self, row: pd.Series) -> dict[str, Any]:
        """Extract common metadata fields returned by all dataset variants."""
        return {
            "label": int(row["class_id"]),
            "class_id": int(row["class_id"]),
            "class_name": str(row["class_name"]),
            "image_path": str(row["image_path"]),
        }

    def _load_image(self, row: pd.Series) -> Image.Image:
        """Load one image referenced by the manifest row."""
        resolved_path = _resolve_image_path(row["image_path"], root_dir=self.root_dir)
        return load_image_rgb(resolved_path)


class DocExploreClipDataset(_BaseDocExploreDataset):
    """
    Dataset for CLIP-style image inputs.

    If `processor` is provided, the dataset returns a `clip_inputs` dictionary
    ready to feed into a CLIP image encoder. Otherwise it returns the raw RGB
    PIL image under the `image` key so callers can plug in their own pipeline.
    """

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        split: str | None = None,
        root_dir: str | Path = ROOT_DIR,
        processor: Any | None = None,
        transform: Any | None = None,
    ) -> None:
        super().__init__(manifest_path=manifest_path, dataframe=dataframe, split=split, root_dir=root_dir)

        if processor is not None and transform is not None:
            raise ValueError("Provide either `processor` or `transform`, not both.")

        self.processor = processor
        self.transform = transform

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        image = self._load_image(row)
        sample = self._row_to_metadata(row)

        if self.processor is not None:
            processed = self.processor(images=image, return_tensors="pt")
            if isinstance(processed, Mapping):
                sample["clip_inputs"] = _squeeze_processor_batch(processed)
            else:
                sample["clip_inputs"] = processed
        elif self.transform is not None:
            sample["image"] = self.transform(image)
        else:
            sample["image"] = image

        return sample


class DocExploreCNNDataset(_BaseDocExploreDataset):
    """
    Dataset for CNN baselines based on direct image resizing.

    The default preprocessing is intentionally simple for Hito 2: convert to
    RGB, resize directly to a fixed size, then return a float tensor in `[0, 1]`.
    """

    def __init__(
        self,
        manifest_path: str | Path | None = None,
        dataframe: pd.DataFrame | None = None,
        split: str | None = None,
        root_dir: str | Path = ROOT_DIR,
        image_size: tuple[int, int] = DEFAULT_CNN_IMAGE_SIZE,
        transform: Any | None = None,
    ) -> None:
        super().__init__(manifest_path=manifest_path, dataframe=dataframe, split=split, root_dir=root_dir)
        self.image_size = tuple(image_size)
        self.transform = transform

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        image = self._load_image(row)
        resized_image = image.resize(self.image_size, resample=_BILINEAR_RESAMPLE)
        image_tensor = _pil_image_to_tensor(resized_image)

        if self.transform is not None:
            image_tensor = self.transform(image_tensor)

        sample = self._row_to_metadata(row)
        sample["image"] = image_tensor
        return sample
