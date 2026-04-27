"""Training and evaluation utilities for the Hito 2 CLIP+MLP experiment."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from src.hito2.data.clip_embeddings import ClipEmbeddingSplitBundle
from src.hito2.utils.metrics import (
    build_classification_report_dict,
    build_confusion_matrix,
    build_predictions_dataframe,
    calculate_multiclass_metrics,
)


@dataclass
class TrainingConfig:
    """Hyperparameters that control the MLP training loop."""

    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    patience: int
    min_delta: float
    use_class_weights: bool
    seed: int


@dataclass
class EvaluationResult:
    """Metrics and artifacts computed for one dataset split."""

    split_name: str
    loss: float
    metrics: dict[str, float | int]
    classification_report: dict[str, Any]
    predictions_df: pd.DataFrame
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    probabilities: np.ndarray


@dataclass
class TrainingResult:
    """Return value for the full training loop."""

    model: nn.Module
    history_df: pd.DataFrame
    best_epoch: int
    best_val_loss: float
    best_checkpoint_path: Path
    class_weights: list[float] | None
    stopped_early: bool


def _make_tensor_dataset(split_bundle: ClipEmbeddingSplitBundle) -> TensorDataset:
    """Convert one cached split bundle into a TensorDataset."""
    features = torch.from_numpy(split_bundle.embeddings.astype(np.float32, copy=False))
    labels = torch.from_numpy(split_bundle.labels.astype(np.int64, copy=False))
    return TensorDataset(features, labels)


def _make_dataloader(split_bundle: ClipEmbeddingSplitBundle, batch_size: int, shuffle: bool) -> DataLoader:
    """Build a DataLoader for one split."""
    return DataLoader(
        _make_tensor_dataset(split_bundle),
        batch_size=batch_size,
        shuffle=shuffle,
    )


def build_class_weight_tensor(train_labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from the training split only."""
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(np.float32)
    if np.any(class_counts <= 0):
        raise ValueError("All classes must appear in train to compute stable class weights.")
    weights = class_counts.sum() / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32)


def _softmax_numpy(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax for NumPy arrays."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def _run_inference(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    """Run a full forward pass over a dataloader without gradient updates."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_logits: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)

            batch_size = int(labels.size(0))
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size

            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    logits_array = np.vstack(all_logits)
    labels_array = np.concatenate(all_labels)
    probabilities = _softmax_numpy(logits_array)
    predictions = probabilities.argmax(axis=1)

    return {
        "loss": float(total_loss / max(total_samples, 1)),
        "y_true": labels_array,
        "y_pred": predictions,
        "probabilities": probabilities,
    }


def train_clip_mlp(
    model: nn.Module,
    train_bundle: ClipEmbeddingSplitBundle,
    val_bundle: ClipEmbeddingSplitBundle,
    class_names: Sequence[str],
    checkpoint_path: str | Path,
    device: torch.device,
    config: TrainingConfig,
) -> TrainingResult:
    """Train an MLP on cached CLIP embeddings with early stopping."""
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    train_loader = _make_dataloader(train_bundle, batch_size=config.batch_size, shuffle=True)
    val_loader = _make_dataloader(val_bundle, batch_size=config.batch_size, shuffle=False)

    class_weights_tensor = None
    class_weights_list = None
    if config.use_class_weights:
        class_weights_tensor = build_class_weight_tensor(train_bundle.labels, num_classes=len(class_names)).to(device)
        class_weights_list = class_weights_tensor.detach().cpu().tolist()

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_val_loss = float("inf")
    best_epoch = 0
    best_state_dict = deepcopy(model.state_dict())
    epochs_without_improvement = 0
    history_rows: list[dict[str, Any]] = []
    stopped_early = False

    for epoch in range(1, config.epochs + 1):
        model.train()
        total_train_loss = 0.0
        total_train_correct = 0
        total_train_samples = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = int(labels.size(0))
            total_train_loss += float(loss.item()) * batch_size
            total_train_samples += batch_size
            total_train_correct += int((logits.argmax(dim=1) == labels).sum().item())

        train_loss = float(total_train_loss / max(total_train_samples, 1))
        train_accuracy = float(total_train_correct / max(total_train_samples, 1))

        val_outputs = _run_inference(model=model, dataloader=val_loader, criterion=criterion, device=device)
        val_metrics = calculate_multiclass_metrics(
            y_true=val_outputs["y_true"],
            y_pred=val_outputs["y_pred"],
            probabilities=val_outputs["probabilities"],
            top_k=3,
        )
        val_loss = float(val_outputs["loss"])

        improved = val_loss < (best_val_loss - config.min_delta)
        if improved:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state_dict = deepcopy(model.state_dict())
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": best_state_dict,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "class_names": list(class_names),
                    "class_weights": class_weights_list,
                    "training_config": config.__dict__,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "improved_checkpoint": bool(improved),
                "epochs_without_improvement": int(epochs_without_improvement),
            }
        )

        if epochs_without_improvement >= config.patience:
            stopped_early = True
            break

    model.load_state_dict(best_state_dict)

    return TrainingResult(
        model=model,
        history_df=pd.DataFrame(history_rows),
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_checkpoint_path=checkpoint_path,
        class_weights=class_weights_list,
        stopped_early=stopped_early,
    )


def evaluate_clip_mlp_split(
    model: nn.Module,
    split_bundle: ClipEmbeddingSplitBundle,
    class_names: Sequence[str],
    batch_size: int,
    device: torch.device,
    class_weights: Sequence[float] | None = None,
) -> EvaluationResult:
    """Evaluate a trained MLP over one split and build all reporting artifacts."""
    dataloader = _make_dataloader(split_bundle, batch_size=batch_size, shuffle=False)
    weight_tensor = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    outputs = _run_inference(model=model, dataloader=dataloader, criterion=criterion, device=device)
    metrics = calculate_multiclass_metrics(
        y_true=outputs["y_true"],
        y_pred=outputs["y_pred"],
        probabilities=outputs["probabilities"],
        top_k=3,
    )
    metrics["loss"] = float(outputs["loss"])

    return EvaluationResult(
        split_name=split_bundle.split_name,
        loss=float(outputs["loss"]),
        metrics=metrics,
        classification_report=build_classification_report_dict(
            y_true=outputs["y_true"],
            y_pred=outputs["y_pred"],
            class_names=class_names,
        ),
        predictions_df=build_predictions_dataframe(
            metadata_df=split_bundle.dataframe,
            y_true=outputs["y_true"],
            probabilities=outputs["probabilities"],
            class_names=class_names,
            top_k=3,
        ),
        confusion_matrix=build_confusion_matrix(
            y_true=outputs["y_true"],
            y_pred=outputs["y_pred"],
            class_names=class_names,
        ),
        y_true=outputs["y_true"],
        y_pred=outputs["y_pred"],
        probabilities=outputs["probabilities"],
    )
