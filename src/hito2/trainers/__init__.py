"""Training loops for Hito 2."""

from src.hito2.trainers.cnn_trainer import CNNTrainingConfig, CNNEvaluationResult, CNNTrainingResult, evaluate_cnn_split, train_cnn
from src.hito2.trainers.clip_mlp_trainer import EvaluationResult, TrainingConfig, TrainingResult, evaluate_clip_mlp_split, train_clip_mlp

__all__ = [
    "CNNTrainingConfig",
    "CNNEvaluationResult",
    "CNNTrainingResult",
    "EvaluationResult",
    "TrainingConfig",
    "TrainingResult",
    "evaluate_cnn_split",
    "evaluate_clip_mlp_split",
    "train_cnn",
    "train_clip_mlp",
]
