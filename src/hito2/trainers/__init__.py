"""Training loops for Hito 2."""

from src.hito2.trainers.clip_mlp_trainer import EvaluationResult, TrainingConfig, TrainingResult, evaluate_clip_mlp_split, train_clip_mlp

__all__ = [
    "EvaluationResult",
    "TrainingConfig",
    "TrainingResult",
    "evaluate_clip_mlp_split",
    "train_clip_mlp",
]
