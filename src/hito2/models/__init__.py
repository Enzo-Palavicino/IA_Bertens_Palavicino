"""Model definitions for Hito 2."""

from src.hito2.models.cnn import SimpleCNNClassifier
from src.hito2.models.clip_mlp import ClipMLPClassifier

__all__ = ["ClipMLPClassifier", "SimpleCNNClassifier"]
