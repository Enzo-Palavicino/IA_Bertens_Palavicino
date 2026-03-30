"""Central configuration for paths and base experiment parameters."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

QUERIES_DIR = RAW_DATA_DIR / "DocExplore_queries_web"
PAGES_DIR = RAW_DATA_DIR / "DocExplore_images"
EVALUATION_KIT_DIR = RAW_DATA_DIR / "evaluation_kit_v2"
CLIP_EMBEDDINGS_DIR = PROCESSED_DATA_DIR / "clip_embeddings"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

DEFAULT_TARGET_CLASSES = ("marqeur", "simple_sep")
RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_IMAGE_SIZE = (224, 224)
PIXEL_BASELINE_IMAGE_SIZE = (64, 64)
THRESHOLD_VALUES = tuple(round(value / 20, 2) for value in range(1, 20))
DEFAULT_EXAMPLES_PER_GROUP = 6

BASELINE_EXPERIMENT_NAME = "baseline_pixels"
CLIP_EXPERIMENT_NAME = "clip_logreg"

LOGREG_MAX_ITER = 2_000
CLIP_MODEL_ID = "openai/clip-vit-base-patch32"
CLIP_BATCH_SIZE = 16

VALID_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".gif",
}


def ensure_project_directories() -> None:
    """Create project directories used by the pipelines if they do not exist."""
    for directory in (
        OUTPUTS_DIR,
        FIGURES_DIR,
        METRICS_DIR,
        MODELS_DIR,
        PROCESSED_DATA_DIR,
        CLIP_EMBEDDINGS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def ensure_output_directories() -> None:
    """Backward-compatible wrapper for the original phase 1 helper."""
    ensure_project_directories()
