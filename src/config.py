"""Central configuration for paths and base experiment parameters."""

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT_DIR / "data" / "raw"
QUERIES_DIR = DATA_DIR / "DocExplore_queries_web"
PAGES_DIR = DATA_DIR / "DocExplore_images"
EVALUATION_KIT_DIR = DATA_DIR / "evaluation_kit_v2"

OUTPUTS_DIR = ROOT_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"

DEFAULT_TARGET_CLASSES = ("marqeur", "simple_sep")
RANDOM_STATE = 42
TEST_SIZE = 0.2
BASE_IMAGE_SIZE = (224, 224)

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


def ensure_output_directories() -> None:
    """Create output directories used by the project if they do not exist."""
    for directory in (OUTPUTS_DIR, FIGURES_DIR, METRICS_DIR, MODELS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
