import os
from pathlib import Path

# Primary project root and dataset directory resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_DIR = PROJECT_ROOT / "datasets"
FALLBACK_DATASET_DIR = Path(r"C:\Users\Mahi\Downloads\AI Logistics\datasets")

if DEFAULT_DATASET_DIR.exists():
    DATASET_DIR = DEFAULT_DATASET_DIR
elif FALLBACK_DATASET_DIR.exists():
    DATASET_DIR = FALLBACK_DATASET_DIR
else:
    DATASET_DIR = DEFAULT_DATASET_DIR

# Supported formats
SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls", ".json", ".parquet"}

def get_dataset_path(filename: str) -> Path:
    """Returns absolute path for a given dataset filename within DATASET_DIR."""
    path = DATASET_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset file '{filename}' not found in {DATASET_DIR}")
    return path
