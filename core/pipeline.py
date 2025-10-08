from api.config import settings
from pathlib import Path
import torch

WEIGHT_FILE = "....pth"


def get_weight_path() -> Path:
    return settings.PATHS["models"] / "pipeline" / WEIGHT_FILE


def load_model():
    weight = get_weight_path()
    assert weight.exists(), f"Weight not found: {weight}"
    # ... load state dict with torch ...
