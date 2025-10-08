# utils/paths.py
import os
from pathlib import Path


def r(path):
    return Path(os.path.expandvars(os.path.expanduser(path)))


MODELS_ROOT = r(os.getenv("MODELS_ROOT", "/mnt/models"))
SD_MODEL_PATH = r(os.getenv("SD_MODEL_PATH", MODELS_ROOT / "sd" / "sdxl"))
CONTROLNET_MODEL_PATH = r(
    os.getenv("CONTROLNET_MODEL_PATH", MODELS_ROOT / "sd" / "controlnet")
)
LORA_MODEL_PATH = r(os.getenv("LORA_MODEL_PATH", MODELS_ROOT / "sd" / "lora"))
VAE_MODEL_PATH = r(os.getenv("VAE_MODEL_PATH", MODELS_ROOT / "sd" / "vae"))
HF_HOME = r(os.getenv("HF_HOME", MODELS_ROOT / "hf"))
