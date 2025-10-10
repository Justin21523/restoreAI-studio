import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from pydantic import field_validator
from pydantic_settings import BaseSettings
from functools import lru_cache
import torch


class Settings(BaseSettings):
    """Application settings with environment variables"""

    # API Configuration
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"

    # Shared Cache Configuration
    ai_cache_root: Path = Path("/mnt/c/AI_LLM_projects/ai_warehouse")
    models_dir: Path = Path()
    cache_dir: Path = Path()

    # Model defaults
    default_sr_model: str = "realesrgan-x4plus"
    default_face_model: str = "gfpgan"
    default_face_strength: float = 0.5

    # Performance settings
    device: str = "cuda"  # "cuda" or "cpu"
    use_fp16: bool = True
    max_workers: int = 2
    tile_size: int = 512  # for tiled processing of large images

    # Limits
    max_image_size: int = 2048  # max dimension
    max_file_mb: int = 100
    max_batch_size: int = 10

    # Telemetry
    return_latency: bool = True
    return_vram: bool = True
    # Video settings
    video_max_seconds: int = 30
    video_max_res: str = "1080p"  # "720p", "1080p", "2k", "4k"
    target_fps: int = 60

    # Video processing
    chunk_frames: int = 120
    overlap_frames: int = 2
    encoder: str = "libx264"
    crf: int = 23

    # Job system
    jobs_max_workers: int = 2
    jobs_max_queue: int = 10
    jobs_ttl_seconds: int = 3600  # 1 hour

    # Safety settings
    enable_nsfw: bool = True
    enable_face_blur: bool = False
    nsfw_threshold: float = 0.5

    # Export settings
    export_max_rows: int = 1000

    # Presets
    presets_path: Path = Path("configs/presets.json")

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Expand user paths and resolve absolute paths
        self.models_dir = self.ai_cache_root / "models"
        self.cache_dir = self.ai_cache_root / "cache"

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @field_validator("device")
    def validate_device(cls, v):
        if v == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"
        return v

    @property
    def model_registry(self) -> Dict[str, Dict[str, Any]]:
        """Model registry with default file names and paths"""
        return {
            "realesrgan-x4plus": {
                "family": "enhancement",
                "subdir": "esrgan",
                "filename": "RealESRGAN_x4plus.pth",
                "source": "xinntao/Real-ESRGAN",
                "scale": 4,
            },
            "realesrgan-x2plus": {
                "family": "enhancement",
                "subdir": "esrgan",
                "filename": "RealESRGAN_x2plus.pth",
                "source": "xinntao/Real-ESRGAN",
                "scale": 2,
            },
            "gfpgan": {
                "family": "enhancement",
                "subdir": "gfpgan",
                "filename": "GFPGANv1.4.pth",
                "source": "TencentARC/GFPGAN",
                "strength": 0.5,
            },
            "codeformer": {
                "family": "enhancement",
                "subdir": "codeformer",
                "filename": "codeformer.pth",
                "source": "sczhou/CodeFormer",
                "strength": 0.5,
            },
        }

    @property
    def paths(self) -> Dict[str, Path]:
        """Get all important paths for the application"""
        return {
            "cache_root": self.ai_cache_root,
            "models": self.models_dir,
            "enhancement": self.models_dir / "enhancement",
            "video": self.models_dir / "video",
            "outputs": self.ai_cache_root / "outputs" / "restorai",
            "temp": self.ai_cache_root / "temp",
        }

    @property
    def resolution_limits(self) -> Dict[str, Tuple[int, int]]:
        """Maximum resolution limits for video processing"""
        return {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "2k": (2560, 1440),
            "4k": (3840, 2160),
        }


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance"""
    return Settings()
