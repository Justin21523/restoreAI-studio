from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Single source of truth for API, worker, storage, and model paths."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    model_root: Path = Path("/mnt/c/ai_models")
    model_manifest: Path = Path(__file__).with_name("model_manifest.yaml")
    model_auto_download: bool = False
    device: Literal["auto", "cuda", "cpu"] = "auto"
    use_fp16: bool = True
    tile_size: int = 512

    storage_root: Path = Path("data/storage")
    artifact_retention_hours: int = 24
    database_url: str = "postgresql+psycopg://restorai:restorai@127.0.0.1:55432/restorai"
    redis_url: str = "redis://127.0.0.1:56379/0"
    queue_name: str = "restorai-gpu"
    queue_enabled: bool = True

    max_image_bytes: int = 25 * 1024 * 1024
    max_image_pixels: int = 16_000_000
    max_output_pixels: int = 150_000_000
    max_video_bytes: int = 2 * 1024 * 1024 * 1024
    max_video_duration_seconds: int = 600
    max_video_width: int = 3840
    max_video_height: int = 2160
    max_video_fps: int = 120
    max_image_batch: int = 50
    max_video_batch: int = 10

    @field_validator("model_root")
    @classmethod
    def require_absolute_model_root(cls, value: Path) -> Path:
        expanded = value.expanduser()
        if not expanded.is_absolute():
            raise ValueError("MODEL_ROOT must be an absolute path")
        return expanded

    def ensure_runtime_directories(self) -> None:
        if not self.model_root.is_dir():
            raise RuntimeError(f"MODEL_ROOT is not accessible: {self.model_root}")
        for name in ("uploads", "work", "artifacts"):
            (self.storage_root / name).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
