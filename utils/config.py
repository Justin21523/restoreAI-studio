# utils/config.py
"""
Configuration management for RestorAI MVP.
Simple, lightweight configuration with centralized AI warehouse support.

Why: Centralized AI warehouse enables sharing models across multiple projects,
reducing disk usage and improving model management efficiency.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional


def load_env(env_file: Path) -> Dict[str, str]:
    """Load environment variables from .env file."""
    env_vars = {}

    if not env_file.exists():
        return env_vars

    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                env_vars[key] = value
                os.environ[key] = value

    return env_vars


class Config:
    """Simple configuration class with centralized AI warehouse support."""

    def __init__(self):
        # Load .env file if exists
        env_file = Path(".env")
        if env_file.exists():
            load_env(env_file)

    def _setup_paths(self):
        """Setup centralized AI warehouse paths."""
        # Get AI warehouse root (defaults to ~/ai-warehouse)
        warehouse_root = os.getenv("AI_WAREHOUSE_ROOT")
        if not warehouse_root:
            warehouse_root = str(Path.home() / "ai-warehouse")

        warehouse_path = Path(warehouse_root)

        # Setup warehouse structure
        self._paths = {
            "warehouse": warehouse_path,
            "models": warehouse_path / "models",
            "cache": warehouse_path / "cache",
            "temp": warehouse_path / "temp",
        }

        # Allow override via environment variables
        if models_dir := os.getenv("MODELS_DIR"):
            self._paths["models"] = Path(models_dir)

        if cache_dir := os.getenv("CACHE_DIR"):
            self._paths["cache"] = Path(cache_dir)

        # Project-specific paths (remain in project directory)
        project_root = Path.cwd()
        self._paths.update(
            {
                "project": project_root,
                "input": project_root / "data" / "input",
                "output": project_root / "data" / "output",
                "project_temp": project_root
                / "data"
                / "temp",  # For project-specific temp files
            }
        )

    @property
    def PATHS(self) -> Dict[str, Path]:
        """Get all configured paths."""
        return self._paths.copy()

    # === Device Configuration ===
    @property
    def device(self) -> str:
        """Processing device (cuda/cpu)."""
        return os.getenv("DEVICE", "cuda" if self._is_cuda_available() else "cpu")

    @property
    def use_fp16(self) -> bool:
        """Use half precision for memory efficiency."""
        return os.getenv("USE_FP16", "true").lower() == "true"

    # === Legacy path properties (for backward compatibility) ===
    @property
    def model_dir(self) -> Path:
        """Directory containing AI models (legacy - use PATHS['models'])."""
        return self._paths["models"]

    @property
    def input_dir(self) -> Path:
        """Input files directory."""
        return self._paths["input"]

    @property
    def output_dir(self) -> Path:
        """Output files directory."""
        return self._paths["output"]

    @property
    def temp_dir(self) -> Path:
        """Temporary files directory (project-specific)."""
        return self._paths["project_temp"]

    # === Processing Configuration ===
    @property
    def max_image_size(self) -> int:
        """Maximum image size for processing."""
        return int(os.getenv("MAX_IMAGE_SIZE", "2048"))

    @property
    def tile_size(self) -> int:
        """Tile size for memory-efficient processing."""
        return int(os.getenv("TILE_SIZE", "512"))

    @property
    def max_batch_size(self) -> int:
        """Maximum batch size for processing."""
        return int(os.getenv("MAX_BATCH_SIZE", "4"))

    # === API Configuration ===
    @property
    def api_host(self) -> str:
        """API server host."""
        return os.getenv("API_HOST", "0.0.0.0")

    @property
    def api_port(self) -> int:
        """API server port."""
        return int(os.getenv("API_PORT", "8000"))

    @property
    def api_title(self) -> str:
        """API title."""
        return os.getenv("API_TITLE", "RestorAI MVP")

    @property
    def api_description(self) -> str:
        """API description."""
        return os.getenv("API_DESCRIPTION", "Lightweight AI image restoration API")

    # === UI Configuration ===
    @property
    def ui_port(self) -> int:
        """UI server port."""
        return int(os.getenv("UI_PORT", "7860"))

    @property
    def ui_share(self) -> bool:
        """Share UI publicly."""
        return os.getenv("UI_SHARE", "false").lower() == "true"

    # === Model Paths (using centralized warehouse) ===
    @property
    def esrgan_model_path(self) -> Path:
        """Real-ESRGAN model path."""
        return self._paths["models"] / "esrgan" / "RealESRGAN_x4plus.pth"

    @property
    def gfpgan_model_path(self) -> Path:
        """GFPGAN model path."""
        return self._paths["models"] / "gfpgan" / "GFPGANv1.4.pth"

    @property
    def rife_model_path(self) -> Path:
        """RIFE model path."""
        return self._paths["models"] / "rife" / "flownet.pkl"

    @property
    def edvr_model_path(self) -> Path:
        """EDVR model path."""
        return self._paths["models"] / "edvr" / "EDVR_L_x4_SR_Vimeo90K.pth"

    # === Utility Methods ===
    def _is_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def create_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self._paths["models"],
            self._paths["cache"],
            self._paths["temp"],
            self._paths["input"],
            self._paths["output"],
            self._paths["project_temp"],
            # Create model subdirectories
            self._paths["models"] / "esrgan",
            self._paths["models"] / "gfpgan",
            self._paths["models"] / "rife",
            self._paths["models"] / "edvr",
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

        # Create symlink for backward compatibility (if not exists)
        legacy_models_dir = Path("data/models")
        if not legacy_models_dir.exists() and not legacy_models_dir.is_symlink():
            try:
                legacy_models_dir.parent.mkdir(exist_ok=True)
                legacy_models_dir.symlink_to(
                    self._paths["models"], target_is_directory=True
                )
            except (OSError, NotImplementedError):
                # Symlink creation failed (Windows without admin rights, etc.)
                pass

    def validate_paths(self) -> Dict[str, bool]:
        """Validate that required paths exist and are accessible."""
        validation_results = {}

        critical_paths = ["models", "cache"]
        for path_name in critical_paths:
            path = self._paths[path_name]
            try:
                # Check if path exists or can be created
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)

                # Test write permission
                test_file = path / f".write_test_{os.getpid()}"
                try:
                    test_file.write_text("test")
                    test_file.unlink()
                    validation_results[path_name] = True
                except (OSError, PermissionError):
                    validation_results[path_name] = False

            except Exception:
                validation_results[path_name] = False

        return validation_results

    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about available models."""
        models = {
            "esrgan": {
                "name": "Real-ESRGAN",
                "path": self.esrgan_model_path,
                "available": self.esrgan_model_path.exists(),
                "size_mb": (
                    self._get_file_size_mb(self.esrgan_model_path)
                    if self.esrgan_model_path.exists()
                    else 0
                ),
            },
            "gfpgan": {
                "name": "GFPGAN",
                "path": self.gfpgan_model_path,
                "available": self.gfpgan_model_path.exists(),
                "size_mb": (
                    self._get_file_size_mb(self.gfpgan_model_path)
                    if self.gfpgan_model_path.exists()
                    else 0
                ),
            },
            "rife": {
                "name": "RIFE",
                "path": self.rife_model_path,
                "available": self.rife_model_path.exists(),
                "size_mb": (
                    self._get_file_size_mb(self.rife_model_path)
                    if self.rife_model_path.exists()
                    else 0
                ),
            },
            "edvr": {
                "name": "EDVR",
                "path": self.edvr_model_path,
                "available": self.edvr_model_path.exists(),
                "size_mb": (
                    self._get_file_size_mb(self.edvr_model_path)
                    if self.edvr_model_path.exists()
                    else 0
                ),
            },
        }
        return models

    def _get_file_size_mb(self, path: Path) -> float:
        """Get file size in MB."""
        try:
            return path.stat().st_size / (1024 * 1024)
        except (OSError, FileNotFoundError):
            return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "device": self.device,
            "use_fp16": self.use_fp16,
            "paths": {k: str(v) for k, v in self._paths.items()},
            "model_dir": str(self.model_dir),
            "input_dir": str(self.input_dir),
            "output_dir": str(self.output_dir),
            "temp_dir": str(self.temp_dir),
            "max_image_size": self.max_image_size,
            "tile_size": self.tile_size,
            "max_batch_size": self.max_batch_size,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "ui_port": self.ui_port,
            "models": self.get_model_info(),
        }
