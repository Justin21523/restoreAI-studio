import json
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging
from enum import Enum

from api.config import get_settings
from utils.logger import logger

settings = get_settings()


class TaskType(str, Enum):
    SUPER_RESOLUTION = "super_resolution"
    FACE_RESTORATION = "face_restoration"
    VIDEO_UPSCALE = "video_upscale"
    VIDEO_INTERPOLATE = "video_interpolate"


@dataclass
class Preset:
    """Preset configuration for a specific task"""
    name: str
    task_type: TaskType
    parameters: Dict[str, Any]
    description: str = ""
    tags: list = []

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class PresetManager:
    """Manage server-side presets for different tasks"""

    def __init__(self, presets_path: Optional[Path] = None):
        self.presets_path = presets_path or Path("configs/presets.json")
        self.presets: Dict[str, Preset] = {}
        self.load_presets()

    def load_presets(self):
        """Load presets from JSON file"""
        if not self.presets_path.exists():
            self._create_default_presets()
            return

        try:
            with open(self.presets_path, 'r', encoding='utf-8') as f:
                presets_data = json.load(f)

            for preset_data in presets_data.get("presets", []):
                preset = Preset(
                    name=preset_data["name"],
                    task_type=TaskType(preset_data["task_type"]),
                    parameters=preset_data["parameters"],
                    description=preset_data.get("description", ""),
                    tags=preset_data.get("tags", [])
                )
                self.presets[preset.name] = preset

            logger.info(f"Loaded {len(self.presets)} presets from {self.presets_path}")

        except Exception as e:
            logger.error(f"Failed to load presets: {e}")
            self._create_default_presets()

    def _create_default_presets(self):
        """Create default presets"""
        default_presets = [
            Preset(
                name="sr_balanced",
                task_type=TaskType.SUPER_RESOLUTION,
                parameters={"scale": 2, "model": "realesrgan-x2plus"},
                description="Balanced super-resolution with 2x scaling",
                tags=["balanced", "fast"]
            ),
            Preset(
                name="sr_high_quality",
                task_type=TaskType.SUPER_RESOLUTION,
                parameters={"scale": 4, "model": "realesrgan-x4plus"},
                description="High-quality super-resolution with 4x scaling",
                tags=["quality", "slow"]
            ),
            Preset(
                name="face_natural",
                task_type=TaskType.FACE_RESTORATION,
                parameters={"method": "gfpgan", "strength": 0.3},
                description="Natural face restoration with subtle enhancement",
                tags=["natural", "subtle"]
            ),
            Preset(
                name="face_enhanced",
                task_type=TaskType.FACE_RESTORATION,
                parameters={"method": "codeformer", "strength": 0.7},
                description="Enhanced face restoration with strong improvement",
                tags=["enhanced", "strong"]
            ),
            Preset(
                name="video_smooth",
                task_type=TaskType.VIDEO_INTERPOLATE,
                parameters={"target_fps": 60, "model": "rife-v4.6"},
                description="Smooth video interpolation to 60 FPS",
                tags=["smooth", "standard"]
            ),
            Preset(
                name="video_ultra_smooth",
                task_type=TaskType.VIDEO_INTERPOLATE,
                parameters={"target_fps": 120, "model": "rife-v4.6"},
                description="Ultra-smooth video interpolation to 120 FPS",
                tags["ultra-smooth", "high-fps"]
            )
        ]

        for preset in default_presets:
            self.presets[preset.name] = preset

        self.save_presets()
        logger.info("Created default presets")

    def save_presets(self):
        """Save presets to JSON file"""
        try:
            self.presets_path.parent.mkdir(parents=True, exist_ok=True)

            presets_data = {
                "presets": [
                    {
                        "name": preset.name,
                        "task_type": preset.task_type.value,
                        "parameters": preset.parameters,
                        "description": preset.description,
                        "tags": preset.tags
                    }
                    for preset in self.presets.values()
                ]
            }

            with open(self.presets_path, 'w', encoding='utf-8') as f:
                json.dump(presets_data, f, indent=2, ensure_ascii=False)

            logger.info(f"Saved {len(self.presets)} presets to {self.presets_path}")

        except Exception as e:
            logger.error(f"Failed to save presets: {e}")

    def get_preset(self, name: str) -> Optional[Preset]:
        """Get preset by name"""
        return self.presets.get(name)

    def get_presets_by_task(self, task_type: TaskType) -> list[Preset]:
        """Get all presets for a specific task type"""
        return [preset for preset in self.presets.values()
                if preset.task_type == task_type]

    def apply_preset(self, user_params: Dict[str, Any], preset_name: str) -> Dict[str, Any]:
        """
        Apply preset parameters to user parameters
        Preset values are overridden by user values
        """
        preset = self.get_preset(preset_name)
        if not preset:
            logger.warning(f"Preset '{preset_name}' not found")
            return user_params

        # Merge parameters (user values override preset values)
        merged_params = {**preset.parameters, **user_params}
        logger.debug(f"Applied preset '{preset_name}': {merged_params}")

        return merged_params

    def create_preset(self, name: str, task_type: TaskType, parameters: Dict[str, Any],
                     description: str = "", tags: list = None) -> bool:
        """Create a new preset"""
        if name in self.presets:
            logger.warning(f"Preset '{name}' already exists")
            return False

        preset = Preset(
            name=name,
            task_type=task_type,
            parameters=parameters,
            description=description,
            tags=tags or []
        )

        self.presets[name] = preset
        self.save_presets()
        logger.info(f"Created new preset: {name}")
        return True

    def delete_preset(self, name: str) -> bool:
        """Delete a preset"""
        if name not in self.presets:
            logger.warning(f"Preset '{name}' not found")
            return False

        del self.presets[name]
        self.save_presets()
        logger.info(f"Deleted preset: {name}")
        return True


# Global preset manager instance
preset_manager = PresetManager()