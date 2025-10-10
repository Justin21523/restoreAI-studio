import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import time
import tempfile
import logging
from datetime import datetime

from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ArtifactVersion(str, Enum):
    V2 = "v2"


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"
    CANCELLED = "cancelled"


class TaskType(str, Enum):
    IMAGE_SUPER_RESOLUTION = "image_super_resolution"
    IMAGE_FACE_RESTORATION = "image_face_restoration"
    VIDEO_INTERPOLATION = "video_interpolation"
    VIDEO_UPSCALE = "video_upscale"
    BATCH_PROCESSING = "batch_processing"


@dataclass
class SafetyInfo:
    """Safety processing information"""

    nsfw_checked: bool = False
    nsfw_detected: bool = False
    nsfw_confidence: float = 0.0
    faces_detected: int = 0
    face_blur_applied: bool = False
    passed_safety: bool = True
    safety_error: Optional[str] = None
    action_taken: Optional[str] = None  # "none", "blur", "block"


@dataclass
class PerformanceMetrics:
    """Performance metrics for processing"""

    latency_ms: float = 0.0
    vram_peak_gb: float = 0.0
    processing_steps: List[str] = None
    step_durations: Dict[str, float] = None
    input_size_mb: float = 0.0
    output_size_mb: float = 0.0

    def __post_init__(self):
        if self.processing_steps is None:
            self.processing_steps = []
        if self.step_durations is None:
            self.step_durations = {}


@dataclass
class FileMetadata:
    """File metadata and hashes"""

    original_path: str
    original_size: int
    original_hash: str
    processed_path: str
    processed_size: int
    processed_hash: str
    width: int = 0
    height: int = 0
    duration: float = 0.0  # For videos
    fps: float = 0.0  # For videos
    thumbnail_path: Optional[str] = None
    preview_paths: List[str] = None

    def __post_init__(self):
        if self.preview_paths is None:
            self.preview_paths = []


@dataclass
class RoutingInfo:
    """Model routing information"""

    model_key: str
    parameters: Dict[str, Any]
    reason: str
    confidence: float
    policy_used: Optional[Dict[str, Any]] = None


@dataclass
class QualityMetrics:
    """Quality assessment metrics"""

    faces_detected: int = 0
    restored_faces: int = 0
    sharpen_level: float = 0.0
    denoise_level: float = 0.0
    ssim_score: Optional[float] = None
    psnr_score: Optional[float] = None
    artifacts_detected: bool = False


@dataclass
class ArtifactV2:
    """Artifact schema version 2"""

    # Core identification
    version: ArtifactVersion = ArtifactVersion.V2
    artifact_id: str = ""
    run_id: str = ""
    task_type: TaskType = None
    timestamp: float = 0.0

    # Processing information
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    parameters: Dict[str, Any] = None
    preset_used: Optional[str] = None
    routing_info: Optional[RoutingInfo] = None

    # File information
    files: FileMetadata = None

    # Safety information
    safety: SafetyInfo = None

    # Performance metrics
    metrics: PerformanceMetrics = None

    # Quality metrics
    quality: QualityMetrics = None

    # Additional metadata
    tags: List[str] = None
    user_metadata: Dict[str, Any] = None
    integrity_hash: Optional[str] = None

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.files is None:
            self.files = FileMetadata("", 0, "", "", 0, "")
        if self.safety is None:
            self.safety = SafetyInfo()
        if self.metrics is None:
            self.metrics = PerformanceMetrics()
        if self.quality is None:
            self.quality = QualityMetrics()
        if self.tags is None:
            self.tags = []
        if self.user_metadata is None:
            self.user_metadata = {}

        if not self.artifact_id:
            self.artifact_id = f"artifact_{int(time.time())}_{hash(self)}"

        if not self.timestamp:
            self.timestamp = time.time()

        # Compute integrity hash
        self.integrity_hash = self._compute_integrity_hash()

    def _compute_integrity_hash(self) -> str:
        """Compute integrity hash for the artifact"""
        import hashlib

        # Create a stable representation for hashing
        data = {
            "artifact_id": self.artifact_id,
            "run_id": self.run_id,
            "task_type": self.task_type.value if self.task_type else None,
            "timestamp": self.timestamp,
            "files": {
                "original_hash": self.files.original_hash,
                "processed_hash": self.files.processed_hash,
            },
            "parameters": json.dumps(self.parameters, sort_keys=True),
        }

        data_str = json.dumps(data, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(data_str.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ArtifactV2":
        """Create ArtifactV2 from dictionary"""
        # Handle nested dataclasses
        if "files" in data and isinstance(data["files"], dict):
            data["files"] = FileMetadata(**data["files"])
        if "safety" in data and isinstance(data["safety"], dict):
            data["safety"] = SafetyInfo(**data["safety"])
        if "metrics" in data and isinstance(data["metrics"], dict):
            data["metrics"] = PerformanceMetrics(**data["metrics"])
        if "quality" in data and isinstance(data["quality"], dict):
            data["quality"] = QualityMetrics(**data["quality"])
        if "routing_info" in data and isinstance(data["routing_info"], dict):
            data["routing_info"] = RoutingInfo(**data["routing_info"])

        return cls(**data)
