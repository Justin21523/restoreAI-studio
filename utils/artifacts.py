# utils/artifacts.py

import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
import tempfile
import logging
from enum import Enum
import hashlib

from api.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class ArtifactVersion(str, Enum):
    V2 = "v2"


class ProcessingStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    BLOCKED = "blocked"  # Blocked by safety checks


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


@dataclass
class PerformanceMetrics:
    """Performance metrics for processing"""

    latency_ms: float = 0.0
    vram_peak_gb: float = 0.0
    processing_steps: List[str] = None  # type: ignore
    step_durations: Dict[str, float] = None  # type: ignore

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


@dataclass
class RoutingInfo:
    """Model routing information"""

    model_key: str
    parameters: Dict[str, Any]
    reason: str
    confidence: float


@dataclass
class ArtifactV2:
    """Artifact schema version 2"""

    # Core identification
    version: ArtifactVersion = ArtifactVersion.V2
    artifact_id: str = ""
    run_id: str = ""
    task_type: TaskType = None  # type: ignore
    timestamp: float = 0.0

    # Processing information
    status: ProcessingStatus = ProcessingStatus.SUCCESS
    parameters: Dict[str, Any] = None  # type: ignore
    preset_used: Optional[str] = None
    routing_info: Optional[RoutingInfo] = None

    # File information
    files: FileMetadata = None  # type: ignore

    # Safety information
    safety: SafetyInfo = None  # type: ignore

    # Performance metrics
    metrics: PerformanceMetrics = None  # type: ignore

    # Additional metadata
    tags: List[str] = None  # type: ignore
    user_metadata: Dict[str, Any] = None  # type: ignore

    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.files is None:
            self.files = FileMetadata("", 0, "", "", 0, "")
        if self.safety is None:
            self.safety = SafetyInfo()
        if self.metrics is None:
            self.metrics = PerformanceMetrics()
        if self.tags is None:
            self.tags = []
        if self.user_metadata is None:
            self.user_metadata = {}

        if not self.artifact_id:
            self.artifact_id = f"artifact_{int(time.time())}_{hash(self)}"

        if not self.timestamp:
            self.timestamp = time.time()

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
        if "routing_info" in data and isinstance(data["routing_info"], dict):
            data["routing_info"] = RoutingInfo(**data["routing_info"])
        return cls(**data)


class ArtifactWriter:
    """Write and read artifact files in v2 schema"""

    def __init__(self, warehouse_root: Path):
        self.warehouse_root = warehouse_root
        self.artifacts_dir = warehouse_root / "outputs" / "restorai" / "artifacts"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def write_artifact(
        self, artifact: ArtifactV2, ensure_integrity: bool = True
    ) -> Path:
        """Write artifact atomically with integrity check"""
        run_dir = self.artifacts_dir / artifact.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary file for atomic write
        with tempfile.NamedTemporaryFile(
            mode="w", dir=run_dir, delete=False, suffix=".json"
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            artifact_dict = artifact.to_dict()
            json.dump(
                artifact_dict, tmp_file, indent=2, ensure_ascii=False, default=str
            )

        # Final artifact path
        artifact_path = run_dir / f"{artifact.artifact_id}.json"

        # Atomic move
        tmp_path.rename(artifact_path)

        if ensure_integrity:
            self._verify_artifact_integrity(artifact_path)

        logger.info(f"Artifact written: {artifact_path}")
        return artifact_path

    def _verify_artifact_integrity(self, artifact_path: Path):
        """Verify artifact file integrity"""
        try:
            with open(artifact_path, "r") as f:
                data = json.load(f)
            # Basic structure check
            required_fields = ["version", "artifact_id", "run_id", "task_type"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")
            logger.debug(f"Artifact integrity verified: {artifact_path}")
        except Exception as e:
            logger.error(f"Artifact integrity check failed: {e}")
            # Remove corrupted artifact
            artifact_path.unlink()
            raise

    def read_artifact(self, run_id: str, filename: str) -> Dict[str, Any]:
        """Read artifact from file"""
        artifact_path = self.artifacts_dir / run_id / filename
        with open(artifact_path, "r") as f:
            return json.load(f)

    def list_artifacts(self, run_id: str) -> list:
        """List all artifacts for a run_id"""
        artifact_dir = self.artifacts_dir / run_id
        if not artifact_dir.exists():
            return []
        return [f.name for f in artifact_dir.glob("*.json")]


class ArtifactReader:
    """Read artifacts with integrity validation"""

    def __init__(self, warehouse_root: Path):
        self.warehouse_root = warehouse_root
        self.artifacts_dir = warehouse_root / "outputs" / "restorai" / "artifacts"

    def read_artifact(self, run_id: str, artifact_id: str) -> Optional[ArtifactV2]:
        """Read artifact by run_id and artifact_id"""
        artifact_path = self.artifacts_dir / run_id / f"{artifact_id}.json"
        if not artifact_path.exists():
            return None

        try:
            with open(artifact_path, "r") as f:
                data = json.load(f)
            artifact = ArtifactV2.from_dict(data)
            return artifact
        except Exception as e:
            logger.error(f"Failed to read artifact {artifact_path}: {e}")
            return None

    def list_artifacts(self, run_id: Optional[str] = None) -> List[ArtifactV2]:
        """List artifacts, optionally filtered by run_id"""
        artifacts = []

        if run_id:
            run_dir = self.artifacts_dir / run_id
            if run_dir.exists():
                artifact_files = run_dir.glob("*.json")
            else:
                artifact_files = []
        else:
            artifact_files = self.artifacts_dir.rglob("*.json")

        for artifact_file in artifact_files:
            try:
                artifact_id = artifact_file.stem
                run_id = artifact_file.parent.name
                artifact = self.read_artifact(run_id, artifact_id)
                if artifact:
                    artifacts.append(artifact)
            except Exception as e:
                logger.debug(f"Skipping invalid artifact {artifact_file}: {e}")

        return artifacts


class ArtifactManager:
    """Manager for artifact operations"""

    def __init__(self, warehouse_root: Path):
        self.writer = ArtifactWriter(warehouse_root)
        self.reader = ArtifactReader(warehouse_root)
        self.warehouse_root = warehouse_root

    def create_artifact(self, artifact: ArtifactV2) -> Path:
        """Create a new artifact"""
        return self.writer.write_artifact(artifact)

    def get_artifact(self, run_id: str, artifact_id: str) -> Optional[ArtifactV2]:
        """Get artifact by ID"""
        return self.reader.read_artifact(run_id, artifact_id)

    def list_artifacts(self, run_id: Optional[str] = None) -> List[ArtifactV2]:
        """List artifacts"""
        return self.reader.list_artifacts(run_id)

    def verify_artifact_files(self, artifact: ArtifactV2) -> Tuple[bool, List[str]]:
        """Verify that files referenced in artifact exist and match hashes"""
        issues = []

        # Check original file
        original_path = Path(artifact.files.original_path)
        if not original_path.exists():
            issues.append(f"Original file not found: {original_path}")
        else:
            actual_hash = self._compute_file_hash(original_path)
            if actual_hash != artifact.files.original_hash:
                issues.append(f"Original file hash mismatch: {original_path}")

        # Check processed file
        processed_path = Path(artifact.files.processed_path)
        if not processed_path.exists():
            issues.append(f"Processed file not found: {processed_path}")
        else:
            actual_hash = self._compute_file_hash(processed_path)
            if actual_hash != artifact.files.processed_hash:
                issues.append(f"Processed file hash mismatch: {processed_path}")

        return len(issues) == 0, issues

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()


def create_artifact_v2(
    run_id: str,
    pipeline: str,
    input_file: Path,
    output_file: Path,
    parameters: Dict[str, Any],
    safety: Dict[str, Any],
    metrics: Dict[str, Any],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a v2 artifact dictionary"""
    return {
        "version": "v2",
        "run_id": run_id,
        "pipeline": pipeline,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "parameters": parameters,
        "safety": safety,
        "metrics": metrics,
        "metadata": metadata,
        "timestamps": {
            "created": time.time(),
            "started": metrics.get("start_time", time.time()),
            "completed": time.time(),
        },
    }


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of file"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()
