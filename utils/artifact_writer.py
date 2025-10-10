import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Optional
import tempfile
import logging

from utils.artifacts_v2 import ArtifactV2
from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class ArtifactWriter:
    """Write artifacts atomically with integrity checks"""

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

            # Verify integrity hash if present
            if "integrity_hash" in data:
                artifact = ArtifactV2.from_dict(data)
                if artifact.integrity_hash != data["integrity_hash"]:
                    raise ValueError("Integrity hash mismatch")

            logger.debug(f"Artifact integrity verified: {artifact_path}")

        except Exception as e:
            logger.error(f"Artifact integrity check failed: {e}")
            # Remove corrupted artifact
            artifact_path.unlink()
            raise


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

        # Check thumbnail if exists
        if artifact.files.thumbnail_path:
            thumb_path = Path(artifact.files.thumbnail_path)
            if not thumb_path.exists():
                issues.append(f"Thumbnail file not found: {thumb_path}")

        return len(issues) == 0, issues

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
