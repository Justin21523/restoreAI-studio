import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import logging

from utils.artifacts import ArtifactV2, ArtifactManager
from api.config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)


class SortField(str, Enum):
    TIMESTAMP = "timestamp"
    LATENCY = "latency"
    FILE_SIZE = "file_size"


class SortOrder(str, Enum):
    ASC = "asc"
    DESC = "desc"


@dataclass
class SearchFilters:
    """Search filters for artifacts"""

    task_types: Optional[List[str]] = None
    statuses: Optional[List[str]] = None
    models: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    nsfw_blocked: Optional[bool] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    min_latency: Optional[float] = None
    max_latency: Optional[float] = None


class ArtifactIndex:
    """Index and search artifacts with filtering and pagination"""

    def __init__(self, warehouse_root: Path):
        self.warehouse_root = warehouse_root
        self.index_path = warehouse_root / "outputs" / "restorai" / "index.jsonl"
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_manager = ArtifactManager(warehouse_root)

    def add_artifact(self, artifact: ArtifactV2):
        """Add artifact to index"""
        index_entry = {
            "artifact_id": artifact.artifact_id,
            "run_id": artifact.run_id,
            "task_type": artifact.task_type.value,
            "timestamp": artifact.timestamp,
            "status": artifact.status.value,
            "tags": artifact.tags,
            "parameters": artifact.parameters,
            "files": {
                "original_path": artifact.files.original_path,
                "processed_path": artifact.files.processed_path,
            },
            "safety": {
                "nsfw_detected": artifact.safety.nsfw_detected,
                "face_blur_applied": artifact.safety.face_blur_applied,
            },
            "metrics": {
                "latency_ms": artifact.metrics.latency_ms,
            },
        }

        with open(self.index_path, "a") as f:
            f.write(json.dumps(index_entry) + "\n")

        logger.debug(f"Added artifact to index: {artifact.artifact_id}")

    def query(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Query index with filters"""
        if not self.index_path.exists():
            return []

        results = []
        with open(self.index_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                if self._matches_filters(entry, filters):
                    results.append(entry)

        # Sort by timestamp descending
        results.sort(key=lambda x: x["timestamp"], reverse=True)

        return results[offset : offset + limit]

    def _matches_filters(
        self, entry: Dict[str, Any], filters: Optional[Dict[str, Any]]
    ) -> bool:
        """Check if entry matches all filters"""
        if not filters:
            return True

        for key, value in filters.items():
            if key not in entry:
                return False

            if isinstance(value, (list, tuple)):
                if entry[key] not in value:
                    return False
            else:
                if entry[key] != value:
                    return False

        return True

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        if not self.index_path.exists():
            return {"total_artifacts": 0}

        total_artifacts = 0
        task_type_counts = {}
        status_counts = {}

        with open(self.index_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                total_artifacts += 1

                task_type = entry["task_type"]
                task_type_counts[task_type] = task_type_counts.get(task_type, 0) + 1

                status = entry["status"]
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            "total_artifacts": total_artifacts,
            "task_type_counts": task_type_counts,
            "status_counts": status_counts,
        }

    def search_artifacts(
        self,
        filters: SearchFilters,
        sort_field: SortField = SortField.TIMESTAMP,
        sort_order: SortOrder = SortOrder.DESC,
        page: int = 1,
        page_size: int = 50,
    ) -> Tuple[List[ArtifactV2], int]:
        """Search artifacts with filtering and pagination"""
        # Get all artifacts
        all_artifacts = self.artifact_manager.list_artifacts(limit=10000)

        # Apply filters
        filtered_artifacts = self._apply_filters(all_artifacts, filters)

        # Sort artifacts
        sorted_artifacts = self._sort_artifacts(
            filtered_artifacts, sort_field, sort_order
        )

        # Paginate
        total_count = len(sorted_artifacts)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_artifacts = sorted_artifacts[start_idx:end_idx]

        return paginated_artifacts, total_count

    def _apply_filters(
        self, artifacts: List[ArtifactV2], filters: SearchFilters
    ) -> List[ArtifactV2]:
        """Apply search filters to artifacts"""
        filtered = []

        for artifact in artifacts:
            # Filter by task type
            if filters.task_types and artifact.task_type not in filters.task_types:
                continue

            # Filter by status
            if filters.statuses and artifact.status.value not in filters.statuses:
                continue

            # Filter by model (from parameters)
            if filters.models:
                model = artifact.parameters.get("model")
                if not model or model not in filters.models:
                    continue

            # Filter by tags
            if filters.tags and not any(tag in artifact.tags for tag in filters.tags):
                continue

            # Filter by NSFW blocked
            if filters.nsfw_blocked is not None:
                if artifact.safety.nsfw_detected != filters.nsfw_blocked:
                    continue

            # Filter by date range
            if filters.date_from or filters.date_to:
                artifact_date = datetime.fromtimestamp(artifact.timestamp)
                if filters.date_from and artifact_date < filters.date_from:
                    continue
                if filters.date_to and artifact_date > filters.date_to:
                    continue

            # Filter by latency
            if (
                filters.min_latency
                and artifact.metrics.latency_ms < filters.min_latency
            ):
                continue
            if (
                filters.max_latency
                and artifact.metrics.latency_ms > filters.max_latency
            ):
                continue

            filtered.append(artifact)

        return filtered

    def _sort_artifacts(
        self, artifacts: List[ArtifactV2], sort_field: SortField, sort_order: SortOrder
    ) -> List[ArtifactV2]:
        """Sort artifacts by specified field and order"""
        reverse = sort_order == SortOrder.DESC

        if sort_field == SortField.TIMESTAMP:
            return sorted(artifacts, key=lambda x: x.timestamp, reverse=reverse)
        elif sort_field == SortField.LATENCY:
            return sorted(
                artifacts, key=lambda x: x.metrics.latency_ms, reverse=reverse
            )
        elif sort_field == SortField.FILE_SIZE:
            return sorted(
                artifacts, key=lambda x: x.files.processed_size, reverse=reverse
            )
        else:
            return sorted(artifacts, key=lambda x: x.timestamp, reverse=reverse)

    def get_artifact_stats(
        self, filters: Optional[SearchFilters] = None
    ) -> Dict[str, Any]:
        """Get statistics about artifacts"""
        artifacts = self.artifact_manager.list_artifacts(limit=10000)

        if filters:
            artifacts = self._apply_filters(artifacts, filters)

        stats = {
            "total_count": len(artifacts),
            "task_type_counts": {},
            "status_counts": {},
            "model_counts": {},
            "date_range": {"earliest": None, "latest": None},
            "latency_stats": {"min": float("inf"), "max": 0, "avg": 0},
        }

        if not artifacts:
            return stats

        # Calculate statistics
        total_latency = 0
        timestamps = []

        for artifact in artifacts:
            # Task type counts
            stats["task_type_counts"][artifact.task_type] = (
                stats["task_type_counts"].get(artifact.task_type, 0) + 1
            )

            # Status counts
            stats["status_counts"][artifact.status.value] = (
                stats["status_counts"].get(artifact.status.value, 0) + 1
            )

            # Model counts
            model = artifact.parameters.get("model", "unknown")
            stats["model_counts"][model] = stats["model_counts"].get(model, 0) + 1

            # Latency stats
            latency = artifact.metrics.latency_ms
            stats["latency_stats"]["min"] = min(stats["latency_stats"]["min"], latency)
            stats["latency_stats"]["max"] = max(stats["latency_stats"]["max"], latency)
            total_latency += latency

            # Timestamps for date range
            timestamps.append(artifact.timestamp)

        # Finalize latency stats
        stats["latency_stats"]["avg"] = total_latency / len(artifacts)
        if stats["latency_stats"]["min"] == float("inf"):
            stats["latency_stats"]["min"] = 0

        # Date range
        stats["date_range"]["earliest"] = min(timestamps)
        stats["date_range"]["latest"] = max(timestamps)

        return stats

    def generate_thumbnail(
        self, artifact: ArtifactV2, max_size: int = 512
    ) -> Optional[Path]:
        """Generate thumbnail for artifact"""
        try:
            import cv2
            from pathlib import Path

            # Determine source file
            source_path = Path(artifact.files.processed_path)
            if not source_path.exists():
                source_path = Path(artifact.files.original_path)

            if not source_path.exists():
                return None

            # Create thumbnails directory
            thumbnails_dir = self.artifact_manager.artifacts_dir.parent / "thumbnails"
            thumbnails_dir.mkdir(parents=True, exist_ok=True)

            thumbnail_path = thumbnails_dir / f"{artifact.artifact_id}_thumb.jpg"

            # Check if thumbnail already exists
            if thumbnail_path.exists():
                return thumbnail_path

            # Generate thumbnail
            if source_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                # Image thumbnail
                img = cv2.imread(str(source_path))
                if img is None:
                    return None

                height, width = img.shape[:2]
                if max(height, width) > max_size:
                    # Calculate scaling factor
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)

                    img = cv2.resize(
                        img, (new_width, new_height), interpolation=cv2.INTER_AREA
                    )

                cv2.imwrite(str(thumbnail_path), img, [cv2.IMWRITE_JPEG_QUALITY, 85])

            elif source_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
                # Video thumbnail - extract first frame
                cap = cv2.VideoCapture(str(source_path))
                if not cap.isOpened():
                    return None

                success, frame = cap.read()
                cap.release()

                if success:
                    height, width = frame.shape[:2]
                    if max(height, width) > max_size:
                        scale = max_size / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(
                            frame, (new_width, new_height), interpolation=cv2.INTER_AREA
                        )

                    cv2.imwrite(
                        str(thumbnail_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85]
                    )

            return thumbnail_path if thumbnail_path.exists() else None

        except Exception as e:
            print(f"Thumbnail generation failed: {e}")
            return None


# Global artifact index instance
artifact_index = ArtifactIndex()
