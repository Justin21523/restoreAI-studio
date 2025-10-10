import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta
import json

from api.config import get_settings
from utils.artifacts_index import artifact_index
from utils.logging import get_logger

settings = get_settings()
logger = get_logger("cache_admin")


class CacheAdmin:
    """Cache management and governance utilities"""

    def __init__(self):
        self.settings = get_settings()

    def get_cache_size(self) -> Dict[str, Any]:
        """Get cache size breakdown"""
        cache_info = {
            "total_size_bytes": 0,
            "components": {},
            "last_updated": datetime.now().isoformat(),
        }

        # Models cache
        models_size = self._get_directory_size(self.settings.paths["models"])
        cache_info["components"]["models"] = {
            "size_bytes": models_size,
            "size_human": self._format_size(models_size),
        }
        cache_info["total_size_bytes"] += models_size

        # Outputs cache
        outputs_size = self._get_directory_size(self.settings.paths["outputs"])
        cache_info["components"]["outputs"] = {
            "size_bytes": outputs_size,
            "size_human": self._format_size(outputs_size),
        }
        cache_info["total_size_bytes"] += outputs_size

        # Temp cache
        temp_size = self._get_directory_size(self.settings.paths["temp"])
        cache_info["components"]["temp"] = {
            "size_bytes": temp_size,
            "size_human": self._format_size(temp_size),
        }
        cache_info["total_size_bytes"] += temp_size

        cache_info["total_size_human"] = self._format_size(
            cache_info["total_size_bytes"]
        )

        return cache_info

    def _get_directory_size(self, directory: Path) -> int:
        """Calculate directory size in bytes"""
        if not directory.exists():
            return 0

        total_size = 0
        for file_path in directory.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size

        return total_size

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable format"""
        if size_bytes == 0:
            return "0 B"

        units = ["B", "KB", "MB", "GB", "TB"]
        import math

        i = int(math.floor(math.log(size_bytes, 1024)))
        p = math.pow(1024, i)
        s = round(size_bytes / p, 2)
        return f"{s} {units[i]}"

    def prune_outputs(
        self,
        dry_run: bool = True,
        retention_days: int = None,  # type: ignore
        keep_per_task: int = None,  # type: ignore
    ) -> Dict[str, Any]:
        """Prune old output files based on retention policy"""
        if retention_days is None:
            retention_days = settings.retention_days
        if keep_per_task is None:
            keep_per_task = settings.retention_keep_per_task

        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cutoff_timestamp = cutoff_date.timestamp()

        # Get all artifacts
        artifacts = artifact_index.artifact_manager.list_artifacts(limit=10000)

        # Group artifacts by task type
        artifacts_by_task = {}
        for artifact in artifacts:
            task_type = artifact.task_type
            if task_type not in artifacts_by_task:
                artifacts_by_task[task_type] = []
            artifacts_by_task[task_type].append(artifact)

        # Identify artifacts to delete
        to_delete = []
        preserved = []

        for task_type, task_artifacts in artifacts_by_task.items():
            # Sort by timestamp (newest first)
            task_artifacts.sort(key=lambda x: x.timestamp, reverse=True)

            # Keep most recent artifacts per task
            keep_count = min(keep_per_task, len(task_artifacts))
            preserved.extend(task_artifacts[:keep_count])

            # Check remaining artifacts against retention period
            for artifact in task_artifacts[keep_count:]:
                if artifact.timestamp < cutoff_timestamp:
                    to_delete.append(artifact)
                else:
                    preserved.append(artifact)

        # Prepare deletion report
        report = {
            "dry_run": dry_run,
            "retention_days": retention_days,
            "keep_per_task": keep_per_task,
            "total_artifacts": len(artifacts),
            "to_delete": len(to_delete),
            "to_preserve": len(preserved),
            "deleted_files": [],
        }

        if not dry_run and settings.prune_confirm:
            # Actually delete files
            for artifact in to_delete:
                try:
                    # Delete processed file
                    processed_path = Path(artifact.files.processed_path)
                    if processed_path.exists():
                        processed_path.unlink()
                        report["deleted_files"].append(str(processed_path))

                    # Delete original file if it's in output directory
                    original_path = Path(artifact.files.original_path)
                    if original_path.exists() and str(original_path).startswith(
                        str(settings.paths["outputs"])
                    ):
                        original_path.unlink()
                        report["deleted_files"].append(str(original_path))

                    # Delete artifact file
                    artifact_path = (
                        settings.paths["outputs"]
                        / "artifacts"
                        / artifact.run_id
                        / f"{artifact.artifact_id}.json"
                    )
                    if artifact_path.exists():
                        artifact_path.unlink()
                        report["deleted_files"].append(str(artifact_path))

                except Exception as e:
                    logger.error(
                        f"Failed to delete artifact {artifact.artifact_id}: {e}"
                    )

        return report

    def verify_models(self) -> Dict[str, Any]:
        """Verify that required model files exist"""
        model_registry = settings.model_registry
        verification_report = {
            "total_models": len(model_registry),
            "verified": 0,
            "missing": [],
            "corrupted": [],
            "last_checked": datetime.now().isoformat(),
        }

        for model_name, model_info in model_registry.items():
            try:
                family = model_info.get("family", "enhancement")
                subdir = model_info.get("subdir", "")
                filename = model_info.get("filename", "")

                model_path = settings.paths["models"] / family / subdir / filename

                if not model_path.exists():
                    verification_report["missing"].append(
                        {
                            "model": model_name,
                            "expected_path": str(model_path),
                            "source": model_info.get("source", "unknown"),
                        }
                    )
                    continue

                # Basic file integrity check
                file_size = model_path.stat().st_size
                if file_size < 1024:  # Less than 1KB is suspicious
                    verification_report["corrupted"].append(
                        {
                            "model": model_name,
                            "path": str(model_path),
                            "size_bytes": file_size,
                            "issue": "File size too small",
                        }
                    )
                    continue

                verification_report["verified"] += 1

            except Exception as e:
                verification_report["corrupted"].append(
                    {"model": model_name, "error": str(e)}
                )

        return verification_report

    def cleanup_temp_files(self, older_than_hours: int = 24) -> Dict[str, Any]:
        """Clean up temporary files older than specified hours"""
        temp_dir = settings.paths["temp"]
        cutoff_time = datetime.now().timestamp() - (older_than_hours * 3600)

        report = {
            "temp_directory": str(temp_dir),
            "cutoff_hours": older_than_hours,
            "deleted_files": [],
            "deleted_directories": [],
            "total_freed_bytes": 0,
        }

        if not temp_dir.exists():
            return report

        for file_path in temp_dir.rglob("*"):
            try:
                if file_path.is_file():
                    if file_path.stat().st_mtime < cutoff_time:
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        report["deleted_files"].append(str(file_path))
                        report["total_freed_bytes"] += file_size

                elif file_path.is_dir():
                    # Check if directory is empty
                    if not any(file_path.iterdir()):
                        file_path.rmdir()
                        report["deleted_directories"].append(str(file_path))

            except Exception as e:
                logger.error(f"Failed to clean up {file_path}: {e}")

        report["total_freed_human"] = self._format_size(report["total_freed_bytes"])
        return report


# Global cache admin instance
cache_admin = CacheAdmin()
