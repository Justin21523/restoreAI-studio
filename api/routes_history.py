from fastapi import APIRouter, Query, HTTPException
from typing import List, Dict, Any, Optional

from datetime import datetime, timedelta
import json
from pathlib import Path

from utils.artifacts_index import artifact_index, SearchFilters, SortField, SortOrder
from utils.artifacts import artifact_manager
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()


@router.get("/history")
async def get_history(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    task_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    tags: Optional[str] = Query(None),
    nsfw_blocked: Optional[bool] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
    min_latency: Optional[float] = Query(None),
    max_latency: Optional[float] = Query(None),
    sort_by: str = Query("timestamp"),
    sort_order: str = Query("desc"),
):
    """
    Get paginated history of processing artifacts
    """
    try:
        # Parse filters
        filters = SearchFilters()

        if task_type:
            filters.task_types = [task_type]
        if status:
            filters.statuses = [status]
        if model:
            filters.models = [model]
        if tags:
            filters.tags = [tag.strip() for tag in tags.split(",")]

        filters.nsfw_blocked = nsfw_blocked

        # Parse date range
        if date_from:
            filters.date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        if date_to:
            filters.date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

        filters.min_latency = min_latency
        filters.max_latency = max_latency

        # Parse sort parameters
        try:
            sort_field = SortField(sort_by)
        except ValueError:
            sort_field = SortField.TIMESTAMP

        try:
            sort_order_enum = SortOrder(sort_order)
        except ValueError:
            sort_order_enum = SortOrder.DESC

        # Search artifacts
        artifacts, total_count = artifact_index.search_artifacts(
            filters=filters,
            sort_field=sort_field,
            sort_order=sort_order_enum,
            page=page,
            page_size=page_size,
        )

        # Convert to response format
        artifacts_response = []
        for artifact in artifacts:
            artifact_data = {
                "artifact_id": artifact.artifact_id,
                "run_id": artifact.run_id,
                "task_type": artifact.task_type,
                "status": artifact.status.value,
                "timestamp": artifact.timestamp,
                "parameters": artifact.parameters,
                "preset_used": artifact.preset_used,
                "safety": {
                    "nsfw_detected": artifact.safety.nsfw_detected,
                    "face_blur_applied": artifact.safety.face_blur_applied,
                    "faces_detected": artifact.safety.faces_detected,
                    "passed_safety": artifact.safety.passed_safety,
                },
                "metrics": {
                    "latency_ms": artifact.metrics.latency_ms,
                    "vram_peak_gb": artifact.metrics.vram_peak_gb,
                },
                "files": {
                    "original_path": artifact.files.original_path,
                    "processed_path": artifact.files.processed_path,
                    "original_size": artifact.files.original_size,
                    "processed_size": artifact.files.processed_size,
                    "width": artifact.files.width,
                    "height": artifact.files.height,
                },
                "tags": artifact.tags,
            }

            # Generate thumbnail URL if enabled
            if settings.enable_thumbnails:
                thumbnail_path = artifact_index.generate_thumbnail(artifact)
                if thumbnail_path:
                    artifact_data["thumbnail_url"] = (
                        f"/api/v1/thumbnails/{thumbnail_path.name}"
                    )

            artifacts_response.append(artifact_data)

        return {
            "artifacts": artifacts_response,
            "pagination": {
                "page": page,
                "page_size": page_size,
                "total": total_count,
                "total_pages": (total_count + page_size - 1) // page_size,
            },
        }

    except Exception as e:
        logger.error(f"History query failed: {str(e)}")
        raise HTTPException(500, f"History query failed: {str(e)}")


@router.get("/history/stats")
async def get_history_stats(
    task_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None),
):
    """
    Get statistics about processing history
    """
    try:
        # Parse filters
        filters = SearchFilters()

        if task_type:
            filters.task_types = [task_type]
        if status:
            filters.statuses = [status]
        if model:
            filters.models = [model]

        # Parse date range
        if date_from:
            filters.date_from = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        if date_to:
            filters.date_to = datetime.fromisoformat(date_to.replace("Z", "+00:00"))

        # Get statistics
        stats = artifact_index.get_artifact_stats(filters)

        return {
            "statistics": stats,
            "filters_applied": {
                "task_type": task_type,
                "status": status,
                "model": model,
                "date_from": date_from,
                "date_to": date_to,
            },
        }

    except Exception as e:
        logger.error(f"History stats query failed: {str(e)}")
        raise HTTPException(500, f"History stats query failed: {str(e)}")


@router.get("/artifacts/{run_id}/{artifact_id}")
async def get_artifact_detail(run_id: str, artifact_id: str):
    """
    Get detailed artifact information
    """
    try:
        artifact = artifact_manager.load_artifact(run_id, artifact_id)
        if not artifact:
            raise HTTPException(404, "Artifact not found")

        # Convert to response format
        artifact_data = {
            "artifact_id": artifact.artifact_id,
            "run_id": artifact.run_id,
            "task_type": artifact.task_type,
            "status": artifact.status.value,
            "timestamp": artifact.timestamp,
            "parameters": artifact.parameters,
            "preset_used": artifact.preset_used,
            "safety": {
                "nsfw_checked": artifact.safety.nsfw_checked,
                "nsfw_detected": artifact.safety.nsfw_detected,
                "nsfw_confidence": artifact.safety.nsfw_confidence,
                "faces_detected": artifact.safety.faces_detected,
                "face_blur_applied": artifact.safety.face_blur_applied,
                "passed_safety": artifact.safety.passed_safety,
                "safety_error": artifact.safety.safety_error,
            },
            "metrics": {
                "latency_ms": artifact.metrics.latency_ms,
                "vram_peak_gb": artifact.metrics.vram_peak_gb,
                "processing_steps": artifact.metrics.processing_steps,
                "step_durations": artifact.metrics.step_durations,
            },
            "files": {
                "original_path": artifact.files.original_path,
                "processed_path": artifact.files.processed_path,
                "original_size": artifact.files.original_size,
                "processed_size": artifact.files.processed_size,
                "width": artifact.files.width,
                "height": artifact.files.height,
                "duration": artifact.files.duration,
                "fps": artifact.files.fps,
            },
            "tags": artifact.tags,
            "user_metadata": artifact.user_metadata,
        }

        return artifact_data

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Artifact detail query failed: {str(e)}")
        raise HTTPException(500, f"Artifact detail query failed: {str(e)}")


@router.get("/artifacts/stats")
async def get_artifacts_stats():
    """
    Get statistics about artifacts
    """
    try:
        artifacts = artifact_manager.list_artifacts(
            limit=10000
        )  # Increased limit for stats

        # Basic counts
        total_artifacts = len(artifacts)

        # Count by task type
        task_type_counts = {}
        status_counts = {}
        model_counts = {}

        for artifact in artifacts:
            # Task type
            task_type_counts[artifact.task_type] = (
                task_type_counts.get(artifact.task_type, 0) + 1
            )

            # Status
            status_counts[artifact.status.value] = (
                status_counts.get(artifact.status.value, 0) + 1
            )

            # Model (from parameters)
            model = artifact.parameters.get("model", "unknown")
            model_counts[model] = model_counts.get(model, 0) + 1

        # Date range
        timestamps = [artifact.timestamp for artifact in artifacts]
        oldest = min(timestamps) if timestamps else 0
        newest = max(timestamps) if timestamps else 0

        return {
            "total_artifacts": total_artifacts,
            "task_type_counts": task_type_counts,
            "status_counts": status_counts,
            "model_counts": model_counts,
            "date_range": {"oldest": oldest, "newest": newest},
        }

    except Exception as e:
        logger.error(f"Artifacts stats query failed: {str(e)}")
        raise HTTPException(500, f"Artifacts stats query failed: {str(e)}")


@router.get("/thumbnails/{thumbnail_name}")
async def get_thumbnail(thumbnail_name: str):
    """
    Serve generated thumbnails
    """
    try:
        thumbnails_dir = settings.paths["outputs"].parent / "thumbnails"
        thumbnail_path = thumbnails_dir / thumbnail_name

        if not thumbnail_path.exists():
            raise HTTPException(404, "Thumbnail not found")

        from fastapi.responses import FileResponse

        return FileResponse(
            thumbnail_path, media_type="image/jpeg", filename=thumbnail_name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Thumbnail serving failed: {str(e)}")
        raise HTTPException(500, f"Thumbnail serving failed: {str(e)}")


@router.get("/outputs/{path:path}")
async def serve_output_file(path: str):
    """
    Serve output files from the shared cache
    """
    try:
        # Security check - ensure path doesn't traverse directories
        if ".." in path or path.startswith("/"):
            raise HTTPException(400, "Invalid file path")

        output_path = settings.paths["outputs"] / path

        if not output_path.exists():
            raise HTTPException(404, "File not found")

        # Determine media type
        media_type = "application/octet-stream"
        if output_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            media_type = "image/jpeg"
        elif output_path.suffix.lower() in [".mp4", ".avi", ".mov"]:
            media_type = "video/mp4"
        elif output_path.suffix.lower() == ".json":
            media_type = "application/json"

        from fastapi.responses import FileResponse

        return FileResponse(
            output_path, media_type=media_type, filename=output_path.name
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File serving failed: {str(e)}")
        raise HTTPException(500, f"File serving failed: {str(e)}")


@router.get("/artifacts/{run_id}/{artifact_id}/download")
async def download_artifact_file(
    run_id: str, artifact_id: str, file_type: str = "processed"
):
    """
    Generate download URL for artifact file
    """
    try:
        artifact = artifact_manager.load_artifact(run_id, artifact_id)
        if not artifact:
            raise HTTPException(404, "Artifact not found")

        if file_type == "original":
            file_path = Path(artifact.files.original_path)
        else:
            file_path = Path(artifact.files.processed_path)

        if not file_path.exists():
            raise HTTPException(404, "File not found")

        # In a production environment, you might want to generate a signed URL here
        # For now, we return a relative path that the frontend can use
        # The frontend should use the /outputs/ endpoint to download files

        return {
            "file_path": str(file_path),
            "file_type": file_type,
            "artifact_id": artifact_id,
            "run_id": run_id,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download URL generation failed: {str(e)}")
        raise HTTPException(500, f"Download URL generation failed: {str(e)}")
