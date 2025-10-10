from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import csv
import json
import io
from typing import List, Optional
from datetime import datetime, timedelta

from utils.artifacts import artifact_manager, ProcessingStatus
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()


@router.get("/export/report")
async def export_report(
    format: str = Query("json", regex="^(json|csv)$"),
    task_type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    days: int = Query(30, ge=1, le=365),
    tags: Optional[str] = Query(None),
):
    """
    Export processing report in JSON or CSV format
    """
    try:
        # Parse filters
        status_filter = ProcessingStatus(status) if status else None
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else None

        # Calculate date range
        end_date = datetime.now().timestamp()
        start_date = (datetime.now() - timedelta(days=days)).timestamp()
        date_range = (start_date, end_date)

        # Get artifacts
        artifacts = artifact_manager.search_artifacts(
            tags=tag_list, status=status_filter, date_range=date_range
        )

        # Apply row limit
        artifacts = artifacts[: settings.export_max_rows]

        if format == "json":
            # Return as JSON
            return {
                "export_info": {
                    "format": "json",
                    "generated_at": datetime.now().isoformat(),
                    "total_artifacts": len(artifacts),
                    "filters": {
                        "task_type": task_type,
                        "status": status,
                        "days": days,
                        "tags": tags,
                    },
                },
                "artifacts": [
                    {
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
                        },
                        "tags": artifact.tags,
                    }
                    for artifact in artifacts
                ],
            }
        else:
            # Convert to CSV
            if not artifacts:
                raise HTTPException(404, "No artifacts to export")

            # Flatten artifacts for CSV
            flattened = []
            for artifact in artifacts:
                flat = {
                    "artifact_id": artifact.artifact_id,
                    "run_id": artifact.run_id,
                    "task_type": artifact.task_type,
                    "status": artifact.status.value,
                    "timestamp": datetime.fromtimestamp(artifact.timestamp).isoformat(),
                    "parameters": json.dumps(artifact.parameters),
                    "preset_used": artifact.preset_used or "",
                    "nsfw_detected": artifact.safety.nsfw_detected,
                    "face_blur_applied": artifact.safety.face_blur_applied,
                    "faces_detected": artifact.safety.faces_detected,
                    "latency_ms": artifact.metrics.latency_ms,
                    "vram_peak_gb": artifact.metrics.vram_peak_gb,
                    "original_path": artifact.files.original_path,
                    "processed_path": artifact.files.processed_path,
                    "original_size": artifact.files.original_size,
                    "processed_size": artifact.files.processed_size,
                    "tags": ", ".join(artifact.tags),
                }
                flattened.append(flat)

            # Create CSV in memory
            stream = io.StringIO()
            if flattened:
                writer = csv.DictWriter(stream, fieldnames=flattened[0].keys())
                writer.writeheader()
                writer.writerows(flattened)

            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=restorai_report.csv"
                },
            )
            return response

    except Exception as e:
        logger.error(f"Export failed: {str(e)}")
        raise HTTPException(500, f"Export failed: {str(e)}")


@router.get("/export/artifacts/{run_id}")
async def export_artifacts_by_run(run_id: str, format: str = Query("json")):
    """Export all artifacts for a specific run"""
    try:
        artifacts = artifact_manager.list_artifacts(run_id=run_id)

        if not artifacts:
            raise HTTPException(404, f"No artifacts found for run {run_id}")

        if format == "json":
            return {
                "run_id": run_id,
                "artifacts": [
                    {
                        "artifact_id": artifact.artifact_id,
                        "task_type": artifact.task_type,
                        "status": artifact.status.value,
                        "timestamp": artifact.timestamp,
                        "parameters": artifact.parameters,
                    }
                    for artifact in artifacts
                ],
            }
        else:
            # CSV format
            flattened = []
            for artifact in artifacts:
                flat = {
                    "artifact_id": artifact.artifact_id,
                    "task_type": artifact.task_type,
                    "status": artifact.status.value,
                    "timestamp": datetime.fromtimestamp(artifact.timestamp).isoformat(),
                    "parameters": json.dumps(artifact.parameters),
                    "latency_ms": artifact.metrics.latency_ms,
                }
                flattened.append(flat)

            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=flattened[0].keys())
            writer.writeheader()
            writer.writerows(flattened)

            response = StreamingResponse(
                iter([stream.getvalue()]),
                media_type="text/csv",
                headers={
                    "Content-Disposition": f"attachment; filename=run_{run_id}_artifacts.csv"
                },
            )
            return response

    except Exception as e:
        logger.error(f"Run export failed: {str(e)}")
        raise HTTPException(500, f"Run export failed: {str(e)}")
