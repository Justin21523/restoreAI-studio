from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
import asyncio
from pathlib import Path

from api.jobs import job_registry
from api.job_manager import job_manager, Job, JobStatus
from api.events import event_manager, EventType
from utils.artifacts import ArtifactManager
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()
artifact_manager = ArtifactManager(settings.paths["cache_root"])


@router.post("/jobs")
async def create_job(
    background_tasks: BackgroundTasks, task_type: str, parameters: dict
):
    """Create a new job"""
    try:
        # Validate task type
        valid_tasks = [
            "super_resolution",
            "face_restoration",
            "video_interpolation",
            "video_upscale",
        ]
        if task_type not in valid_tasks:
            raise HTTPException(
                400, f"Invalid task type. Must be one of: {valid_tasks}"
            )

        # Create job
        job_id = job_registry.create_job(task_type, parameters)

        # Submit job for processing (in background)
        background_tasks.add_task(
            submit_job_for_processing, job_id, task_type, parameters
        )

        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Job created and queued for processing",
        }

    except Exception as e:
        logger.error(f"Job creation failed: {str(e)}")
        raise HTTPException(500, f"Job creation failed: {str(e)}")


@router.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get job status and result"""
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    response = {
        "job_id": job.id,
        "task_type": job.task_type,
        "status": job.status.value,
        "progress": job.progress,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "completed_at": job.completed_at,
    }

    if job.result:
        response["result"] = job.result

    if job.error:
        response["error"] = job.error

    if job.artifact_id:
        response["artifact_id"] = job.artifact_id
        # Include artifact details if available
        artifact = artifact_manager.get_artifact(job_id, job.artifact_id)
        if artifact:
            response["artifact"] = artifact.to_dict()

    return response


@router.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str):
    """Stream job events via Server-Sent Events"""
    # Verify job exists
    job = job_registry.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")

    async def event_generator():
        try:
            async for message in event_manager.generate_events(job_id):
                yield message
        except Exception as e:
            logger.error(f"Event stream error for job {job_id}: {e}")

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@router.delete("/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel a running job"""
    success = job_registry.cancel_job(job_id)
    if not success:
        raise HTTPException(404, "Job not found or cannot be cancelled")

    # Notify subscribers
    await event_manager.publish(
        job_id,
        EventType.COMPLETED,
        {"status": "cancelled", "message": "Job was cancelled"},
    )

    return {"message": "Job cancelled successfully"}


@router.get("/jobs")
async def list_jobs(status: Optional[str] = None, limit: int = 50):
    """List jobs with optional status filter"""
    try:
        status_enum = JobStatus(status) if status else None
        jobs = job_registry.list_jobs(status=status_enum, limit=limit)

        return {
            "jobs": [
                {
                    "job_id": job.id,
                    "task_type": job.task_type,
                    "status": job.status.value,
                    "progress": job.progress,
                    "created_at": job.created_at,
                    "started_at": job.started_at,
                    "completed_at": job.completed_at,
                }
                for job in jobs
            ],
            "total": len(jobs),
        }

    except ValueError:
        raise HTTPException(400, "Invalid status value")


async def submit_job_for_processing(job_id: str, task_type: str, parameters: dict):
    """Submit job for processing in background"""
    try:
        # Import processing functions based on task type
        if task_type == "super_resolution":
            from core.pipeline_image import ImagePipeline

            pipeline = ImagePipeline()
            task_fn = lambda params, progress, emit: pipeline.process_image(
                Path(params["input_path"]),
                params["config"],
                progress_callback=progress,
                event_callback=emit,
            )
        elif task_type == "face_restoration":
            from core.pipeline_image import ImagePipeline

            pipeline = ImagePipeline()
            task_fn = lambda params, progress, emit: pipeline.process_image(
                Path(params["input_path"]),
                params["config"],
                progress_callback=progress,
                event_callback=emit,
            )
        elif task_type == "video_interpolation":
            from core.video_pipeline import VideoPipeline

            pipeline = VideoPipeline()
            task_fn = lambda params, progress, emit: pipeline.interpolate_video(
                Path(params["input_path"]),
                Path(params["output_path"]),
                params["target_fps"],
                progress_callback=progress,
                event_callback=emit,
            )
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

        # Define progress callback that publishes events
        async def progress_callback(job_id: str, progress: float, message: str = ""):
            await event_manager.publish(
                job_id, EventType.PROGRESS, {"progress": progress, "message": message}
            )

        def sync_progress_callback(progress: float, message: str = ""):
            # Run async function in event loop
            asyncio.create_task(progress_callback(job_id, progress, message))

        # Define event callback
        async def event_callback(job_id: str, event_type: str, data: dict):
            await event_manager.publish(job_id, EventType(event_type), data)

        def sync_event_callback(event_type: str, data: dict):
            asyncio.create_task(event_callback(job_id, event_type, data))

        # Submit job
        job_manager.submit_job(
            job_id,
            task_fn,
            progress_callback=sync_progress_callback,
            event_callback=sync_event_callback,
        )

    except Exception as e:
        logger.error(f"Job submission failed: {e}")
        # Update job status
        job = job_manager.get_job(job_id)
        if job:
            job.status = JobStatus.FAILED
            job.error = str(e)

        # Notify subscribers
        await event_manager.publish(job_id, EventType.ERROR, {"error": str(e)})
