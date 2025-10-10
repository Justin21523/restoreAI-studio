from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from api.models import TaskResponse
from core.video_pipeline import VideoPipeline
from api.jobs import job_runner, JobStatus
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()
video_pipeline = VideoPipeline()


@router.post("/video/interpolate")
async def video_interpolation(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    target_fps: float = Form(60),
    model: str = Form("rife-v4.6"),
):
    """
    Video frame interpolation endpoint
    """
    # Validate input
    if target_fps <= 0:
        raise HTTPException(400, "Target FPS must be positive")

    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files are supported")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        # Process video
        result = video_pipeline.interpolate_video(
            video_path=tmp_path, target_fps=target_fps, model_name=model
        )

        return result

    except Exception as e:
        logger.error(f"Video interpolation failed: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        # Cleanup temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/video/upscale")
async def video_upscale(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: int = Form(2),
    model: str = Form("realesrgan-x4plus"),
):
    """
    Video upscaling endpoint
    """
    # Validate input
    if scale not in [2, 4]:
        raise HTTPException(400, "Scale must be 2 or 4")

    if not file.content_type.startswith("video/"):
        raise HTTPException(400, "Only video files are supported")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        # Process video
        result = video_pipeline.upscale_video(
            video_path=tmp_path, scale=scale, model_name=model
        )

        return result

    except Exception as e:
        logger.error(f"Video upscale failed: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        # Cleanup temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/batch/submit")
async def batch_submit(
    task: str = Form(...),
    file: UploadFile = File(...),
    # Common parameters
    scale: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
    target_fps: Optional[float] = Form(None),
    strength: Optional[float] = Form(None),
):
    """
    Submit batch job for processing
    """
    # Validate task type
    valid_tasks = ["sr", "face", "upscale", "interpolate"]
    if task not in valid_tasks:
        raise HTTPException(400, f"Task must be one of: {valid_tasks}")

    # Create temporary file
    file_suffix = (
        ".zip" if task == "batch" else ".png" if task in ["sr", "face"] else ".mp4"
    )
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    # Prepare task function and parameters
    if task == "sr":
        task_fn = video_pipeline.super_resolution
        params = {
            "image_path": tmp_path,
            "scale": scale or 2,
            "model_name": model or settings.default_sr_model,
        }
    elif task == "face":
        task_fn = video_pipeline.face_restoration
        params = {
            "image_path": tmp_path,
            "method": model or settings.default_face_model,
            "strength": strength or 0.5,
        }
    elif task == "upscale":
        task_fn = video_pipeline.upscale_video
        params = {
            "video_path": tmp_path,
            "scale": scale or 2,
            "model_name": model or settings.default_sr_model,
        }
    elif task == "interpolate":
        task_fn = video_pipeline.interpolate_video
        params = {
            "video_path": tmp_path,
            "target_fps": target_fps or settings.target_fps,
            "model_name": model or "rife-v4.6",
        }

    # Define progress callback
    def progress_callback(progress: float):
        logger.info(f"Batch job progress: {progress:.1%}")

    try:
        # Submit job
        job_id = job_runner.submit(
            task_type=task,
            parameters=params,
            task_fn=task_fn,
            progress_callback=progress_callback,
        )

        return {"task_id": job_id}

    except Exception as e:
        logger.error(f"Batch submission failed: {str(e)}")
        raise HTTPException(500, f"Job submission failed: {str(e)}")


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task_status(task_id: str):
    """
    Get task status and result
    """
    job = job_runner.get_job(task_id)
    if not job:
        raise HTTPException(404, "Task not found")

    return TaskResponse(
        task_id=job.id,
        status=job.status,
        progress=job.progress,
        result=job.result,
        error=job.error,
    )


@router.delete("/tasks/{task_id}")
async def cancel_task(task_id: str):
    """
    Cancel a running task
    """
    success = job_runner.cancel_job(task_id)
    if not success:
        raise HTTPException(404, "Task not found or cannot be cancelled")

    return {"message": "Task cancelled successfully"}


@router.get("/jobs/queue")
async def get_queue_status():
    """
    Get job queue status
    """
    return job_runner.get_queue_status()
