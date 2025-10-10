from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import tempfile
import uuid
from pathlib import Path
import shutil
import time
import psutil
import torch

from api.models import (
    SRRequest,
    SRResponse,
    FaceRestoreRequest,
    FaceRestoreResponse,
    TaskResponse,
    ErrorResponse,
)
from core.pipeline import ImagePipeline
from api.config import get_settings
from utils.logger import logger
from pydantic import BaseModel


router = APIRouter()
settings = get_settings()
pipeline = ImagePipeline()

# In-memory task store (replace with Redis in production)
task_store = {}


class HealthResponse(BaseModel):
    """Health check response model"""

    status: str
    timestamp: float
    system_info: dict


@router.get("/healthz", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns system status and basic diagnostics
    """
    try:
        # Basic system info
        system_info = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage("/").percent,
            "gpu_available": torch.cuda.is_available(),
        }

        if torch.cuda.is_available():
            system_info.update(
                {
                    "gpu_count": torch.cuda.device_count(),
                    "gpu_memory_allocated": torch.cuda.memory_allocated()
                    / 1024**3,  # GB
                    "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3,  # GB
                }
            )

        return HealthResponse(
            status="ok", timestamp=time.time(), system_info=system_info
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")


@router.get("/cache/paths")
async def get_cache_paths():
    """Return configured cache paths for debugging"""
    return {
        "paths": {key: str(path) for key, path in settings.paths.items()},
        "env_loaded": True,
    }


@router.post("/restore/sr", response_model=SRResponse)
async def super_resolution(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    scale: int = Form(2),
    model: str = Form("realesrgan-x4plus"),
):
    """
    Image super-resolution endpoint
    """
    # Validate input
    if scale not in [2, 4]:
        raise HTTPException(400, "Scale must be 2 or 4")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only JPEG and PNG images are supported")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        # Process image
        result = pipeline.super_resolution(
            image_path=tmp_path, scale=scale, model_name=model
        )

        return SRResponse(**result)

    except Exception as e:
        logger.error(f"Super-resolution failed: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        # Cleanup temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@router.post("/restore/face", response_model=FaceRestoreResponse)
async def face_restoration(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    method: str = Form("gfpgan"),
    strength: float = Form(0.5),
):
    """
    Face restoration endpoint
    """
    # Validate input
    if method not in ["gfpgan", "codeformer"]:
        raise HTTPException(400, "Method must be 'gfpgan' or 'codeformer'")

    if not 0 <= strength <= 1:
        raise HTTPException(400, "Strength must be between 0 and 1")

    if file.content_type not in ["image/jpeg", "image/png", "image/jpg"]:
        raise HTTPException(400, "Only JPEG and PNG images are supported")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        # Process image
        result = pipeline.face_restoration(
            image_path=tmp_path, method=method, strength=strength
        )

        return FaceRestoreResponse(**result)

    except Exception as e:
        logger.error(f"Face restoration failed: {str(e)}")
        raise HTTPException(500, f"Processing failed: {str(e)}")

    finally:
        # Cleanup temporary file
        if tmp_path.exists():
            tmp_path.unlink()


@router.get("/outputs/{run_id}/{filename}")
async def get_output_file(run_id: str, filename: str):
    """
    Serve output files
    """
    output_path = settings.paths["outputs"] / "images" / run_id / filename

    if not output_path.exists():
        raise HTTPException(404, "File not found")

    return FileResponse(path=output_path, media_type="image/png", filename=filename)
