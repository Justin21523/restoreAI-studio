from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Form
from typing import List, Optional
import zipfile
import tempfile
from pathlib import Path
import asyncio

from api.job_manager import job_manager
from api.events import event_manager, EventType
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()


@router.post("/batch/sr")
async def batch_super_resolution(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    scale: int = Form(2),
    model: str = Form("realesrgan-x4plus"),
):
    """Batch super resolution processing"""
    return await _process_batch(
        background_tasks, files, "super_resolution", {"scale": scale, "model": model}
    )


@router.post("/batch/face")
async def batch_face_restoration(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    method: str = Form("gfpgan"),
    strength: float = Form(0.5),
):
    """Batch face restoration processing"""
    return await _process_batch(
        background_tasks,
        files,
        "face_restoration",
        {"method": method, "strength": strength},
    )


@router.post("/batch/zip")
async def batch_from_zip(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    task_type: str = Form(...),
    scale: Optional[int] = Form(None),
    model: Optional[str] = Form(None),
    method: Optional[str] = Form(None),
    strength: Optional[float] = Form(None),
):
    """Batch processing from ZIP file"""
    try:
        # Create temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            zip_path = temp_path / "batch.zip"

            # Save uploaded ZIP
            with open(zip_path, "wb") as f:
                f.write(await file.read())

            # Extract ZIP
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(temp_path)

            # Find image files
            image_files = []
            for ext in ["*.png", "*.jpg", "*.jpeg"]:
                image_files.extend(temp_path.rglob(ext))

            if not image_files:
                raise HTTPException(400, "No image files found in ZIP")

            # Prepare parameters
            parameters = {}
            if task_type == "super_resolution":
                parameters = {
                    "scale": scale or 2,
                    "model": model or "realesrgan-x4plus",
                }
            elif task_type == "face_restoration":
                parameters = {"method": method or "gfpgan", "strength": strength or 0.5}
            else:
                raise HTTPException(400, "Invalid task type")

            # Create batch job
            batch_id = job_manager.create_job(
                f"batch_{task_type}",
                {"files_count": len(image_files), "parameters": parameters},
            )

            # Process each file
            for image_file in image_files:
                file_job_id = job_manager.create_job(
                    task_type, {"input_path": str(image_file), **parameters}
                )

                # Submit individual job
                background_tasks.add_task(
                    submit_job_for_processing,
                    file_job_id,
                    task_type,
                    {"input_path": str(image_file), **parameters},
                )

            return {
                "batch_id": batch_id,
                "files_processed": len(image_files),
                "task_type": task_type,
                "parameters": parameters,
            }

    except Exception as e:
        logger.error(f"Batch ZIP processing failed: {str(e)}")
        raise HTTPException(500, f"Batch processing failed: {str(e)}")


async def _process_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile],
    task_type: str,
    parameters: dict,
):
    """Process batch of files"""
    try:
        # Validate files
        if len(files) > settings.max_batch_size:
            raise HTTPException(
                400, f"Too many files. Maximum: {settings.max_batch_size}"
            )

        # Create batch job
        batch_id = job_manager.create_job(
            f"batch_{task_type}", {"files_count": len(files), "parameters": parameters}
        )

        # Process each file
        for file in files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
                tmp_path = Path(tmp_file.name)
                content = await file.read()
                tmp_file.write(content)

            # Create individual job
            file_job_id = job_manager.create_job(
                task_type, {"input_path": str(tmp_path), **parameters}
            )

            # Submit job
            background_tasks.add_task(
                submit_job_for_processing,
                file_job_id,
                task_type,
                {"input_path": str(tmp_path), **parameters},
            )

        return {
            "batch_id": batch_id,
            "files_processed": len(files),
            "task_type": task_type,
            "parameters": parameters,
        }

    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        raise HTTPException(500, f"Batch processing failed: {str(e)}")
