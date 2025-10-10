from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any

from utils.safety import safety_checker
from utils.presets import preset_manager, TaskType
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()


@router.post("/safety/test")
async def test_safety(
    file: UploadFile = File(...),
    enable_nsfw: bool = Form(True),
    enable_face_blur: bool = Form(False),
):
    """
    Test safety checks on an image (development only)
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(400, "Only image files are supported")

    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        # Create output path
        output_path = tmp_path.parent / f"{tmp_path.stem}_safe{tmp_path.suffix}"

        # Run safety processing
        safety_result = safety_checker.process_image_safety(
            tmp_path,
            output_path,
            enable_nsfw=enable_nsfw,
            enable_face_blur=enable_face_blur,
        )

        # Prepare response
        response = {
            "safety_result": safety_result,
            "input_file": str(tmp_path),
            "output_file": str(output_path) if output_path.exists() else None,
        }

        return response

    except Exception as e:
        logger.error(f"Safety test failed: {str(e)}")
        raise HTTPException(500, f"Safety test failed: {str(e)}")

    finally:
        # Cleanup temporary files
        for path in [tmp_path, output_path]:
            if path and path.exists():
                path.unlink()


@router.post("/safety/check")
async def safety_check(file: UploadFile = File(...)):
    """Check image for safety issues"""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
            tmp_path = Path(tmp_file.name)
            content = await file.read()
            tmp_file.write(content)

        # Read image
        import cv2

        image = cv2.imread(str(tmp_path))
        if image is None:
            raise HTTPException(400, "Cannot read image file")

        # Perform safety check
        safety_result = safety_checker.process_safety(image, tmp_path)

        # Cleanup
        tmp_path.unlink()

        return {
            "safety_check": safety_result,
            "settings": {
                "safety_enabled": settings.safety_enabled,
                "nsfw_threshold": settings.nsfw_threshold,
                "auto_face_blur": settings.auto_face_blur,
            },
        }

    except Exception as e:
        logger.error(f"Safety check failed: {str(e)}")
        raise HTTPException(500, f"Safety check failed: {str(e)}")


@router.get("/safety/status")
async def get_safety_status():
    """Get safety module status"""
    return {
        "nsfw_enabled": settings.enable_nsfw,
        "face_blur_enabled": settings.enable_face_blur,
        "nsfw_model_loaded": safety_checker.nsfw_model is not None,
        "face_detector_loaded": safety_checker.face_detector is not None,
    }


@router.post("/presets")
async def create_preset(
    name: str = Form(...),
    task_type: str = Form(...),
    parameters: str = Form(...),  # JSON string
    description: str = Form(""),
    tags: str = Form(""),  # Comma-separated
):
    """Create a new preset"""
    try:
        # Parse parameters
        import json

        params_dict = json.loads(parameters)

        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(",")] if tags else []

        # Create preset
        success = preset_manager.create_preset(
            name=name,
            task_type=TaskType(task_type),
            parameters=params_dict,
            description=description,
            tags=tag_list,
        )

        if not success:
            raise HTTPException(400, "Preset already exists")

        return {"message": f"Preset '{name}' created successfully"}

    except Exception as e:
        logger.error(f"Preset creation failed: {str(e)}")
        raise HTTPException(500, f"Preset creation failed: {str(e)}")


@router.get("/presets")
async def list_presets(task_type: Optional[str] = None):
    """List available presets"""
    try:
        if task_type:
            presets = preset_manager.get_presets_by_task(TaskType(task_type))
        else:
            presets = list(preset_manager.presets.values())

        return {
            "presets": [
                {
                    "name": preset.name,
                    "task_type": preset.task_type.value,
                    "parameters": preset.parameters,
                    "description": preset.description,
                    "tags": preset.tags,
                }
                for preset in presets
            ]
        }

    except Exception as e:
        logger.error(f"Failed to list presets: {str(e)}")
        raise HTTPException(500, f"Failed to list presets: {str(e)}")


@router.delete("/presets/{preset_name}")
async def delete_preset(preset_name: str):
    """Delete a preset"""
    try:
        success = preset_manager.delete_preset(preset_name)

        if not success:
            raise HTTPException(404, "Preset not found")

        return {"message": f"Preset '{preset_name}' deleted successfully"}

    except Exception as e:
        logger.error(f"Preset deletion failed: {str(e)}")
        raise HTTPException(500, f"Preset deletion failed: {str(e)}")
