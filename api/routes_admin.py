from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from utils.cache_admin import cache_admin
from utils.presets import preset_manager
from api.config import get_settings
from utils.logger import logger

router = APIRouter()
settings = get_settings()


@router.get("/admin/cache/size")
async def get_cache_size():
    """Get cache size information"""
    try:
        cache_info = cache_admin.get_cache_size()
        return cache_info

    except Exception as e:
        logger.error(f"Cache size query failed: {str(e)}")
        raise HTTPException(500, f"Cache size query failed: {str(e)}")


@router.post("/admin/cache/prune")
async def prune_cache(
    dry_run: bool = Query(True, description="Dry run without actual deletion"),
    retention_days: Optional[int] = Query(None),
    keep_per_task: Optional[int] = Query(None),
):
    """Prune old cache entries"""
    try:
        report = cache_admin.prune_outputs(
            dry_run=dry_run, retention_days=retention_days, keep_per_task=keep_per_task
        )

        return report

    except Exception as e:
        logger.error(f"Cache pruning failed: {str(e)}")
        raise HTTPException(500, f"Cache pruning failed: {str(e)}")


@router.get("/admin/cache/verify")
async def verify_cache():
    """Verify cache integrity"""
    try:
        verification_report = cache_admin.verify_models()
        return verification_report

    except Exception as e:
        logger.error(f"Cache verification failed: {str(e)}")
        raise HTTPException(500, f"Cache verification failed: {str(e)}")


@router.post("/admin/cache/cleanup")
async def cleanup_temp_files(older_than_hours: int = Query(24)):
    """Clean up temporary files"""
    try:
        report = cache_admin.cleanup_temp_files(older_than_hours=older_than_hours)
        return report

    except Exception as e:
        logger.error(f"Temp file cleanup failed: {str(e)}")
        raise HTTPException(500, f"Temp file cleanup failed: {str(e)}")


@router.get("/admin/presets/export")
async def export_presets():
    """Export presets configuration"""
    try:
        presets_data = {
            "presets": [
                {
                    "name": preset.name,
                    "task_type": preset.task_type.value,
                    "parameters": preset.parameters,
                    "description": preset.description,
                    "tags": preset.tags,
                }
                for preset in preset_manager.presets.values()
            ],
            "exported_at": datetime.now().isoformat(),
            "version": "1.0",
        }

        return presets_data

    except Exception as e:
        logger.error(f"Presets export failed: {str(e)}")
        raise HTTPException(500, f"Presets export failed: {str(e)}")


@router.post("/admin/presets/import")
async def import_presets(presets_data: dict):
    """Import presets configuration"""
    try:
        imported_count = 0
        errors = []

        for preset_data in presets_data.get("presets", []):
            try:
                success = preset_manager.create_preset(
                    name=preset_data["name"],
                    task_type=preset_data["task_type"],
                    parameters=preset_data["parameters"],
                    description=preset_data.get("description", ""),
                    tags=preset_data.get("tags", []),
                )

                if success:
                    imported_count += 1
                else:
                    errors.append(f"Preset {preset_data['name']} already exists")

            except Exception as e:
                errors.append(
                    f"Failed to import {preset_data.get('name', 'unknown')}: {str(e)}"
                )

        return {
            "imported_count": imported_count,
            "errors": errors,
            "total_presets": len(preset_manager.presets),
        }

    except Exception as e:
        logger.error(f"Presets import failed: {str(e)}")
        raise HTTPException(500, f"Presets import failed: {str(e)}")
