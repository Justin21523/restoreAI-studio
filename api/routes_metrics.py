from fastapi import APIRouter
from fastapi.responses import Response

from utils.metrics import metrics_collector
from api.config import get_settings

router = APIRouter()
settings = get_settings()


@router.get("/metrics")
async def get_metrics():
    """
    Expose application metrics in Prometheus format
    """
    if not settings.metrics_enabled:
        return {"message": "Metrics are disabled"}

    metrics_text = metrics_collector.get_metrics_text()
    return Response(content=metrics_text, media_type="text/plain")


@router.get("/diagnostics/health")
async def health_check():
    """Comprehensive health check"""
    from core.manager import resource_manager
    from core.lifecycle import lifecycle_manager

    try:
        # Basic system info
        system_info = {
            "status": "healthy",
            "timestamp": time.time(),
            "version": "1.0.0",
        }

        # Resource manager status
        resource_status = resource_manager.get_status()
        system_info["resources"] = {
            "gpu_available": any(
                d.startswith("cuda") for d in resource_status["devices"]
            ),
            "models_loaded": len(resource_status["model_pool"]["models"]),
            "memory_usage": resource_status["memory_info"],
        }

        # Lifecycle manager status
        lifecycle_status = lifecycle_manager.get_status()
        system_info["models"] = {
            "registered": len(lifecycle_status["registered_models"]),
            "healthy": sum(
                1
                for s in lifecycle_status["health_status"].values()
                if s.get("healthy", False)
            ),
        }

        return system_info

    except Exception as e:
        return {"status": "unhealthy", "error": str(e), "timestamp": time.time()}
