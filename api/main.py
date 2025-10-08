# api/main.py
"""
RestorAI MVP - FastAPI Server
Lightweight REST API for AI image restoration with centralized warehouse support.

Why: Uses centralized AI warehouse for model storage to enable sharing across projects
and provides proper error handling when warehouse paths are not accessible.
"""

import os
import sys
import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.config import Config
from api.routes import router as api_router

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration
config = Config()


# Validate warehouse paths on startup
def validate_warehouse_setup():
    """Validate that AI warehouse is properly configured and accessible."""
    try:
        # Check warehouse health
        validation_results = config.validate_paths()

        if not validation_results.get("models", False):
            raise Exception(
                f"AI warehouse models directory is not accessible: {config.PATHS['models']}\n"
                f"Please check AI_WAREHOUSE_ROOT, MODELS_DIR environment variables, "
                f"or run: python scripts/download_models.py --health"
            )

        if not validation_results.get("cache", False):
            raise Exception(
                f"AI warehouse cache directory is not accessible: {config.PATHS['cache']}\n"
                f"Please check AI_WAREHOUSE_ROOT, CACHE_DIR environment variables"
            )

        # Create directories if needed
        config.create_directories()

        logger.info(f"✅ AI Warehouse validated: {config.PATHS['warehouse']}")
        return True

    except Exception as e:
        logger.error(f"❌ AI Warehouse validation failed: {e}")
        return False


# Validate warehouse before creating app
warehouse_healthy = validate_warehouse_setup()

# Create FastAPI app
app = FastAPI(
    title=config.api_title,
    description=config.api_description,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api/v1")

# Serve static files (output directory)
if config.output_dir.exists():
    app.mount("/outputs", StaticFiles(directory=str(config.output_dir)), name="outputs")


@app.get("/")
async def root():
    """Root endpoint with basic information."""
    # Check warehouse status
    if not warehouse_healthy:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "AI Warehouse not accessible",
                "message": "The centralized AI warehouse is not properly configured or accessible.",
                "warehouse_path": str(config.PATHS["warehouse"]),
                "models_path": str(config.PATHS["models"]),
                "suggestions": [
                    "Check environment variables: AI_WAREHOUSE_ROOT, MODELS_DIR, CACHE_DIR",
                    "Run: python scripts/download_models.py --health",
                    "Ensure warehouse directory has write permissions",
                ],
            },
        )

    return {
        "name": config.api_title,
        "description": config.api_description,
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
        "warehouse": {
            "location": str(config.PATHS["warehouse"]),
            "models_dir": str(config.PATHS["models"]),
            "cache_dir": str(config.PATHS["cache"]),
        },
        "config": {
            "device": config.device,
            "models_available": len(
                [m for m in config.get_model_info().values() if m["available"]]
            ),
            "max_image_size": config.max_image_size,
        },
    }


@app.get("/health")
async def health_check():
    """Comprehensive health check including warehouse status."""
    try:
        # Check warehouse health
        validation_results = config.validate_paths()
        warehouse_status = "healthy" if all(validation_results.values()) else "degraded"

        if warehouse_status == "degraded":
            failed_paths = [k for k, v in validation_results.items() if not v]
            logger.warning(f"Warehouse degraded - failed paths: {failed_paths}")

        # Check model availability
        models = config.get_model_info()
        available_models = [name for name, info in models.items() if info["available"]]

        # Check device availability
        device_status = "ok"
        if config.device == "cuda":
            try:
                import torch

                if not torch.cuda.is_available():
                    device_status = "cuda_unavailable"
            except ImportError:
                device_status = "torch_not_installed"

        return {
            "status": "healthy" if warehouse_status == "healthy" else "degraded",
            "warehouse": {
                "status": warehouse_status,
                "location": str(config.PATHS["warehouse"]),
                "validation": validation_results,
            },
            "device": config.device,
            "device_status": device_status,
            "models_available": available_models,
            "directories_ok": warehouse_status == "healthy",
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "warehouse": {
                    "status": "error",
                    "location": str(config.PATHS["warehouse"]),
                },
            },
        )


@app.get("/models")
async def get_models():
    """Get information about available models in centralized warehouse."""
    try:
        # Validate warehouse before checking models
        if not warehouse_healthy:
            raise HTTPException(
                status_code=500,
                detail="AI warehouse is not accessible. Check configuration and run health check.",
            )

        models = config.get_model_info()
        return {
            "models": models,
            "warehouse_location": str(config.PATHS["models"]),
            "summary": {
                "total": len(models),
                "available": len([m for m in models.values() if m["available"]]),
                "total_size_mb": sum(
                    m["size_mb"] for m in models.values() if m["available"]
                ),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {config.api_title}")
    logger.info(f"Device: {config.device}")
    logger.info(f"AI Warehouse: {config.PATHS['warehouse']}")
    logger.info(f"Models directory: {config.PATHS['models']}")
    logger.info(f"Cache directory: {config.PATHS['cache']}")

    if not warehouse_healthy:
        logger.error("❌ AI Warehouse is not healthy - some features may not work")
        logger.error("   Run: python scripts/download_models.py --health")
        return

    # Log available models
    models = config.get_model_info()
    available_models = [name for name, info in models.items() if info["available"]]
    if available_models:
        logger.info(f"Available models in warehouse: {', '.join(available_models)}")
    else:
        logger.warning(
            "No models found in warehouse. Download with: python scripts/download_models.py --essential"
        )


@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks."""
    logger.info("Shutting down RestorAI MVP API")

    # Clean up any loaded models
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}")

    # Check if it's a warehouse-related error
    if "warehouse" in str(exc).lower() or "models" in str(exc).lower():
        return JSONResponse(
            status_code=500,
            content={
                "error": "AI Warehouse Error",
                "detail": str(exc),
                "warehouse_location": str(config.PATHS["warehouse"]),
                "suggestions": [
                    "Check AI warehouse configuration",
                    "Run: python scripts/download_models.py --health",
                    "Verify environment variables: AI_WAREHOUSE_ROOT, MODELS_DIR",
                ],
            },
        )

    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": (
                str(exc) if config.device == "cpu" else "An error occurred"
            ),  # Hide details in production
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host=config.api_host, port=config.api_port, reload=True)
