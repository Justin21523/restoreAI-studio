# api/main.py
"""
restoreAI-studio - FastAPI Server
Lightweight REST API for AI image restoration.
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
config.create_directories()

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
    return {
        "name": config.api_title,
        "description": config.api_description,
        "version": "1.0.0",
        "status": "running",
        "docs_url": "/docs",
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
    """Simple health check endpoint."""
    try:
        # Check if required directories exist
        config.create_directories()

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
            "status": "healthy",
            "device": config.device,
            "device_status": device_status,
            "models_available": available_models,
            "directories_ok": True,
        }

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503, content={"status": "unhealthy", "error": str(e)}
        )


@app.get("/models")
async def get_models():
    """Get information about available models."""
    try:
        models = config.get_model_info()
        return {
            "models": models,
            "summary": {
                "total": len(models),
                "available": len([m for m in models.values() if m["available"]]),
                "total_size_mb": sum(
                    m["size_mb"] for m in models.values() if m["available"]
                ),
            },
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Application startup tasks."""
    logger.info(f"Starting {config.api_title}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Model directory: {config.model_dir}")

    # Ensure directories exist
    config.create_directories()

    # Log available models
    models = config.get_model_info()
    available_models = [name for name, info in models.items() if info["available"]]
    if available_models:
        logger.info(f"Available models: {', '.join(available_models)}")
    else:
        logger.warning(
            "No models found. Download with: python scripts/download_models.py"
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
