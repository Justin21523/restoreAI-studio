import time
import uuid
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Any
from fastapi import Request, Response, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import json
from utils.logging import request_id_ctx, RequestTimer, get_logger
from utils.metrics import metrics_collector
from utils.errors import RestorAIError

from core.safety import safety_checker
from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class InputValidationMiddleware(BaseHTTPMiddleware):
    """Middleware for input validation and safety checks"""

    async def dispatch(self, request: Request, call_next):
        # Skip validation for certain paths
        if request.url.path in ["/healthz", "/docs", "/openapi.json"]:
            return await call_next(request)

        # Check content length for file uploads
        if request.method == "POST" and any(
            path in request.url.path for path in ["/restore", "/video", "/batch"]
        ):
            content_length = request.headers.get("content-length")
            if content_length:
                max_size = settings.max_file_mb * 1024 * 1024
                if int(content_length) > max_size:
                    return JSONResponse(
                        status_code=413,
                        content={
                            "error": "File too large",
                            "message": f"Maximum file size is {settings.max_file_mb}MB",
                        },
                    )

        # Process request
        response = await call_next(request)
        return response


class SafetyMiddleware(BaseHTTPMiddleware):
    """Middleware for safety checks"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Add safety headers
        response.headers["X-Content-Safety"] = "enabled"
        response.headers["X-NSFW-Check"] = (
            "enabled" if settings.enable_nsfw else "disabled"
        )

        return response


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Middleware to assign request IDs"""

    async def dispatch(self, request: Request, call_next):
        # Generate request ID
        request_id = str(uuid.uuid4())
        request_id_ctx.set(request_id)

        # Add request ID to response headers
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect request metrics"""

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            duration = time.time() - start_time

            # Record metrics
            metrics_collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration=duration,
            )

            return response

        except Exception as e:
            duration = time.time() - start_time

            # Record metrics for failed requests
            status_code = 500
            if isinstance(e, RestorAIError):
                # Map RestorAIError to appropriate status code
                if e.code.value.startswith("4"):
                    status_code = 400
                else:
                    status_code = 500

            metrics_collector.record_request(
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                duration=duration,
            )

            raise


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to time request processing stages"""

    async def dispatch(self, request: Request, call_next):
        request_id = request_id_ctx.get(None)
        if not request_id:
            request_id = str(uuid.uuid4())
            request_id_ctx.set(request_id)

        timer = RequestTimer(request_id)

        try:
            # Time overall request
            timer.start_stage("total")

            response = await call_next(request)

            timer.end_stage()
            return response

        finally:
            # Ensure timer is cleaned up
            if timer.current_stage:
                timer.end_stage()


async def global_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global error handler for RestorAI exceptions"""
    if isinstance(exc, RestorAIError):
        # Log the error
        logger.error(
            f"RestorAI error: {exc.code.value} - {exc.message}",
            extra={
                "props": {
                    "error_code": exc.code.value,
                    "error_message": exc.message,
                    "path": request.url.path,
                    "method": request.method,
                }
            },
        )

        # Determine status code
        status_code = 500
        if exc.code.value.startswith("4"):
            status_code = 400

        return JSONResponse(status_code=status_code, content=exc.to_dict())

    # Handle unexpected errors
    logger.error(
        f"Unexpected error: {str(exc)}",
        extra={
            "props": {
                "error_type": type(exc).__name__,
                "path": request.url.path,
                "method": request.method,
            }
        },
        exc_info=True,
    )

    return JSONResponse(
        status_code=500,
        content={
            "code": "INTERNAL_ERROR",
            "message": "An unexpected error occurred",
            "hint": "Please contact support if this persists",
        },
    )


async def validate_upload_file(file_path: Path, file_type: str) -> Dict[str, Any]:
    """Validate uploaded file"""
    is_valid, message = safety_checker.validate_input(file_path, file_type)

    if not is_valid:
        raise HTTPException(400, f"File validation failed: {message}")

    return {
        "valid": True,
        "file_size": file_path.stat().st_size,
        "file_type": file_type,
    }
