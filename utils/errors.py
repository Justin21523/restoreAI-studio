from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ErrorCode(str, Enum):
    """Standardized error codes"""

    # 4xx Errors
    VALIDATION_ERROR = "VALIDATION_ERROR"
    INVALID_INPUT = "INVALID_INPUT"
    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    UNSUPPORTED_FORMAT = "UNSUPPORTED_FORMAT"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    JOB_QUEUE_FULL = "JOB_QUEUE_FULL"

    # 5xx Errors
    INTERNAL_ERROR = "INTERNAL_ERROR"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    PROCESSING_FAILED = "PROCESSING_FAILED"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    EXTERNAL_SERVICE_ERROR = "EXTERNAL_SERVICE_ERROR"


@dataclass
class RestorAIError(Exception):
    """Base exception for RestorAI application"""

    code: ErrorCode
    message: str
    hint: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for API response"""
        error_dict = {"code": self.code.value, "message": self.message}

        if self.hint:
            error_dict["hint"] = self.hint

        if self.details:
            error_dict["details"] = self.details

        return error_dict


class ValidationError(RestorAIError):
    """Input validation error"""

    def __init__(
        self, message: str, field: Optional[str] = None, hint: Optional[str] = None
    ):
        details = {"field": field} if field else None
        super().__init__(
            code=ErrorCode.VALIDATION_ERROR, message=message, hint=hint, details=details
        )


class FileTooLargeError(RestorAIError):
    """File size exceeds limit"""

    def __init__(self, max_size_mb: int, actual_size_mb: float):
        super().__init__(
            code=ErrorCode.FILE_TOO_LARGE,
            message=f"File size {actual_size_mb:.1f}MB exceeds maximum {max_size_mb}MB",
            hint="Please upload a smaller file or adjust the size limits",
        )


class UnsupportedFormatError(RestorAIError):
    """Unsupported file format"""

    def __init__(self, format: str, supported_formats: list):
        super().__init__(
            code=ErrorCode.UNSUPPORTED_FORMAT,
            message=f"Unsupported format: {format}",
            hint=f"Supported formats: {', '.join(supported_formats)}",
        )


class ModelLoadError(RestorAIError):
    """Model loading failed"""

    def __init__(self, model_name: str, reason: str):
        super().__init__(
            code=ErrorCode.MODEL_LOAD_FAILED,
            message=f"Failed to load model: {model_name}",
            hint=reason,
            details={"model": model_name},
        )


class ProcessingError(RestorAIError):
    """Processing failed"""

    def __init__(self, task_type: str, reason: str):
        super().__init__(
            code=ErrorCode.PROCESSING_FAILED,
            message=f"Processing failed for {task_type}",
            hint=reason,
            details={"task_type": task_type},
        )


class OutOfMemoryError(RestorAIError):
    """Out of memory error"""

    def __init__(self, allocated: float, available: float):
        super().__init__(
            code=ErrorCode.OUT_OF_MEMORY,
            message="Insufficient memory for processing",
            hint="Try using a smaller input or enable memory optimizations",
            details={"allocated_gb": allocated, "available_gb": available},
        )


class JobQueueFullError(RestorAIError):
    """Job queue is full"""

    def __init__(self, max_queue_size: int):
        super().__init__(
            code=ErrorCode.JOB_QUEUE_FULL,
            message="Job queue is full",
            hint="Please try again later or increase the queue size",
            details={"max_queue_size": max_queue_size},
        )


def error_handler(func):
    """Decorator to handle and convert exceptions to RestorAIError"""
    from functools import wraps

    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except RestorAIError:
            raise
        except ValueError as e:
            raise ValidationError(str(e))
        except FileNotFoundError as e:
            raise RestorAIError(
                code=ErrorCode.INVALID_INPUT, message="File not found", hint=str(e)
            )
        except Exception as e:
            raise RestorAIError(
                code=ErrorCode.INTERNAL_ERROR,
                message="Internal server error",
                hint="Please contact support if this persists",
                details={"internal_error": str(e)},
            )

    return wrapper
