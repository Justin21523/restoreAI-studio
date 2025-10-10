import logging
import logging.config
import json
import time
import uuid
from contextvars import ContextVar
from typing import Dict, Any, Optional
from pathlib import Path

from api.config import get_settings

settings = get_settings()

# Context variable for request ID
request_id_ctx: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class StructuredFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record: logging.LogRecord) -> str:
        # Create structured log entry
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add request ID if available
        request_id = request_id_ctx.get(None)
        if request_id:
            log_entry["request_id"] = request_id

        # Add extra fields
        if hasattr(record, "props"):
            log_entry.update(record.props)

        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, ensure_ascii=False)


class RequestIdFilter(logging.Filter):
    """Add request ID to log records"""

    def filter(self, record: logging.LogRecord) -> bool:
        request_id = request_id_ctx.get(None)
        if request_id:
            record.request_id = request_id
        return True


def setup_logging():
    """Setup structured logging configuration"""

    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    logging_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {"()": StructuredFormatter},
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "filters": {"request_id": {"()": RequestIdFilter}},
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": settings.log_level.upper(),
                "formatter": (
                    "structured" if settings.log_format == "json" else "simple"
                ),
                "filters": ["request_id"],
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "INFO",
                "formatter": "structured",
                "filters": ["request_id"],
                "filename": log_dir / "restorai.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5,
            },
        },
        "loggers": {
            "restorai": {
                "level": settings.log_level.upper(),
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "api": {
                "level": settings.log_level.upper(),
                "handlers": ["console", "file"],
                "propagate": False,
            },
            "core": {
                "level": settings.log_level.upper(),
                "handlers": ["console", "file"],
                "propagate": False,
            },
        },
        "root": {"level": "WARNING", "handlers": ["console"]},
    }

    logging.config.dictConfig(logging_config)


def get_logger(name: str) -> logging.Logger:
    """Get logger with structured logging support"""
    return logging.getLogger(name)


class RequestTimer:
    """Timer for measuring request processing stages"""

    def __init__(self, request_id: str):
        self.request_id = request_id
        self.stages: Dict[str, float] = {}
        self.current_stage: Optional[str] = None
        self.start_time: Optional[float] = None

    def start_stage(self, stage_name: str):
        """Start timing a processing stage"""
        if self.current_stage:
            self.end_stage()

        self.current_stage = stage_name
        self.start_time = time.time()

    def end_stage(self):
        """End timing of current stage"""
        if self.current_stage and self.start_time:
            duration = time.time() - self.start_time
            self.stages[self.current_stage] = duration

            # Log stage completion
            logger = get_logger("api.timing")
            logger.info(
                f"Stage {self.current_stage} completed",
                extra={
                    "props": {
                        "request_id": self.request_id,
                        "stage": self.current_stage,
                        "duration_ms": duration * 1000,
                    }
                },
            )

            self.current_stage = None
            self.start_time = None

    def get_stage_durations(self) -> Dict[str, float]:
        """Get all stage durations"""
        return self.stages.copy()


# Global logger instance
logger = get_logger("restorai")
