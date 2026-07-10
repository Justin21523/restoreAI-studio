from __future__ import annotations

import errno
import subprocess

from restorai.model_registry import ModelRegistryError


def classify_processing_error(exc: Exception) -> tuple[str, str]:
    """Map internal exceptions to stable, user-facing product errors."""
    message = str(exc)
    lowered = message.lower()
    if isinstance(exc, ModelRegistryError):
        return "MODEL_UNAVAILABLE", "A required model is missing or failed verification"
    if isinstance(exc, subprocess.CalledProcessError):
        return "FFMPEG_FAILED", "FFmpeg failed while decoding or exporting the video"
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return "STORAGE_FULL", "Storage is full; free disk space and retry"
    if "out of memory" in lowered or "cuda error: memory" in lowered:
        return "CUDA_OUT_OF_MEMORY", "GPU memory was exhausted; reduce scale or input size"
    if "max_output" in lowered or "maximum resolution" in lowered or "output exceeds" in lowered:
        return "OUTPUT_LIMIT_EXCEEDED", message[:500]
    if isinstance(exc, ValueError):
        kind = "INVALID_VIDEO" if "video" in lowered or "fps" in lowered else "INVALID_IMAGE"
        return kind, message[:500]
    return "PROCESSING_FAILED", "Processing failed unexpectedly; inspect worker logs"
