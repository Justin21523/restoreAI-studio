from __future__ import annotations

import errno
import subprocess

import pytest

from restorai.model_registry import ModelRegistryError
from restorai.services.errors import classify_processing_error


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (ModelRegistryError("missing"), "MODEL_UNAVAILABLE"),
        (RuntimeError("CUDA out of memory"), "CUDA_OUT_OF_MEMORY"),
        (OSError(errno.ENOSPC, "disk full"), "STORAGE_FULL"),
        (ValueError("invalid video duration"), "INVALID_VIDEO"),
        (ValueError("requested output exceeds MAX_OUTPUT_PIXELS"), "OUTPUT_LIMIT_EXCEEDED"),
        (subprocess.CalledProcessError(1, ["ffmpeg"]), "FFMPEG_FAILED"),
    ],
)
def test_error_classifier(error: Exception, code: str) -> None:
    assert classify_processing_error(error)[0] == code
