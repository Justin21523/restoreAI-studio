from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any

import httpx


def submit_directory(
    directory: Path,
    *,
    operation: str,
    pattern: str = "*",
    api_base: str = "http://127.0.0.1:8000/api/v1",
) -> dict[str, Any]:
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    paths = [path for path in sorted(directory.glob(pattern)) if path.is_file()]
    if not paths:
        raise ValueError("No matching files")
    is_video = operation in {"interpolate", "interpolate_upscale"}
    endpoint = "batches/videos" if is_video else "batches/images"
    handles = [path.open("rb") for path in paths]
    try:
        files = [
            ("files", (path.name, handle, mimetypes.guess_type(path.name)[0]))
            for path, handle in zip(paths, handles, strict=True)
        ]
        response = httpx.post(
            f"{api_base}/{endpoint}",
            files=files,
            data={"operation": operation},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    finally:
        for handle in handles:
            handle.close()
