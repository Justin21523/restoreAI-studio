from __future__ import annotations

import time
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

from restorai.adapters.base import CancelToken, ProcessingResult, ProgressCallback
from restorai.adapters.face import FaceRestoreAdapter
from restorai.adapters.realesrgan import RealESRGANAdapter
from restorai.model_registry import ModelRegistry
from restorai.settings import Settings, get_settings


class ImageOperation(str, Enum):
    UPSCALE = "upscale"
    FACE_RESTORE = "face_restore"
    FACE_RESTORE_UPSCALE = "face_restore_upscale"


class ImagePipeline:
    def __init__(
        self, *, settings: Settings | None = None, registry: ModelRegistry | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(self.settings)
        self._upscalers: dict[int, RealESRGANAdapter] = {}
        self._face_adapters: dict[str, FaceRestoreAdapter] = {}

    def validate(self, path: Path, *, scale: int = 1) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError("input image does not exist")
        if path.stat().st_size > self.settings.max_image_bytes:
            raise ValueError("image exceeds MAX_IMAGE_BYTES")
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                width, height = ImageOps.exif_transpose(image).size
                image_format = image.format
        except Exception as exc:
            raise ValueError("invalid or unsupported image") from exc
        pixels = width * height
        if pixels > self.settings.max_image_pixels:
            raise ValueError("image exceeds MAX_IMAGE_PIXELS")
        if pixels * scale * scale > self.settings.max_output_pixels:
            raise ValueError("requested output exceeds MAX_OUTPUT_PIXELS")
        return {
            "width": width,
            "height": height,
            "pixels": pixels,
            "format": image_format,
            "size_bytes": path.stat().st_size,
        }

    @staticmethod
    def _read_oriented(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode == "RGBA":
                rgba = np.asarray(image.convert("RGBA"))
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
            rgb = np.asarray(image.convert("RGB"))
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _upscaler(self, scale: int) -> RealESRGANAdapter:
        if scale not in self._upscalers:
            model_id = "realesrgan-x2plus" if scale == 2 else "realesrgan-x4plus"
            self._upscalers[scale] = RealESRGANAdapter(
                model_id, settings=self.settings, registry=self.registry
            )
        return self._upscalers[scale]

    def _face(self, method: str) -> FaceRestoreAdapter:
        if method not in ("gfpgan", "codeformer"):
            raise ValueError("face method must be gfpgan or codeformer")
        if method not in self._face_adapters:
            self._face_adapters[method] = FaceRestoreAdapter(
                method, settings=self.settings, registry=self.registry
            )
        return self._face_adapters[method]

    def process(
        self,
        input_path: Path,
        output_path: Path,
        *,
        operation: ImageOperation,
        scale: int = 2,
        face_method: str = "codeformer",
        strength: float = 0.8,
        fidelity: float = 0.7,
        progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> ProcessingResult:
        started = time.perf_counter()
        cancel = cancel or CancelToken()
        effective_scale = (
            scale
            if operation
            in (
                ImageOperation.UPSCALE,
                ImageOperation.FACE_RESTORE_UPSCALE,
            )
            else 1
        )
        input_meta = self.validate(input_path, scale=effective_scale)
        if progress:
            progress("validate", 0.03, "Input validated")
        image = self._read_oriented(input_path)
        metadata: dict[str, Any] = {"operation": operation.value, "input": input_meta}
        if operation in (ImageOperation.FACE_RESTORE, ImageOperation.FACE_RESTORE_UPSCALE):
            restored = self._face(face_method).process(
                image,
                strength=strength,
                fidelity=fidelity,
                progress=progress,
                cancel=cancel,
            )
            assert restored.image is not None
            image = restored.image
            metadata["face"] = restored.metadata
        if operation in (ImageOperation.UPSCALE, ImageOperation.FACE_RESTORE_UPSCALE):
            upscaled = self._upscaler(scale).process(
                image, scale=scale, progress=progress, cancel=cancel
            )
            assert upscaled.image is not None
            image = upscaled.image
            metadata["upscale"] = upscaled.metadata
        cancel.raise_if_cancelled()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        extension = output_path.suffix.lower()
        if extension not in (".png", ".jpg", ".jpeg", ".webp"):
            output_path = output_path.with_suffix(".png")
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"failed to write output image: {output_path}")
        metadata.update(
            {
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "output_width": int(image.shape[1]),
                "output_height": int(image.shape[0]),
                "output_size_bytes": output_path.stat().st_size,
            }
        )
        if progress:
            progress("export", 1.0, "Artifact written")
        return ProcessingResult(image=image, output_path=output_path, metadata=metadata)
