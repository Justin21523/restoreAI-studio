from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from restorai.adapters.base import CancelToken, ModelAdapter, ProcessingResult, ProgressCallback
from restorai.cuda_runtime import preload_nvrtc_builtins
from restorai.model_registry import ModelRegistry
from restorai.settings import Settings, get_settings


class RealESRGANAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str = "realesrgan-x4plus",
        *,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(self.settings)
        self.model_id = model_id
        self.upsampler: Any | None = None
        self.device = "cpu"

    def load(self) -> dict[str, Any]:
        if self.upsampler is not None:
            return self.health()
        preload_nvrtc_builtins()
        import torch
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        spec = self.registry.specs[self.model_id]
        scale = int(spec.native_scale or 4)
        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
            scale=scale,
        )
        if self.settings.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.settings.device
        self.upsampler = RealESRGANer(
            scale=scale,
            model_path=str(self.registry.require(self.model_id)),
            model=model,
            tile=self.settings.tile_size,
            tile_pad=10,
            pre_pad=0,
            half=self.settings.use_fp16 and self.device == "cuda",
            device=self.device,
        )
        return self.health()

    def process(
        self,
        image_bgr: np.ndarray,
        *,
        scale: int,
        progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> ProcessingResult:
        if scale not in (2, 4):
            raise ValueError("scale must be 2 or 4")
        if image_bgr.ndim != 3 or image_bgr.shape[2] not in (3, 4):
            raise ValueError("input must be a BGR/BGRA image")
        cancel = cancel or CancelToken()
        cancel.raise_if_cancelled()
        if progress:
            progress("model_load", 0.05, f"Loading {self.model_id}")
        self.load()
        if progress:
            progress("upscale", 0.2, f"Upscaling {scale}x")
        assert self.upsampler is not None
        alpha = image_bgr[:, :, 3] if image_bgr.shape[2] == 4 else None
        source = image_bgr[:, :, :3]
        output, _ = self.upsampler.enhance(source, outscale=scale)
        cancel.raise_if_cancelled()
        if alpha is not None:
            alpha = cv2.resize(
                alpha,
                (output.shape[1], output.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
            output = np.dstack((output, alpha))
        if progress:
            progress("upscale", 1.0, "Upscale complete")
        return ProcessingResult(
            image=output,
            metadata={"model_id": self.model_id, "scale": scale, "device": self.device},
        )

    def unload(self) -> None:
        self.upsampler = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def health(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self.upsampler is not None,
            "device": self.device,
            "path": str(self.registry.resolve(self.model_id)),
        }
