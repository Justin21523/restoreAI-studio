from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from restorai.adapters.base import CancelToken, ModelAdapter, ProcessingResult, ProgressCallback
from restorai.cuda_runtime import preload_nvrtc_builtins
from restorai.model_registry import ModelRegistry
from restorai.settings import Settings, get_settings


class RIFEAdapter(ModelAdapter):
    model_id = "rife-v4.25"

    def __init__(
        self, *, settings: Settings | None = None, registry: ModelRegistry | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(self.settings)
        self.model: Any | None = None
        self.device: Any = None

    def load(self) -> dict[str, Any]:
        if self.model is not None:
            return self.health()
        preload_nvrtc_builtins()
        import torch

        from restorai.vendor.rife.train_log.IFNet_HDv3 import IFNet

        self.device = torch.device(
            "cuda"
            if self.settings.device in ("auto", "cuda") and torch.cuda.is_available()
            else "cpu"
        )
        self.model = IFNet().to(self.device)
        checkpoint = torch.load(
            self.registry.require(self.model_id), map_location="cpu", weights_only=True
        )
        state = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        return self.health()

    @staticmethod
    def _tensor(frame_bgr: np.ndarray, device: Any) -> tuple[Any, tuple[int, int]]:
        import torch

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        height, width = rgb.shape[:2]
        # RIFE v4.25's five-scale cascade requires dimensions divisible by 64;
        # padding only to 32 breaks inputs such as 96×54 at later flow stages.
        padded_h = ((height + 63) // 64) * 64
        padded_w = ((width + 63) // 64) * 64
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).float().div_(255).unsqueeze(0)
        tensor = torch.nn.functional.pad(tensor, (0, padded_w - width, 0, padded_h - height))
        return tensor.to(device), (height, width)

    def interpolate(
        self,
        first_bgr: np.ndarray,
        second_bgr: np.ndarray,
        *,
        timestep: float = 0.5,
        progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> ProcessingResult:
        import torch

        if not 0 < timestep < 1:
            raise ValueError("timestep must be between 0 and 1")
        cancel = cancel or CancelToken()
        cancel.raise_if_cancelled()
        self.load()
        assert self.model is not None
        first, shape = self._tensor(first_bgr, self.device)
        second, _ = self._tensor(second_bgr, self.device)
        with torch.inference_mode():
            _, _, merged = self.model(torch.cat((first, second), 1), timestep, [16, 8, 4, 2, 1])
        output = merged[-1][0, :, : shape[0], : shape[1]].clamp(0, 1)
        rgb = output.mul(255).byte().cpu().numpy().transpose(1, 2, 0)
        if progress:
            progress("interpolate", 1.0, "Frame interpolation complete")
        return ProcessingResult(
            image=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            metadata={"model_id": self.model_id, "timestep": timestep},
        )

    def unload(self) -> None:
        self.model = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def health(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "loaded": self.model is not None,
            "device": str(self.device) if self.device is not None else None,
            "path": str(self.registry.resolve(self.model_id)),
        }
