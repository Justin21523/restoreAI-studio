from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np

from restorai.adapters.base import CancelToken, ModelAdapter, ProcessingResult, ProgressCallback
from restorai.cuda_runtime import preload_nvrtc_builtins
from restorai.model_registry import ModelRegistry
from restorai.settings import Settings, get_settings


class FaceRestoreAdapter(ModelAdapter):
    def __init__(
        self,
        method: Literal["gfpgan", "codeformer"],
        *,
        settings: Settings | None = None,
        registry: ModelRegistry | None = None,
        background_upsampler: Any | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(self.settings)
        self.method = method
        self.model_id = "gfpgan-v1.4" if method == "gfpgan" else "codeformer"
        self.background_upsampler = background_upsampler
        self.model: Any | None = None
        self.face_helper: Any | None = None
        self.device: Any = None

    def load(self) -> dict[str, Any]:
        if self.model is not None:
            return self.health()
        preload_nvrtc_builtins()
        import torch
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper

        self.device = torch.device(
            "cuda"
            if self.settings.device in ("auto", "cuda") and torch.cuda.is_available()
            else "cpu"
        )
        if self.method == "gfpgan":
            from restorai.vendor.gfpgan.gfpganv1_clean_arch import GFPGANv1Clean

            self.model = GFPGANv1Clean(
                out_size=512,
                num_style_feat=512,
                channel_multiplier=2,
                decoder_load_path=None,
                fix_decoder=False,
                num_mlp=8,
                input_is_latent=True,
                different_w=True,
                narrow=1,
                sft_half=True,
            )
        else:
            from restorai.vendor.codeformer.codeformer_arch import CodeFormer

            self.model = CodeFormer(
                dim_embd=512,
                codebook_size=1024,
                n_head=8,
                n_layers=9,
                connect_list=["32", "64", "128", "256"],
            )
        checkpoint = torch.load(
            self.registry.require(self.model_id), map_location="cpu", weights_only=True
        )
        state = checkpoint.get("params_ema", checkpoint.get("params", checkpoint))
        self.model.load_state_dict(state, strict=True)
        self.model.eval().to(self.device)
        facelib_root = self.settings.model_root / "vision/face_restore/facelib"
        self.face_helper = FaceRestoreHelper(
            1,
            face_size=512,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            save_ext="png",
            use_parse=True,
            device=self.device,
            model_rootpath=str(facelib_root),
        )
        return self.health()

    def process(
        self,
        image_bgr: np.ndarray,
        *,
        strength: float = 0.8,
        fidelity: float = 0.7,
        only_center_face: bool = False,
        progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> ProcessingResult:
        import torch
        from basicsr.utils import img2tensor, tensor2img
        from torchvision.transforms.functional import normalize

        if not 0 <= strength <= 1 or not 0 <= fidelity <= 1:
            raise ValueError("strength and fidelity must be between 0 and 1")
        cancel = cancel or CancelToken()
        if progress:
            progress("model_load", 0.05, f"Loading {self.model_id}")
        self.load()
        assert self.face_helper is not None and self.model is not None
        helper = self.face_helper
        helper.clean_all()
        helper.read_image(image_bgr[:, :, :3])
        face_count = helper.get_face_landmarks_5(
            only_center_face=only_center_face, resize=640, eye_dist_threshold=5
        )
        if progress:
            progress("face_detect", 0.25, f"Detected {face_count} face(s)")
        helper.align_warp_face()
        for index, cropped_face in enumerate(helper.cropped_faces):
            cancel.raise_if_cancelled()
            tensor = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
            normalize(tensor, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
            tensor = tensor.unsqueeze(0).to(self.device)
            with torch.inference_mode():
                if self.method == "gfpgan":
                    output = self.model(tensor, return_rgb=False, weight=strength)[0]
                else:
                    output = self.model(tensor, w=fidelity, adain=True)[0]
            restored = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
            restored = restored.astype(np.uint8)
            if self.method == "gfpgan" and strength < 1:
                restored = cv2.addWeighted(restored, strength, cropped_face, 1 - strength, 0)
            helper.add_restored_face(restored)
            if progress:
                progress(
                    "face_restore",
                    0.3 + 0.5 * ((index + 1) / max(face_count, 1)),
                    f"Restored face {index + 1}/{face_count}",
                )
        if face_count == 0:
            result = image_bgr[:, :, :3].copy()
        else:
            helper.get_inverse_affine(None)
            result = helper.paste_faces_to_input_image()
        if progress:
            progress("face_restore", 1.0, "Face restoration complete")
        return ProcessingResult(
            image=result,
            metadata={
                "model_id": self.model_id,
                "face_count": int(face_count),
                "strength": strength,
                "fidelity": fidelity,
                "device": str(self.device),
            },
        )

    def unload(self) -> None:
        self.model = None
        self.face_helper = None
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
