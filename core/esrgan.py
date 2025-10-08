"""
Real-ESRGAN Super-Resolution Processor for RestorAI MVP.
Lightweight wrapper around Real-ESRGAN for image upscaling.

Why: Uses centralized AI warehouse (settings.PATHS["models"]) instead of
project-local ./data/models to enable model sharing across projects.
"""

import logging
import time
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
import torch
import cv2
from PIL import Image

from core.base import ImageProcessor, ProcessingError, ModelNotFoundError

logger = logging.getLogger(__name__)


class ESRGANProcessor(ImageProcessor):
    """Real-ESRGAN super-resolution processor."""

    def __init__(
        self, model_path: Union[str, Path] = None, device: str = "auto", **kwargs  # type: ignore
    ):
        """
        Initialize Real-ESRGAN processor.

        Args:
            model_path: Path to Real-ESRGAN model file (auto-detected if None)
            device: Device to run on ('cuda', 'cpu', or 'auto')
            **kwargs: Additional arguments
        """
        # Get model path from centralized warehouse if not provided
        if model_path is None:
            from utils.config import Config

            config = Config()
            model_path = config.PATHS["models"] / "esrgan" / "RealESRGAN_x4plus.pth"

        super().__init__(model_path, device)

        # Processing parameters
        self.tile_size = kwargs.get("tile_size", 512)
        self.use_fp16 = kwargs.get("use_fp16", True) and self.device == "cuda"

        # Model components
        self.upsampler = None

        logger.info(
            f"Initialized ESRGANProcessor (device: {self.device}, fp16: {self.use_fp16})"
        )
        logger.info(f"Model path: {self.model_path}")

    def load_model(self) -> None:
        """Load Real-ESRGAN model from centralized warehouse."""
        if not self.model_path.exists():
            raise ModelNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Download with: python scripts/download_models.py --essential"
            )

        try:
            # Try to import Real-ESRGAN
            try:
                from realesrgan import RealESRGANer
                from realesrgan.archs.srvgg_arch import SRVGGNetCompact
                from basicsr.archs.rrdbnet_arch import RRDBNet
            except ImportError:
                # Fallback to manual implementation if realesrgan package not available
                return self._load_model_manual()

            # Determine model architecture
            model_name = self.model_path.name.lower()

            if "x4plus" in model_name:
                # RRDBNet architecture for x4plus models
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )
            elif "anime" in model_name:
                # SRVGGNetCompact for anime models
                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=16,
                    upscale=4,
                    act_type="prelu",
                )
            else:
                # Default to RRDBNet
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=23,
                    num_grow_ch=32,
                    scale=4,
                )

            # Create upsampler
            self.upsampler = RealESRGANer(
                scale=4,
                model_path=str(self.model_path),
                model=model,
                tile=self.tile_size,
                tile_pad=10,
                pre_pad=0,
                half=self.use_fp16,
                device=self.device,
            )

            self.is_loaded = True
            logger.info(f"✅ Loaded Real-ESRGAN model: {self.model_path.name}")

        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            raise ProcessingError(f"Model loading failed: {e}")

    def _load_model_manual(self) -> None:
        """Manual model loading when realesrgan package is not available."""
        try:
            import torch

            # Load state dict
            state_dict = torch.load(str(self.model_path), map_location=self.device)

            # Create a simple upsampler using PyTorch
            from core.simple_esrgan import SimpleESRGAN

            self.upsampler = SimpleESRGAN(
                state_dict=state_dict,
                device=self.device,
                tile_size=self.tile_size,
                use_fp16=self.use_fp16,
            )

            self.is_loaded = True
            logger.info(
                f"✅ Loaded Real-ESRGAN model (manual mode): {self.model_path.name}"
            )

        except Exception as e:
            raise ProcessingError(f"Manual model loading failed: {e}")

    def process(
        self, input_image: Union[str, Path, np.ndarray], scale: int = 4, **kwargs
    ) -> np.ndarray:
        """
        Process image with Real-ESRGAN.

        Args:
            input_image: Input image (path or numpy array)
            scale: Upscale factor (2, 4, 8)
            **kwargs: Additional parameters

        Returns:
            Enhanced image as numpy array
        """
        self.ensure_loaded()

        # Load image if path provided
        if isinstance(input_image, (str, Path)):
            image = self.load_image(input_image)
        else:
            image = input_image

        if image is None:
            raise ProcessingError("Failed to load input image")

        # Validate image
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ProcessingError("Image must be RGB format")

        # Check image size
        h, w = image.shape[:2]
        max_size = kwargs.get("max_size", 2048)

        if max(h, w) > max_size:
            logger.warning(
                f"Image size {w}x{h} exceeds max size {max_size}, resizing..."
            )
            image = self.resize_image(image, max_size)

        try:
            # Process image
            logger.info(f"Processing image {w}x{h} with scale {scale}x")
            start_time = time.time()

            if hasattr(self.upsampler, "enhance"):
                # Real-ESRGAN package method
                output, _ = self.upsampler.enhance(image, outscale=scale)
            else:
                # Manual implementation method
                output = self.upsampler.enhance(image, scale=scale)

            processing_time = time.time() - start_time

            if output is None:
                raise ProcessingError("Model returned None output")

            # Ensure output is in correct format
            if isinstance(output, torch.Tensor):
                output = output.cpu().numpy()

            output = np.clip(output, 0, 255).astype(np.uint8)

            logger.info(
                f"✅ Processing completed in {processing_time:.2f}s, output size: {output.shape}"
            )
            return output

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Real-ESRGAN processing failed: {e}")

    def _process_tile(self, tile: np.ndarray) -> np.ndarray:
        """Process a single tile for memory-efficient processing."""
        if hasattr(self.upsampler, "enhance"):
            output, _ = self.upsampler.enhance(tile, outscale=4)
        else:
            output = self.upsampler.enhance(tile, scale=4)
        return output

    def _get_scale_factor(self) -> int:
        """Get the scale factor for this processor."""
        return 4  # Real-ESRGAN typically uses 4x upscaling

    def enhance_image(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        scale: int = 4,
        quality: int = 95,
    ) -> dict:
        """
        High-level interface for image enhancement.

        Args:
            input_path: Input image path
            output_path: Output image path
            scale: Upscale factor
            quality: JPEG quality for output

        Returns:
            Processing information dictionary
        """
        try:
            # Load and process image
            start_time = time.time()
            result = self.process(input_path, scale=scale)
            processing_time = time.time() - start_time

            # Save result
            self.save_image(result, output_path, quality=quality)

            # Get file sizes
            input_size = Path(input_path).stat().st_size
            output_size = Path(output_path).stat().st_size

            # Calculate metrics
            input_image = self.load_image(input_path)
            input_h, input_w = input_image.shape[:2]
            output_h, output_w = result.shape[:2]

            return {
                "status": "success",
                "processing_time": processing_time,
                "input_size": {"width": input_w, "height": input_h},
                "output_size": {"width": output_w, "height": output_h},
                "scale_factor": output_w / input_w,
                "file_size_input": input_size,
                "file_size_output": output_size,
                "compression_ratio": output_size / input_size,
                "memory_usage": self.get_memory_usage(),
            }

        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "processing_time": (
                    time.time() - start_time if "start_time" in locals() else 0
                ),
            }

    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        return {
            "name": "Real-ESRGAN",
            "model_file": self.model_path.name,
            "device": self.device,
            "use_fp16": self.use_fp16,
            "tile_size": self.tile_size,
            "is_loaded": self.is_loaded,
            "scale_factor": 4,
            "supported_formats": ["jpg", "jpeg", "png", "bmp", "tiff", "webp"],
        }
