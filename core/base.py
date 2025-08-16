# core/base.py
"""
Base classes for AI model processors.
Provides common interface and utilities for all restoration models.
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union, Dict, Any, Optional, Tuple
import torch
import numpy as np
from PIL import Image
import cv2

logger = logging.getLogger(__name__)


class BaseProcessor(ABC):
    """Base class for all AI model processors."""

    def __init__(self, model_path: Union[str, Path], device: str = "auto"):
        """
        Initialize processor with model path and device.

        Args:
            model_path: Path to the model file
            device: Device to run on ('cuda', 'cpu', or 'auto')
        """
        self.model_path = Path(model_path)
        self.device = self._get_device(device)
        self.model = None
        self.is_loaded = False

        logger.info(f"Initialized {self.__class__.__name__} on {self.device}")

    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            return "cpu"
        return device

    @abstractmethod
    def load_model(self) -> None:
        """Load the AI model. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def process(self, input_path: Union[str, Path], **kwargs) -> np.ndarray:
        """Process input and return result. Must be implemented by subclasses."""
        pass

    def ensure_loaded(self) -> None:
        """Ensure model is loaded."""
        if not self.is_loaded:
            self.load_model()

    def unload_model(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self.is_loaded = False

            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info(f"Unloaded {self.__class__.__name__}")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        memory_info = {}

        if self.device == "cuda" and torch.cuda.is_available():
            memory_info["gpu_allocated_mb"] = (
                torch.cuda.memory_allocated() / 1024 / 1024
            )
            memory_info["gpu_reserved_mb"] = torch.cuda.memory_reserved() / 1024 / 1024
            memory_info["gpu_total_mb"] = (
                torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            )

        import psutil

        process = psutil.Process(os.getpid())
        memory_info["cpu_mb"] = process.memory_info().rss / 1024 / 1024

        return memory_info

    def process_with_timing(
        self, input_path: Union[str, Path], **kwargs
    ) -> Tuple[np.ndarray, float]:
        """Process with timing information."""
        start_time = time.time()
        result = self.process(input_path, **kwargs)
        processing_time = time.time() - start_time

        logger.info(f"Processing completed in {processing_time:.2f} seconds")
        return result, processing_time


class ImageProcessor(BaseProcessor):
    """Base class for image processing models."""

    def load_image(self, image_path: Union[str, Path]) -> np.ndarray:
        """Load image from path."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image with OpenCV (BGR)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def save_image(
        self, image: np.ndarray, output_path: Union[str, Path], quality: int = 95
    ) -> None:
        """Save image to path."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert RGB to BGR for OpenCV
        if len(image.shape) == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Save with OpenCV
        success = cv2.imwrite(
            str(output_path), image_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality]
        )
        if not success:
            raise RuntimeError(f"Failed to save image: {output_path}")

        logger.info(f"Saved image: {output_path}")

    def resize_image(self, image: np.ndarray, max_size: int) -> np.ndarray:
        """Resize image if it exceeds max_size."""
        h, w = image.shape[:2]

        if max(h, w) <= max_size:
            return image

        # Calculate new dimensions
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)

        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.info(f"Resized image from {w}x{h} to {new_w}x{new_h}")
        return resized

    def tile_process(
        self, image: np.ndarray, tile_size: int = 512, overlap: int = 32
    ) -> np.ndarray:
        """
        Process large images in tiles to manage memory usage.
        This is a template method - subclasses should override _process_tile.
        """
        h, w = image.shape[:2]

        # If image is small enough, process directly
        if h <= tile_size and w <= tile_size:
            return self._process_tile(image)

        # Calculate tile positions
        tiles = []
        for y in range(0, h, tile_size - overlap):
            for x in range(0, w, tile_size - overlap):
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                tiles.append((y, x, y_end, x_end))

        logger.info(f"Processing {len(tiles)} tiles of size {tile_size}x{tile_size}")

        # Process each tile
        processed_tiles = []
        for i, (y, x, y_end, x_end) in enumerate(tiles):
            tile = image[y:y_end, x:x_end]
            processed_tile = self._process_tile(tile)
            processed_tiles.append((processed_tile, y, x, y_end, x_end))

            if i % 10 == 0:
                logger.info(f"Processed {i+1}/{len(tiles)} tiles")

        # Reconstruct image from tiles
        return self._reconstruct_from_tiles(processed_tiles, h, w, overlap)

    def _process_tile(self, tile: np.ndarray) -> np.ndarray:
        """Process a single tile. Should be overridden by subclasses."""
        raise NotImplementedError("Subclasses must implement _process_tile")

    def _reconstruct_from_tiles(
        self, processed_tiles, original_h: int, original_w: int, overlap: int
    ) -> np.ndarray:
        """Reconstruct image from processed tiles with blending."""
        # This is a simplified reconstruction - can be improved with better blending
        scale_factor = self._get_scale_factor()
        result_h = original_h * scale_factor
        result_w = original_w * scale_factor

        result = np.zeros((result_h, result_w, 3), dtype=np.uint8)

        for processed_tile, y, x, y_end, x_end in processed_tiles:
            # Scale coordinates
            y_scaled = y * scale_factor
            x_scaled = x * scale_factor
            y_end_scaled = min(y_end * scale_factor, result_h)
            x_end_scaled = min(x_end * scale_factor, result_w)

            # Place tile
            tile_h = y_end_scaled - y_scaled
            tile_w = x_end_scaled - x_scaled

            if processed_tile.shape[0] >= tile_h and processed_tile.shape[1] >= tile_w:
                result[y_scaled:y_end_scaled, x_scaled:x_end_scaled] = processed_tile[
                    :tile_h, :tile_w
                ]

        return result

    def _get_scale_factor(self) -> int:
        """Get the scale factor for this processor. Should be overridden."""
        return 1


class VideoProcessor(BaseProcessor):
    """Base class for video processing models."""

    def extract_frames(
        self, video_path: Union[str, Path], output_dir: Union[str, Path]
    ) -> list:
        """Extract frames from video."""
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = output_dir / f"frame_{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frames.append(frame_path)
            frame_count += 1

        cap.release()
        logger.info(f"Extracted {frame_count} frames from {video_path}")
        return frames

    def create_video_from_frames(
        self, frame_paths: list, output_path: Union[str, Path], fps: float = 30.0
    ):
        """Create video from frame images."""
        if not frame_paths:
            raise ValueError("No frames provided")

        # Read first frame to get dimensions
        first_frame = cv2.imread(str(frame_paths[0]))
        h, w = first_frame.shape[:2]  # type: ignore

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            out.write(frame)

        out.release()
        logger.info(
            f"Created video: {output_path} ({len(frame_paths)} frames, {fps} FPS)"
        )


class ProcessingError(Exception):
    """Custom exception for processing errors."""

    pass


class ModelNotFoundError(Exception):
    """Exception raised when model file is not found."""

    pass
