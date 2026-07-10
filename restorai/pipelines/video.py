from __future__ import annotations

import json
import shutil
import subprocess
import time
from enum import Enum
from pathlib import Path
from typing import Any

import cv2

from restorai.adapters.base import CancelToken, ProcessingResult, ProgressCallback
from restorai.adapters.realesrgan import RealESRGANAdapter
from restorai.adapters.rife import RIFEAdapter
from restorai.model_registry import ModelRegistry
from restorai.settings import Settings, get_settings


class VideoOperation(str, Enum):
    INTERPOLATE = "interpolate"
    UPSCALE = "upscale"
    INTERPOLATE_UPSCALE = "interpolate_upscale"


class VideoPipeline:
    def __init__(
        self, *, settings: Settings | None = None, registry: ModelRegistry | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.registry = registry or ModelRegistry(self.settings)
        self.rife = RIFEAdapter(settings=self.settings, registry=self.registry)
        self._upscalers: dict[int, RealESRGANAdapter] = {}

    @staticmethod
    def _ffprobe(path: Path) -> dict[str, Any]:
        completed = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        data = json.loads(completed.stdout)
        video = next((item for item in data["streams"] if item["codec_type"] == "video"), None)
        if video is None:
            raise ValueError("file has no video stream")
        numerator, denominator = video.get("avg_frame_rate", "0/1").split("/")
        fps = float(numerator) / max(float(denominator), 1)
        return {
            "width": int(video["width"]),
            "height": int(video["height"]),
            "fps": fps,
            "duration": float(video.get("duration") or data["format"].get("duration") or 0),
            "has_audio": any(item["codec_type"] == "audio" for item in data["streams"]),
            "format": data,
        }

    def validate(
        self, path: Path, *, operation: VideoOperation, scale: int, target_fps: float | None
    ) -> dict[str, Any]:
        if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
            raise RuntimeError("ffmpeg and ffprobe are required")
        if not path.is_file() or path.stat().st_size > self.settings.max_video_bytes:
            raise ValueError("video is missing or exceeds MAX_VIDEO_BYTES")
        meta = self._ffprobe(path)
        if meta["duration"] <= 0 or meta["duration"] > self.settings.max_video_duration_seconds:
            raise ValueError("video duration is invalid or exceeds the configured limit")
        output_width = meta["width"] * (scale if operation != VideoOperation.INTERPOLATE else 1)
        output_height = meta["height"] * (scale if operation != VideoOperation.INTERPOLATE else 1)
        if (
            output_width > self.settings.max_video_width
            or output_height > self.settings.max_video_height
        ):
            raise ValueError("requested video output exceeds the maximum resolution")
        if target_fps is not None and not meta["fps"] <= target_fps <= self.settings.max_video_fps:
            raise ValueError("target_fps must be between input FPS and MAX_VIDEO_FPS")
        return meta

    def _upscaler(self, scale: int) -> RealESRGANAdapter:
        if scale not in self._upscalers:
            model_id = "realesrgan-x2plus" if scale == 2 else "realesrgan-x4plus"
            self._upscalers[scale] = RealESRGANAdapter(
                model_id, settings=self.settings, registry=self.registry
            )
        return self._upscalers[scale]

    def process(
        self,
        input_path: Path,
        output_path: Path,
        work_dir: Path,
        *,
        operation: VideoOperation,
        scale: int = 2,
        target_fps: float | None = None,
        progress: ProgressCallback | None = None,
        cancel: CancelToken | None = None,
    ) -> ProcessingResult:
        started = time.perf_counter()
        cancel = cancel or CancelToken()
        meta = self.validate(input_path, operation=operation, scale=scale, target_fps=target_fps)
        if operation == VideoOperation.UPSCALE:
            target_fps = meta["fps"]
        else:
            target_fps = target_fps or min(meta["fps"] * 2, self.settings.max_video_fps)
        multiplier = max(1, round(target_fps / meta["fps"]))
        target_fps = meta["fps"] * multiplier
        frames_dir = work_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(input_path))
        total_frames = max(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        ok, previous = capture.read()
        if not ok:
            capture.release()
            raise ValueError("unable to decode first video frame")
        frame_index = 0

        def write_frame(frame: Any) -> None:
            nonlocal frame_index
            cancel.raise_if_cancelled()
            if operation in (VideoOperation.UPSCALE, VideoOperation.INTERPOLATE_UPSCALE):
                result = self._upscaler(scale).process(frame, scale=scale, cancel=cancel)
                assert result.image is not None
                frame = result.image
            frame_path = frames_dir / f"frame_{frame_index:08d}.png"
            if not cv2.imwrite(str(frame_path), frame):
                raise RuntimeError(f"failed to write frame {frame_index}")
            frame_index += 1

        processed_source_frames = 0
        try:
            while True:
                ok, current = capture.read()
                write_frame(previous)
                processed_source_frames += 1
                if not ok:
                    if operation in (
                        VideoOperation.INTERPOLATE,
                        VideoOperation.INTERPOLATE_UPSCALE,
                    ):
                        for _ in range(multiplier - 1):
                            write_frame(previous)
                    break
                if operation in (VideoOperation.INTERPOLATE, VideoOperation.INTERPOLATE_UPSCALE):
                    for step in range(1, multiplier):
                        timestep = step / multiplier
                        interpolated = self.rife.interpolate(
                            previous, current, timestep=timestep, cancel=cancel
                        )
                        assert interpolated.image is not None
                        write_frame(interpolated.image)
                previous = current
                if progress:
                    progress(
                        "frames",
                        min(processed_source_frames / total_frames, 0.98),
                        f"Processed frame {processed_source_frames}/{total_frames}",
                    )
        finally:
            capture.release()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-framerate",
            f"{target_fps:.6f}",
            "-i",
            str(frames_dir / "frame_%08d.png"),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-shortest",
            str(output_path),
        ]
        subprocess.run(command, check=True, capture_output=True, stdin=subprocess.DEVNULL)
        output_meta = self._ffprobe(output_path)
        if progress:
            progress("export", 1.0, "Video artifact written")
        return ProcessingResult(
            output_path=output_path,
            metadata={
                "operation": operation.value,
                "input": {key: value for key, value in meta.items() if key != "format"},
                "output": {key: value for key, value in output_meta.items() if key != "format"},
                "frame_count": frame_index,
                "scale": scale,
                "target_fps": target_fps,
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            },
        )
