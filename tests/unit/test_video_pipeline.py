from __future__ import annotations

from pathlib import Path

import pytest

from restorai.adapters.rife import RIFEAdapter
from restorai.pipelines.video import VideoOperation, VideoPipeline
from restorai.settings import Settings


def _pipeline(tmp_path: Path) -> VideoPipeline:
    return VideoPipeline(
        settings=Settings(
            model_root=Path("/mnt/c/ai_models"),
            max_video_bytes=1000,
            max_video_duration_seconds=10,
            max_video_width=1920,
            max_video_height=1080,
            max_video_fps=60,
            storage_root=tmp_path,
        )
    )


def test_video_validation_rejects_oversized_output(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr("restorai.pipelines.video.shutil.which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        pipeline,
        "_ffprobe",
        lambda _path: {
            "width": 1280,
            "height": 720,
            "fps": 24.0,
            "duration": 1.0,
            "has_audio": True,
            "format": {},
        },
    )
    with pytest.raises(ValueError, match="maximum resolution"):
        pipeline.validate(source, operation=VideoOperation.UPSCALE, scale=2, target_fps=None)


def test_video_validation_rejects_frame_rate_below_input(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "input.mp4"
    source.write_bytes(b"video")
    pipeline = _pipeline(tmp_path)
    monkeypatch.setattr("restorai.pipelines.video.shutil.which", lambda _name: "/usr/bin/tool")
    monkeypatch.setattr(
        pipeline,
        "_ffprobe",
        lambda _path: {
            "width": 320,
            "height": 180,
            "fps": 30.0,
            "duration": 1.0,
            "has_audio": False,
            "format": {},
        },
    )
    with pytest.raises(ValueError, match="between input FPS"):
        pipeline.validate(source, operation=VideoOperation.INTERPOLATE, scale=2, target_fps=24)


def test_rife_tensor_padding_supports_non_power_of_two_width() -> None:
    assert RIFEAdapter._padded_shape(54, 96) == (64, 128)
