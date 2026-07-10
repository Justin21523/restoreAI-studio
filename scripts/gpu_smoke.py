#!/usr/bin/env python3
"""Manual release smoke for every real GPU pipeline, including combined video."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restorai.model_registry import ModelRegistry  # noqa: E402
from restorai.pipelines.image import ImageOperation, ImagePipeline  # noqa: E402
from restorai.pipelines.video import VideoOperation, VideoPipeline  # noqa: E402
from restorai.settings import get_settings  # noqa: E402


def probe(path: Path) -> dict[str, object]:
    completed = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def run_with_logs(function, *args, **kwargs):  # type: ignore[no-untyped-def]
    with redirect_stdout(sys.stderr):
        return function(*args, **kwargs)


def main() -> None:
    settings = get_settings()
    registry = ModelRegistry(settings)
    invalid = [item.model_id for item in registry.verify_all() if not item.valid]
    if invalid:
        raise SystemExit(f"Invalid models: {', '.join(invalid)}")
    portrait = ROOT / "web/public/demo/scenarios/archive-portrait/input.jpg"
    product = ROOT / "web/public/demo/scenarios/product-detail/input.jpg"
    city = ROOT / "web/public/demo/scenarios/city-motion/input.mp4"
    results: dict[str, object] = {}
    with tempfile.TemporaryDirectory(prefix="restorai-gpu-smoke-") as temporary:
        temp = Path(temporary)
        image = ImagePipeline(settings=settings, registry=registry)
        for method in ("gfpgan", "codeformer"):
            result = run_with_logs(
                image.process,
                portrait,
                temp / f"{method}.png",
                operation=ImageOperation.FACE_RESTORE,
                face_method=method,
            )
            results[method] = result.metadata
        upscale = run_with_logs(
            image.process,
            product,
            temp / "upscale.png",
            operation=ImageOperation.UPSCALE,
            scale=2,
        )
        results["realesrgan"] = upscale.metadata
        short = temp / "short.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-y",
                "-i",
                str(city),
                "-vf",
                "scale=96:54,fps=5",
                "-t",
                "1",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                str(short),
            ],
            check=True,
            capture_output=True,
        )
        video = run_with_logs(
            VideoPipeline(settings=settings, registry=registry).process,
            short,
            temp / "combined.mp4",
            temp / "work",
            operation=VideoOperation.INTERPOLATE_UPSCALE,
            target_fps=10,
            scale=2,
        )
        video_probe = probe(temp / "combined.mp4")
        streams = video_probe["streams"]
        assert any(item["codec_type"] == "audio" for item in streams)
        assert (
            next(item for item in streams if item["codec_type"] == "video")["avg_frame_rate"]
            == "10/1"
        )
        results["rife_realesrgan_combined"] = video.metadata
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
