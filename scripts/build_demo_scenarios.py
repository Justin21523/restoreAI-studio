#!/usr/bin/env python3
"""Reproducibly degrade project-owned masters and build real GPU demo outputs."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from restorai.model_registry import ModelRegistry  # noqa: E402
from restorai.pipelines.image import ImageOperation, ImagePipeline  # noqa: E402
from restorai.pipelines.video import VideoOperation, VideoPipeline  # noqa: E402
from restorai.settings import get_settings  # noqa: E402

SOURCES = ROOT / "docs" / "demo" / "sources"
PUBLIC = ROOT / "web" / "public" / "demo" / "scenarios"
MANIFEST = ROOT / "web" / "src" / "demo-scenarios.json"


def degrade(source: Path, target: Path, size: int, quality: int, faded: bool = False) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.GaussianBlur(0.55))
        if faded:
            image = ImageEnhance.Color(image).enhance(0.48)
            image = ImageEnhance.Contrast(image).enhance(0.78)
            warm = Image.new("RGB", image.size, (176, 143, 100))
            image = Image.blend(image, warm, 0.11)
        rng = np.random.default_rng(21523)
        pixels = np.asarray(image, dtype=np.int16)
        noise = rng.normal(0, 4.2, pixels.shape).astype(np.int16)
        image = Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, "JPEG", quality=quality, optimize=True)


def city_video(target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    silent = target.with_name("silent.mp4")
    width, height, fps, seconds = 768, 432, 24, 3
    writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    rng = np.random.default_rng(5080)
    for frame_number in range(fps * seconds):
        t = frame_number / (fps * seconds)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            frame[y, :] = (max(8, 24 - y // 30), max(12, 32 - y // 24), 48 - y // 18)
        cv2.circle(frame, (620, 88), 46, (168, 138, 83), -1, cv2.LINE_AA)
        for index, x in enumerate(range(0, width, 64)):
            building_height = 120 + (index % 4) * 34
            cv2.rectangle(frame, (x, height - building_height), (x + 56, height), (18, 23, 34), -1)
            for window_y in range(height - building_height + 18, height - 25, 30):
                for window_x in range(x + 9, x + 48, 16):
                    if (window_x + window_y + index) % 3:
                        cv2.rectangle(
                            frame,
                            (window_x, window_y),
                            (window_x + 6, window_y + 10),
                            (71, 126, 151),
                            -1,
                        )
        road_y = 342
        cv2.rectangle(frame, (0, road_y), (width, height), (20, 25, 31), -1)
        for stripe in range(-1, 7):
            stripe_x = int((stripe * 150 + t * 450) % 1050) - 120
            cv2.rectangle(frame, (stripe_x, 384), (stripe_x + 78, 389), (112, 112, 105), -1)
        car_x = int(-180 + t * (width + 360))
        cv2.rectangle(frame, (car_x, 326), (car_x + 150, 378), (56, 98, 139), -1)
        cv2.rectangle(frame, (car_x + 30, 310), (car_x + 112, 336), (51, 74, 91), -1)
        cv2.circle(frame, (car_x + 32, 379), 15, (8, 10, 13), -1)
        cv2.circle(frame, (car_x + 120, 379), 15, (8, 10, 13), -1)
        for rain_x in rng.integers(0, width, 45):
            rain_y = rng.integers(0, height)
            cv2.line(
                frame,
                (int(rain_x), int(rain_y)),
                (int(rain_x - 5), int(rain_y + 17)),
                (75, 103, 122),
                1,
            )
        glow = int(28 + 16 * math.sin(t * math.pi))
        frame = cv2.add(frame, np.full_like(frame, glow // 5))
        writer.write(frame)
    writer.release()
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(silent),
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=220:sample_rate=48000:duration=3",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-shortest",
            str(target),
        ],
        check=True,
        capture_output=True,
    )
    silent.unlink(missing_ok=True)


def prepare() -> None:
    required = [SOURCES / "archive-portrait-master.png", SOURCES / "product-detail-master.png"]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing generated masters: {', '.join(map(str, missing))}")
    degrade(required[0], PUBLIC / "archive-portrait" / "input.jpg", 256, 36, faded=True)
    degrade(required[1], PUBLIC / "product-detail" / "input.jpg", 256, 42)
    city_video(PUBLIC / "city-motion" / "input.mp4")


def build_gpu_outputs() -> None:
    settings = get_settings()
    registry = ModelRegistry(settings)
    invalid = [item.model_id for item in registry.verify_all() if not item.valid]
    if invalid:
        raise SystemExit(f"Invalid models: {', '.join(invalid)}")
    image_pipeline = ImagePipeline(settings=settings, registry=registry)
    portrait = image_pipeline.process(
        PUBLIC / "archive-portrait" / "input.jpg",
        PUBLIC / "archive-portrait" / "output.png",
        operation=ImageOperation.FACE_RESTORE_UPSCALE,
        face_method="codeformer",
        fidelity=0.7,
        scale=2,
    )
    product = image_pipeline.process(
        PUBLIC / "product-detail" / "input.jpg",
        PUBLIC / "product-detail" / "output.png",
        operation=ImageOperation.UPSCALE,
        scale=4,
    )
    work = ROOT / "data" / "storage" / "demo-work"
    video_pipeline = VideoPipeline(settings=settings, registry=registry)
    video = video_pipeline.process(
        PUBLIC / "city-motion" / "input.mp4",
        PUBLIC / "city-motion" / "output.mp4",
        work,
        operation=VideoOperation.INTERPOLATE,
        target_fps=48,
    )
    shutil.rmtree(work, ignore_errors=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    manifest[0]["parameters"].update(
        {
            "codeformer_sha256": f"{registry.specs['codeformer'].sha256[:16]}…",
            "realesrgan_sha256": f"{registry.specs['realesrgan-x2plus'].sha256[:16]}…",
        }
    )
    manifest[0]["metrics"].update(
        {
            "elapsed": f"{portrait.metadata['elapsed_ms']} ms",
            "faces": portrait.metadata["face"]["face_count"],
        }
    )
    manifest[1]["metrics"].update({"elapsed": f"{product.metadata['elapsed_ms']} ms"})
    manifest[1]["parameters"].update(
        {"model_sha256": f"{registry.specs['realesrgan-x4plus'].sha256[:16]}…"}
    )
    manifest[2]["metrics"].update(
        {"elapsed": f"{video.metadata['elapsed_ms']} ms", "audio": "preserved"}
    )
    manifest[2]["parameters"].update(
        {"model_sha256": f"{registry.specs['rife-v4.25'].sha256[:16]}…"}
    )
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    if not args.prepare and not args.gpu:
        args.prepare = args.gpu = True
    if args.prepare:
        prepare()
    if args.gpu:
        build_gpu_outputs()


if __name__ == "__main__":
    main()
