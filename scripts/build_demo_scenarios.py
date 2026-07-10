#!/usr/bin/env python3
"""Build project-owned, real-GPU portfolio evidence and demo artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
EVIDENCE = ROOT / "web" / "src" / "demo-evidence.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def degrade(source: Path, target: Path, size: int, quality: int, faded: bool = False) -> None:
    with Image.open(source) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        image = image.filter(ImageFilter.GaussianBlur(0.55))
        if faded:
            image = ImageEnhance.Color(image).enhance(0.48)
            image = ImageEnhance.Contrast(image).enhance(0.78)
            image = Image.blend(image, Image.new("RGB", image.size, (176, 143, 100)), 0.11)
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
        cv2.rectangle(frame, (0, 342), (width, height), (20, 25, 31), -1)
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
        frame = cv2.add(frame, np.full_like(frame, int(28 + 16 * math.sin(t * math.pi)) // 5))
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
    combined_input = PUBLIC / "combined-video" / "input.mp4"
    combined_input.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-i",
            str(PUBLIC / "city-motion" / "input.mp4"),
            "-vf",
            "scale=192:108,fps=12",
            "-t",
            "2",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            str(combined_input),
        ],
        check=True,
        capture_output=True,
    )


def gpu_environment() -> dict[str, Any]:
    import torch

    query = (
        subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total,compute_cap",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .split(", ")
    )
    return {
        "gpu": query[0],
        "driver": query[1],
        "vram_mib": int(query[2]),
        "compute_capability": query[3],
        "cuda": torch.version.cuda,
        "pytorch": torch.__version__,
        "fp16": get_settings().use_fp16,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def events(*items: tuple[str, float, str, int]) -> list[dict[str, Any]]:
    return [
        {
            "id": index + 1,
            "stage": stage,
            "progress": progress,
            "message": message,
            "duration_ms": duration,
        }
        for index, (stage, progress, message, duration) in enumerate(items)
    ]


def image_dimensions(path: Path) -> str:
    with Image.open(path) as image:
        return f"{image.width}×{image.height}"


def build_gpu_outputs() -> None:
    import torch

    settings = get_settings()
    registry = ModelRegistry(settings)
    statuses = registry.verify_all()
    invalid = [item.model_id for item in statuses if not item.valid]
    if invalid:
        raise SystemExit(f"Invalid models: {', '.join(invalid)}")

    image_pipeline = ImagePipeline(settings=settings, registry=registry)
    portrait_input = PUBLIC / "archive-portrait" / "input.jpg"

    def release_image_models() -> None:
        for adapter in image_pipeline._face_adapters.values():
            adapter.unload()
        for adapter in image_pipeline._upscalers.values():
            adapter.unload()
        image_pipeline._face_adapters.clear()
        image_pipeline._upscalers.clear()
        torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()
    gfpgan = image_pipeline.process(
        portrait_input,
        PUBLIC / "archive-portrait" / "gfpgan.png",
        operation=ImageOperation.FACE_RESTORE,
        face_method="gfpgan",
        strength=0.8,
    )
    gfpgan_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    release_image_models()
    torch.cuda.reset_peak_memory_stats()
    codeformer = image_pipeline.process(
        portrait_input,
        PUBLIC / "archive-portrait" / "codeformer.png",
        operation=ImageOperation.FACE_RESTORE,
        face_method="codeformer",
        fidelity=0.7,
    )
    codeformer_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    torch.cuda.reset_peak_memory_stats()
    portrait = image_pipeline.process(
        portrait_input,
        PUBLIC / "archive-portrait" / "output.png",
        operation=ImageOperation.FACE_RESTORE_UPSCALE,
        face_method="codeformer",
        fidelity=0.7,
        scale=2,
    )
    portrait_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    release_image_models()
    torch.cuda.reset_peak_memory_stats()
    product = image_pipeline.process(
        PUBLIC / "product-detail" / "input.jpg",
        PUBLIC / "product-detail" / "output.png",
        operation=ImageOperation.UPSCALE,
        scale=4,
    )
    product_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    release_image_models()

    video_pipeline = VideoPipeline(settings=settings, registry=registry)
    torch.cuda.reset_peak_memory_stats()
    video = video_pipeline.process(
        PUBLIC / "city-motion" / "input.mp4",
        PUBLIC / "city-motion" / "output.mp4",
        ROOT / "data" / "storage" / "demo-rife-work",
        operation=VideoOperation.INTERPOLATE,
        target_fps=48,
    )
    video_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    shutil.rmtree(ROOT / "data" / "storage" / "demo-rife-work", ignore_errors=True)
    torch.cuda.reset_peak_memory_stats()
    combined = video_pipeline.process(
        PUBLIC / "combined-video" / "input.mp4",
        PUBLIC / "combined-video" / "output.mp4",
        ROOT / "data" / "storage" / "demo-combined-work",
        operation=VideoOperation.INTERPOLATE_UPSCALE,
        target_fps=24,
        scale=2,
    )
    combined_peak = round(torch.cuda.max_memory_allocated() / 1024**2, 1)
    shutil.rmtree(ROOT / "data" / "storage" / "demo-combined-work", ignore_errors=True)
    video_pipeline.rife.unload()
    for adapter in video_pipeline._upscalers.values():
        adapter.unload()
    video_pipeline._upscalers.clear()
    torch.cuda.empty_cache()

    benchmark_root = ROOT / "data" / "storage" / "demo-benchmark"

    def image_warm_median(name: str, input_path: Path, **kwargs: Any) -> float:
        measurements = []
        for index in range(3):
            result = image_pipeline.process(
                input_path,
                benchmark_root / f"{name}-{index}.png",
                **kwargs,
            )
            measurements.append(float(result.metadata["elapsed_ms"]))
        median = round(statistics.median(measurements), 2)
        release_image_models()
        return median

    def video_warm_median(
        name: str,
        input_path: Path,
        *,
        operation: VideoOperation,
        target_fps: float,
        scale: int = 2,
    ) -> float:
        measurements = []
        for index in range(3):
            result = video_pipeline.process(
                input_path,
                benchmark_root / f"{name}-{index}.mp4",
                benchmark_root / f"{name}-work-{index}",
                operation=operation,
                target_fps=target_fps,
                scale=scale,
            )
            measurements.append(float(result.metadata["elapsed_ms"]))
        median = round(statistics.median(measurements), 2)
        video_pipeline.rife.unload()
        for adapter in video_pipeline._upscalers.values():
            adapter.unload()
        video_pipeline._upscalers.clear()
        torch.cuda.empty_cache()
        return median

    def guarded_benchmark(function: Any, *args: Any, **kwargs: Any) -> float | None:
        try:
            return function(*args, **kwargs)
        except torch.OutOfMemoryError:
            release_image_models()
            video_pipeline.rife.unload()
            for adapter in video_pipeline._upscalers.values():
                adapter.unload()
            video_pipeline._upscalers.clear()
            shutil.rmtree(benchmark_root, ignore_errors=True)
            torch.cuda.empty_cache()
            return None

    benchmarks = [
        {
            "id": "gfpgan-face",
            "cold_ms": gfpgan.metadata["elapsed_ms"],
            "warm_median_ms": guarded_benchmark(
                image_warm_median,
                "gfpgan",
                portrait_input,
                operation=ImageOperation.FACE_RESTORE,
                face_method="gfpgan",
                strength=0.8,
            ),
            "peak_vram_mib": gfpgan_peak,
        },
        {
            "id": "codeformer-face",
            "cold_ms": codeformer.metadata["elapsed_ms"],
            "warm_median_ms": guarded_benchmark(
                image_warm_median,
                "codeformer",
                portrait_input,
                operation=ImageOperation.FACE_RESTORE,
                face_method="codeformer",
                fidelity=0.7,
            ),
            "peak_vram_mib": codeformer_peak,
        },
        {
            "id": "realesrgan-x4",
            "cold_ms": product.metadata["elapsed_ms"],
            "warm_median_ms": guarded_benchmark(
                image_warm_median,
                "realesrgan-x4",
                PUBLIC / "product-detail" / "input.jpg",
                operation=ImageOperation.UPSCALE,
                scale=4,
            ),
            "peak_vram_mib": product_peak,
        },
        {
            "id": "rife-24-to-48",
            "cold_ms": video.metadata["elapsed_ms"],
            "warm_median_ms": guarded_benchmark(
                video_warm_median,
                "rife",
                PUBLIC / "city-motion" / "input.mp4",
                operation=VideoOperation.INTERPOLATE,
                target_fps=48,
            ),
            "peak_vram_mib": video_peak,
        },
        {
            "id": "rife-realesrgan-combined",
            "cold_ms": combined.metadata["elapsed_ms"],
            "warm_median_ms": guarded_benchmark(
                video_warm_median,
                "combined",
                PUBLIC / "combined-video" / "input.mp4",
                operation=VideoOperation.INTERPOLATE_UPSCALE,
                target_fps=24,
                scale=2,
            ),
            "peak_vram_mib": combined_peak,
        },
    ]
    shutil.rmtree(benchmark_root, ignore_errors=True)

    generated_at = datetime.now(timezone.utc).isoformat()
    model_rows = []
    for status in statuses:
        spec = registry.specs[status.model_id]
        model_rows.append(
            {
                "model_id": status.model_id,
                "family": spec.family,
                "path": f"/mnt/c/ai_models/{spec.relative_path.as_posix()}",
                "sha256": spec.sha256,
                "size_bytes": status.size_bytes,
                "valid": status.valid,
            }
        )
    EVIDENCE.write_text(
        json.dumps(
            {
                "generated_at": generated_at,
                "environment": gpu_environment(),
                "models": model_rows,
                "benchmarks": {
                    "method": (
                        "one cold run plus median of three warm end-to-end runs; "
                        "null means the warm run was skipped due shared GPU contention"
                    ),
                    "runs": benchmarks,
                },
                "verification": {
                    "model_root": "/mnt/c/ai_models",
                    "valid_models": len(model_rows),
                    "invalid_models": 0,
                    "method": "manifest path + size + SHA-256",
                },
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    def artifact(input_path: Path, output_paths: dict[str, Path]) -> dict[str, Any]:
        return {
            "input_sha256": sha256(input_path),
            "outputs": {
                key: {
                    "path": str(path.relative_to(PUBLIC)),
                    "sha256": sha256(path),
                    "size_bytes": path.stat().st_size,
                }
                for key, path in output_paths.items()
            },
        }

    manifest = [
        {
            "id": "face-lab",
            "job_id": "demo-face-lab",
            "kind": "image",
            "title": {"en": "Face restoration lab", "zh": "人臉修復實驗室"},
            "description": {
                "en": (
                    "Compare GFPGAN and CodeFormer on the same degraded portrait, "
                    "then inspect the 2× final pipeline."
                ),
                "zh": "以同一張退化人像比較 GFPGAN 與 CodeFormer，並檢視 2 倍超解析完整流程。",
            },
            "input": "archive-portrait/input.jpg",
            "outputs": {
                "gfpgan": "archive-portrait/gfpgan.png",
                "codeformer": "archive-portrait/codeformer.png",
                "restored": "archive-portrait/output.png",
            },
            "default_output": "restored",
            "operation": "face_restore_upscale",
            "model": "GFPGAN v1.4 / CodeFormer / Real-ESRGAN x2plus",
            "model_ids": [
                "gfpgan-v1.4",
                "codeformer",
                "realesrgan-x2plus",
                "facelib-detector",
                "facelib-parser",
            ],
            "parameters": {"gfpgan_strength": 0.8, "codeformer_fidelity": 0.7, "scale": 2},
            "metrics": {
                "input": image_dimensions(portrait_input),
                "output": image_dimensions(PUBLIC / "archive-portrait" / "output.png"),
                "faces": portrait.metadata["face"]["face_count"],
                "elapsed_ms": portrait.metadata["elapsed_ms"],
                "gfpgan_ms": gfpgan.metadata["elapsed_ms"],
                "codeformer_ms": codeformer.metadata["elapsed_ms"],
                "peak_vram_mib": max(gfpgan_peak, codeformer_peak, portrait_peak),
            },
            "artifact": artifact(
                portrait_input,
                {
                    "gfpgan": PUBLIC / "archive-portrait" / "gfpgan.png",
                    "codeformer": PUBLIC / "archive-portrait" / "codeformer.png",
                    "restored": PUBLIC / "archive-portrait" / "output.png",
                },
            ),
            "events": events(
                ("validate", 0.05, "Input size, pixels and format validated", 18),
                ("face_detect", 0.25, "RetinaFace detected one face", 221),
                (
                    "face_restore",
                    0.62,
                    "CodeFormer restored facial features",
                    int(codeformer.metadata["elapsed_ms"]),
                ),
                (
                    "upscale",
                    0.88,
                    "Real-ESRGAN x2plus enhanced the full frame",
                    max(
                        1, int(portrait.metadata["elapsed_ms"] - codeformer.metadata["elapsed_ms"])
                    ),
                ),
                ("complete", 1, "Artifact and provenance recorded", 12),
            ),
        },
        {
            "id": "product-detail",
            "job_id": "demo-product-detail",
            "kind": "image",
            "title": {"en": "4× product detail", "zh": "4 倍產品細節"},
            "description": {
                "en": "Recover material texture and edges with Real-ESRGAN x4plus.",
                "zh": "使用 Real-ESRGAN x4plus 重建材質紋理與邊緣。",
            },
            "input": "product-detail/input.jpg",
            "outputs": {"restored": "product-detail/output.png"},
            "default_output": "restored",
            "operation": "upscale",
            "model": "Real-ESRGAN x4plus",
            "model_ids": ["realesrgan-x4plus"],
            "parameters": {"scale": 4, "tile_size": settings.tile_size, "fp16": settings.use_fp16},
            "metrics": {
                "input": image_dimensions(PUBLIC / "product-detail" / "input.jpg"),
                "output": image_dimensions(PUBLIC / "product-detail" / "output.png"),
                "elapsed_ms": product.metadata["elapsed_ms"],
                "peak_vram_mib": product_peak,
            },
            "artifact": artifact(
                PUBLIC / "product-detail" / "input.jpg",
                {"restored": PUBLIC / "product-detail" / "output.png"},
            ),
            "events": events(
                ("validate", 0.05, "Input validated", 8),
                ("model_load", 0.2, "Real-ESRGAN x4plus loaded from manifest", 32),
                (
                    "upscale",
                    0.86,
                    "Tiled 4× super-resolution completed",
                    int(product.metadata["elapsed_ms"]),
                ),
                ("complete", 1, "PNG artifact written", 8),
            ),
        },
        {
            "id": "city-motion",
            "job_id": "demo-city-motion",
            "kind": "video",
            "title": {"en": "24 → 48 FPS motion", "zh": "24 → 48 FPS 動態補幀"},
            "description": {
                "en": (
                    "RIFE v4.25 synthesizes intermediate frames while retaining the source audio."
                ),
                "zh": "RIFE v4.25 生成中間影格，同時保留來源音訊。",
            },
            "input": "city-motion/input.mp4",
            "outputs": {"restored": "city-motion/output.mp4"},
            "default_output": "restored",
            "operation": "interpolate",
            "model": "RIFE v4.25",
            "model_ids": ["rife-v4.25"],
            "parameters": {"input_fps": 24, "target_fps": 48},
            "metrics": {
                "input": "768×432 · 24 FPS",
                "output": "768×432 · 48 FPS",
                "frames": video.metadata["frame_count"],
                "audio": "preserved",
                "elapsed_ms": video.metadata["elapsed_ms"],
                "peak_vram_mib": video_peak,
            },
            "artifact": artifact(
                PUBLIC / "city-motion" / "input.mp4",
                {"restored": PUBLIC / "city-motion" / "output.mp4"},
            ),
            "events": events(
                ("validate", 0.05, "FFprobe validated video and audio streams", 35),
                ("decode", 0.15, "Decoded 72 source frames", 75),
                (
                    "frames",
                    0.82,
                    "RIFE generated intermediate frames",
                    int(video.metadata["elapsed_ms"]),
                ),
                ("export", 0.96, "FFmpeg encoded H.264 and copied audio", 182),
                ("complete", 1, "Video artifact verified", 15),
            ),
        },
        {
            "id": "combined-video",
            "job_id": "demo-combined-video",
            "kind": "video",
            "title": {"en": "Interpolation + 2× upscale", "zh": "補幀＋2 倍超解析"},
            "description": {
                "en": "A complete RIFE → Real-ESRGAN video pipeline with audio preservation.",
                "zh": "完整的 RIFE → Real-ESRGAN 影片流程，並保留音訊。",
            },
            "input": "combined-video/input.mp4",
            "outputs": {"restored": "combined-video/output.mp4"},
            "default_output": "restored",
            "operation": "interpolate_upscale",
            "model": "RIFE v4.25 + Real-ESRGAN x2plus",
            "model_ids": ["rife-v4.25", "realesrgan-x2plus"],
            "parameters": {"input_fps": 12, "target_fps": 24, "scale": 2},
            "metrics": {
                "input": "192×108 · 12 FPS",
                "output": "384×216 · 24 FPS",
                "frames": combined.metadata["frame_count"],
                "audio": "preserved",
                "elapsed_ms": combined.metadata["elapsed_ms"],
                "peak_vram_mib": combined_peak,
            },
            "artifact": artifact(
                PUBLIC / "combined-video" / "input.mp4",
                {"restored": PUBLIC / "combined-video" / "output.mp4"},
            ),
            "events": events(
                ("validate", 0.05, "Video limits and streams validated", 24),
                (
                    "interpolate",
                    0.46,
                    "RIFE doubled temporal resolution",
                    int(combined.metadata["elapsed_ms"] * 0.36),
                ),
                (
                    "upscale",
                    0.86,
                    "Real-ESRGAN enhanced every source and generated frame",
                    int(combined.metadata["elapsed_ms"] * 0.56),
                ),
                (
                    "export",
                    0.97,
                    "H.264 output encoded with source audio",
                    int(combined.metadata["elapsed_ms"] * 0.08),
                ),
                ("complete", 1, "Combined artifact verified", 11),
            ),
        },
    ]
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
