from __future__ import annotations

import json
import mimetypes
import shutil
import subprocess
from pathlib import Path
from typing import Any

from redis import Redis
from rq import Queue, Worker

from restorai.adapters.base import CancelToken, ProcessingCancelled
from restorai.db import get_engine, require_database_schema, session_factory
from restorai.model_registry import ModelRegistry
from restorai.orm import Artifact, Batch, Job, utcnow
from restorai.pipelines.image import ImageOperation, ImagePipeline
from restorai.pipelines.video import VideoOperation, VideoPipeline
from restorai.services.errors import classify_processing_error
from restorai.services.jobs import JobService
from restorai.services.storage import StorageService
from restorai.settings import get_settings


def _model_snapshot(registry: ModelRegistry, model_ids: list[str]) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for model_id in model_ids:
        spec = registry.specs[model_id]
        path = registry.require(model_id)
        snapshot[model_id] = {
            "path": str(path.relative_to(registry.root)),
            "sha256": spec.sha256,
            "source": spec.source,
            "source_revision": spec.source_revision,
        }
    return snapshot


def _update_batch(session, batch_id: str | None) -> None:  # type: ignore[no-untyped-def]
    if not batch_id:
        return
    batch = session.get(Batch, batch_id)
    if not batch:
        return
    statuses = [job.status for job in batch.jobs]
    batch.succeeded_items = statuses.count("succeeded")
    batch.failed_items = statuses.count("failed") + statuses.count("cancelled")
    if statuses and all(status == "cancelled" for status in statuses):
        batch.status = "cancelled"
    elif all(status in {"succeeded", "failed", "cancelled"} for status in statuses):
        batch.status = "succeeded" if batch.failed_items == 0 else "partial_failure"
    elif any(status == "running" for status in statuses):
        batch.status = "running"


def execute_job(job_id: str) -> None:
    settings = get_settings()
    factory = session_factory()
    redis = Redis.from_url(settings.redis_url)
    storage = StorageService(settings)
    registry = ModelRegistry(settings)
    work_dir = storage.job_work_dir(job_id)
    with factory() as session:
        job = session.get(Job, job_id)
        if not job or job.status == "cancelled":
            return
        service = JobService(session, settings)
        job.status = "running"
        job.started_at = utcnow()
        batch_id = job.batch_id
        service.add_event(job, "started", "starting", 0.01, "GPU worker started")
        session.commit()

        def cancelled() -> bool:
            return bool(redis.exists(f"restorai:cancel:{job_id}"))

        token = CancelToken(cancelled)

        def progress(stage: str, value: float, message: str) -> None:
            current = session.get(Job, job_id)
            if current is None:
                return
            service.add_event(current, "progress", stage, value, message)
            session.commit()

        failure: Exception | None = None
        try:
            input_path = Path(job.input_path)
            if job.kind == "image":
                suffix = ".png"
                output_path = storage.artifact_path(job.id, suffix)
                pipeline = ImagePipeline(settings=settings, registry=registry)
                result = pipeline.process(
                    input_path,
                    output_path,
                    operation=ImageOperation(job.operation),
                    progress=progress,
                    cancel=token,
                    **job.parameters,
                )
                model_ids = []
                if job.operation in {"upscale", "face_restore_upscale"}:
                    model_ids.append(
                        "realesrgan-x2plus"
                        if int(job.parameters.get("scale", 2)) == 2
                        else "realesrgan-x4plus"
                    )
                if job.operation in {"face_restore", "face_restore_upscale"}:
                    model_ids.extend(
                        [
                            "gfpgan-v1.4"
                            if job.parameters.get("face_method") == "gfpgan"
                            else "codeformer",
                            "facelib-detector",
                            "facelib-parser",
                        ]
                    )
            else:
                suffix = ".mp4"
                output_path = storage.artifact_path(job.id, suffix)
                pipeline = VideoPipeline(settings=settings, registry=registry)
                result = pipeline.process(
                    input_path,
                    output_path,
                    work_dir,
                    operation=VideoOperation(job.operation),
                    progress=progress,
                    cancel=token,
                    **job.parameters,
                )
                model_ids = []
                if job.operation in {"interpolate", "interpolate_upscale"}:
                    model_ids.append("rife-v4.25")
                if job.operation in {"upscale", "interpolate_upscale"}:
                    model_ids.append(
                        "realesrgan-x2plus"
                        if int(job.parameters.get("scale", 2)) == 2
                        else "realesrgan-x4plus"
                    )
            assert result.output_path is not None
            artifact = Artifact(
                job_id=job.id,
                input_sha256=storage.sha256(input_path),
                output_sha256=storage.sha256(result.output_path),
                output_path=str(result.output_path),
                mime_type=mimetypes.guess_type(result.output_path.name)[0]
                or "application/octet-stream",
                size_bytes=result.output_path.stat().st_size,
                metadata_json=result.metadata,
                model_snapshot=_model_snapshot(registry, model_ids),
                expires_at=JobService.expiry_time(settings),
            )
            session.add(artifact)
            job.status = "succeeded"
            job.stage = "complete"
            job.progress = 1
            job.finished_at = utcnow()
            service.add_event(job, "completed", "complete", 1, "Job completed")
        except ProcessingCancelled:
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = utcnow()
            service.add_event(job, "cancelled", "cancelled", job.progress, "Job cancelled")
        except Exception as exc:
            failure = exc
            # A progress callback may itself have failed while flushing. Reset the
            # transaction before persisting the product-level failure state.
            session.rollback()
            job = session.get(Job, job_id)
            if job is None:
                raise
            job.status = "failed"
            job.stage = "failed"
            error_code, safe_message = classify_processing_error(exc)
            job.error_code = error_code
            job.error_message = safe_message
            job.finished_at = utcnow()
            service.add_event(
                job,
                "failed",
                "failed",
                job.progress,
                "Processing failed",
                {"error_code": error_code},
            )
        finally:
            _update_batch(session, batch_id)
            session.commit()
            shutil.rmtree(work_dir, ignore_errors=True)
            redis.delete(f"restorai:cancel:{job_id}")
        if failure is not None:
            # PostgreSQL remains the product source of truth, while re-raising also
            # keeps RQ's failed-job registry and operations tooling accurate.
            raise failure


def run() -> None:
    settings = get_settings()
    settings.ensure_runtime_directories()
    require_database_schema()
    invalid = [item for item in ModelRegistry(settings).verify_all() if not item.valid]
    if invalid:
        details = ", ".join(f"{item.model_id}: {item.error}" for item in invalid)
        raise RuntimeError(f"Model verification failed; worker will not start: {details}")
    # RQ forks a child per job. Never let a live psycopg connection cross that fork.
    get_engine().dispose()
    connection = Redis.from_url(settings.redis_url)
    worker = Worker([Queue(settings.queue_name, connection=connection)], connection=connection)
    try:
        # Never import or initialize torch.cuda in the RQ parent: jobs execute in
        # forked children and CUDA explicitly rejects post-fork reinitialization.
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        name, memory_mib, capability, driver = [
            value.strip() for value in completed.stdout.splitlines()[0].split(",")
        ]
        connection.set(
            "restorai:worker:capabilities",
            json.dumps(
                {
                    "worker_name": worker.name,
                    "cuda_available": True,
                    "device_name": name,
                    "total_vram_bytes": int(memory_mib) * 1024 * 1024,
                    "compute_capability": capability,
                    "driver_version": driver,
                    "fp16_enabled": settings.use_fp16,
                }
            ),
        )
    except Exception:
        connection.set(
            "restorai:worker:capabilities",
            json.dumps({"worker_name": worker.name, "cuda_available": False}),
        )
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run()
