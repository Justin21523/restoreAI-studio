from __future__ import annotations

import asyncio
import json
import mimetypes
import shutil
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from datetime import timezone
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from restorai import __version__
from restorai.db import get_session, require_database_schema, session_factory
from restorai.model_registry import ModelRegistry
from restorai.orm import Artifact, Batch, Job, JobEvent, utcnow
from restorai.schemas import (
    ArtifactRead,
    BatchDetailRead,
    BatchRead,
    JobDetailRead,
    JobRead,
    ModelStatusRead,
    SystemStatusRead,
)
from restorai.services.jobs import JobService
from restorai.services.retention import cleanup_expired
from restorai.services.storage import StorageService
from restorai.settings import Settings, get_settings

TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "expired"}


async def _maintenance() -> None:
    factory = session_factory()
    while True:
        with factory() as session:
            service = JobService(session)
            service.dispatch_pending()
            cleanup_expired(session)
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    settings.ensure_runtime_directories()
    require_database_schema()
    task = asyncio.create_task(_maintenance())
    try:
        yield
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task


app = FastAPI(
    title="RestorAI Studio API",
    version=__version__,
    docs_url="/api/v1/docs",
    openapi_url="/api/v1/openapi.json",
    lifespan=lifespan,
)


def _settings() -> Settings:
    return get_settings()


def _service(session: Session) -> JobService:
    return JobService(session)


def _require_scale(scale: int) -> int:
    if scale not in (2, 4):
        raise HTTPException(422, "scale must be 2 or 4")
    return scale


@app.get("/api/v1/health/live")
def live() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/api/v1/health/ready")
def ready(session: Session = Depends(get_session)) -> dict[str, object]:
    settings = get_settings()
    failures: list[str] = []
    try:
        session.execute(text("SELECT 1"))
    except Exception as exc:
        failures.append(f"database: {exc}")
    statuses = ModelRegistry(settings).verify_all(verify_hash=False)
    failures.extend(f"model {item.model_id}: {item.error}" for item in statuses if not item.valid)
    if settings.queue_enabled:
        try:
            from redis import Redis

            Redis.from_url(settings.redis_url).ping()
        except Exception as exc:
            failures.append(f"redis: {exc}")
    if failures:
        raise HTTPException(503, {"status": "not_ready", "failures": failures})
    return {"status": "ready", "models": len(statuses)}


@app.get("/api/v1/models", response_model=list[ModelStatusRead])
def models() -> list[ModelStatusRead]:
    registry = ModelRegistry()
    return [
        ModelStatusRead(
            model_id=item.model_id,
            family=registry.specs[item.model_id].family,
            path=str(item.path),
            available=item.available,
            valid=item.valid,
            size_bytes=item.size_bytes,
            sha256=item.sha256 or registry.specs[item.model_id].sha256,
            error=item.error,
        )
        for item in registry.verify_all(verify_hash=False)
    ]


async def _create_job_from_upload(
    *,
    upload: UploadFile,
    kind: Literal["image", "video"],
    operation: str,
    parameters: dict[str, object],
    session: Session,
    batch_id: str | None = None,
) -> Job:
    settings = get_settings()
    token = str(uuid.uuid4())
    storage = StorageService(settings)
    input_path = storage.job_upload_path(token, upload.filename or "upload.bin")
    limit = settings.max_image_bytes if kind == "image" else settings.max_video_bytes
    try:
        await storage.save_upload(upload, input_path, limit)
        job = _service(session).create_job(
            kind=kind,
            operation=operation,
            parameters=parameters,
            input_path=input_path,
            original_filename=upload.filename or input_path.name,
            batch_id=batch_id,
        )
        _service(session).dispatch_pending()
        return job
    except ValueError as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(413, str(exc)) from exc
    except Exception:
        input_path.unlink(missing_ok=True)
        raise


@app.post("/api/v1/jobs/images", response_model=JobRead, status_code=202)
async def create_image_job(
    file: Annotated[UploadFile, File()],
    operation: Annotated[
        Literal["upscale", "face_restore", "face_restore_upscale"], Form()
    ] = "upscale",
    scale: Annotated[int, Form()] = 2,
    face_method: Annotated[Literal["gfpgan", "codeformer"], Form()] = "codeformer",
    strength: Annotated[float, Form(ge=0, le=1)] = 0.8,
    fidelity: Annotated[float, Form(ge=0, le=1)] = 0.7,
    session: Session = Depends(get_session),
) -> Job:
    return await _create_job_from_upload(
        upload=file,
        kind="image",
        operation=operation,
        parameters={
            "scale": _require_scale(scale),
            "face_method": face_method,
            "strength": strength,
            "fidelity": fidelity,
        },
        session=session,
    )


@app.post("/api/v1/jobs/videos", response_model=JobRead, status_code=202)
async def create_video_job(
    file: Annotated[UploadFile, File()],
    operation: Annotated[
        Literal["interpolate", "upscale", "interpolate_upscale"], Form()
    ] = "interpolate",
    scale: Annotated[int, Form()] = 2,
    target_fps: Annotated[float | None, Form(ge=1, le=120)] = None,
    session: Session = Depends(get_session),
) -> Job:
    parameters: dict[str, object] = {"scale": _require_scale(scale)}
    if target_fps is not None:
        parameters["target_fps"] = target_fps
    return await _create_job_from_upload(
        upload=file,
        kind="video",
        operation=operation,
        parameters=parameters,
        session=session,
    )


async def _create_batch(
    *,
    files: list[UploadFile],
    kind: Literal["image", "video"],
    operation: str,
    parameters: dict[str, object],
    session: Session,
) -> Batch:
    settings = get_settings()
    maximum = settings.max_image_batch if kind == "image" else settings.max_video_batch
    if not files or len(files) > maximum:
        raise HTTPException(400, f"batch must contain 1 to {maximum} files")
    service = _service(session)
    batch = service.create_batch(kind, operation, parameters)
    session.commit()
    for upload in files:
        await _create_job_from_upload(
            upload=upload,
            kind=kind,
            operation=operation,
            parameters=parameters,
            session=session,
            batch_id=batch.id,
        )
    session.refresh(batch)
    return batch


@app.post("/api/v1/batches/images", response_model=BatchRead, status_code=202)
async def create_image_batch(
    files: Annotated[list[UploadFile], File()],
    operation: Annotated[
        Literal["upscale", "face_restore", "face_restore_upscale"], Form()
    ] = "upscale",
    scale: Annotated[int, Form()] = 2,
    face_method: Annotated[Literal["gfpgan", "codeformer"], Form()] = "codeformer",
    strength: Annotated[float, Form(ge=0, le=1)] = 0.8,
    fidelity: Annotated[float, Form(ge=0, le=1)] = 0.7,
    session: Session = Depends(get_session),
) -> Batch:
    return await _create_batch(
        files=files,
        kind="image",
        operation=operation,
        parameters={
            "scale": _require_scale(scale),
            "face_method": face_method,
            "strength": strength,
            "fidelity": fidelity,
        },
        session=session,
    )


@app.post("/api/v1/batches/videos", response_model=BatchRead, status_code=202)
async def create_video_batch(
    files: Annotated[list[UploadFile], File()],
    operation: Annotated[
        Literal["interpolate", "upscale", "interpolate_upscale"], Form()
    ] = "interpolate",
    scale: Annotated[int, Form()] = 2,
    target_fps: Annotated[float | None, Form(ge=1, le=120)] = None,
    session: Session = Depends(get_session),
) -> Batch:
    parameters: dict[str, object] = {"scale": _require_scale(scale)}
    if target_fps is not None:
        parameters["target_fps"] = target_fps
    return await _create_batch(
        files=files,
        kind="video",
        operation=operation,
        parameters=parameters,
        session=session,
    )


@app.get("/api/v1/jobs", response_model=list[JobRead])
def list_jobs(
    status: str | None = None,
    limit: int = 50,
    session: Session = Depends(get_session),
) -> list[Job]:
    statement = select(Job).order_by(Job.created_at.desc()).limit(min(max(limit, 1), 200))
    if status:
        statement = statement.where(Job.status == status)
    return list(session.scalars(statement))


@app.get("/api/v1/jobs/{job_id}", response_model=JobDetailRead)
def get_job(job_id: str, session: Session = Depends(get_session)) -> Job:
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(404, "job not found")
    return job


@app.get("/api/v1/jobs/{job_id}/input")
def download_job_input(job_id: str, session: Session = Depends(get_session)) -> FileResponse:
    job = get_job(job_id, session)
    if job.status == "expired" or (job.artifact and job.artifact.deleted_at):
        raise HTTPException(410, "input file has expired")
    path = Path(job.input_path).resolve()
    uploads_root = (get_settings().storage_root / "uploads").resolve()
    if uploads_root not in path.parents or not path.is_file():
        raise HTTPException(410, "input file is unavailable")
    media_type = mimetypes.guess_type(job.original_filename)[0] or "application/octet-stream"
    return FileResponse(path, media_type=media_type, filename=job.original_filename)


@app.post("/api/v1/jobs/{job_id}/cancel", response_model=JobRead)
def cancel_job(job_id: str, session: Session = Depends(get_session)) -> Job:
    job = get_job(job_id, session)
    _service(session).request_cancel(job)
    return job


@app.post("/api/v1/jobs/{job_id}/retry", response_model=JobRead, status_code=202)
def retry_job(job_id: str, session: Session = Depends(get_session)) -> Job:
    job = get_job(job_id, session)
    try:
        retried = _service(session).retry(job)
        _service(session).dispatch_pending()
        return retried
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/batches/{batch_id}", response_model=BatchDetailRead)
def get_batch(batch_id: str, session: Session = Depends(get_session)) -> Batch:
    batch = session.get(Batch, batch_id)
    if not batch:
        raise HTTPException(404, "batch not found")
    return batch


@app.post("/api/v1/batches/{batch_id}/cancel", response_model=BatchDetailRead)
def cancel_batch(batch_id: str, session: Session = Depends(get_session)) -> Batch:
    batch = get_batch(batch_id, session)
    _service(session).cancel_batch(batch)
    return batch


@app.post("/api/v1/batches/{batch_id}/retry-failed", response_model=BatchDetailRead)
def retry_failed_batch(batch_id: str, session: Session = Depends(get_session)) -> Batch:
    batch = get_batch(batch_id, session)
    try:
        retried = _service(session).retry_failed_batch(batch)
        _service(session).dispatch_pending()
        return retried
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/v1/jobs/{job_id}/events")
async def job_events(
    job_id: str,
    last_event_id: Annotated[int | None, Header(alias="Last-Event-ID")] = None,
) -> StreamingResponse:
    factory = session_factory()
    with factory() as session:
        if not session.get(Job, job_id):
            raise HTTPException(404, "job not found")

    async def stream() -> AsyncIterator[str]:
        cursor = last_event_id or 0
        idle_terminal_polls = 0
        while True:
            with factory() as session:
                events = session.scalars(
                    select(JobEvent)
                    .where(JobEvent.job_id == job_id, JobEvent.id > cursor)
                    .order_by(JobEvent.id)
                ).all()
                job = session.get(Job, job_id)
                for event in events:
                    cursor = event.id
                    payload = {
                        "id": event.id,
                        "type": event.event_type,
                        "stage": event.stage,
                        "progress": event.progress,
                        "message": event.message,
                        "payload": event.payload,
                        "timestamp": event.created_at.astimezone(timezone.utc).isoformat(),
                    }
                    yield f"id: {event.id}\ndata: {json.dumps(payload)}\n\n"
                if job and job.status in TERMINAL_STATUSES and not events:
                    idle_terminal_polls += 1
                    if idle_terminal_polls >= 2:
                        return
            await asyncio.sleep(0.5)

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.get("/api/v1/artifacts/{artifact_id}", response_model=ArtifactRead)
def get_artifact(artifact_id: str, session: Session = Depends(get_session)) -> Artifact:
    artifact = session.get(Artifact, artifact_id)
    if not artifact:
        raise HTTPException(404, "artifact not found")
    return artifact


@app.get("/api/v1/artifacts/{artifact_id}/download")
def download_artifact(artifact_id: str, session: Session = Depends(get_session)) -> FileResponse:
    artifact = get_artifact(artifact_id, session)
    if artifact.deleted_at or artifact.expires_at <= utcnow():
        raise HTTPException(410, "artifact file has expired")
    path = Path(artifact.output_path).resolve()
    root = get_settings().storage_root.resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(410, "artifact file is unavailable")
    return FileResponse(path, media_type=artifact.mime_type, filename=path.name)


@app.delete("/api/v1/artifacts/{artifact_id}", status_code=204)
def delete_artifact(artifact_id: str, session: Session = Depends(get_session)) -> None:
    artifact = get_artifact(artifact_id, session)
    Path(artifact.output_path).unlink(missing_ok=True)
    job = session.get(Job, artifact.job_id)
    if job:
        Path(job.input_path).unlink(missing_ok=True)
    artifact.deleted_at = utcnow()
    session.commit()


@app.get("/api/v1/system/status", response_model=SystemStatusRead)
def system_status(session: Session = Depends(get_session)) -> dict[str, object]:
    from redis import Redis
    from rq import Queue, Worker

    settings = get_settings()
    session.execute(text("SELECT 1"))
    database = {"status": "ready"}
    redis_status: dict[str, object] = {"status": "unavailable"}
    queue_status: dict[str, object] = {"name": settings.queue_name, "queued": 0}
    worker_status: dict[str, object] = {"status": "offline", "workers": []}
    gpu: dict[str, object] | None = None
    degraded = False
    try:
        connection = Redis.from_url(settings.redis_url)
        connection.ping()
        redis_status = {"status": "ready"}
        queue = Queue(settings.queue_name, connection=connection)
        queue_status = {
            "name": settings.queue_name,
            "queued": queue.count,
            "running": queue.started_job_registry.count,
            "failed": queue.failed_job_registry.count,
        }
        workers = Worker.all(connection=connection, queue=queue)
        worker_rows = [
            {"name": item.name, "state": item.get_state(), "queues": item.queue_names()}
            for item in workers
        ]
        worker_status = {
            "status": "online" if workers else "offline",
            "workers": worker_rows,
        }
        raw_gpu = connection.get("restorai:worker:capabilities")
        if raw_gpu:
            gpu = json.loads(raw_gpu)
        if not workers:
            degraded = True
    except Exception as exc:
        redis_status = {"status": "unavailable", "detail": str(exc)[:200]}
        degraded = True

    storage_root = settings.storage_root.resolve()
    storage_root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(storage_root)
    storage = {
        "path": str(storage_root),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }
    statuses = ModelRegistry(settings).verify_all(verify_hash=False)
    invalid = [item.model_id for item in statuses if not item.valid]
    models_status = {
        "status": "ready" if not invalid else "degraded",
        "total": len(statuses),
        "valid": len(statuses) - len(invalid),
        "invalid": invalid,
    }
    degraded = degraded or bool(invalid)
    return {
        "status": "degraded" if degraded else "ready",
        "database": database,
        "redis": redis_status,
        "queue": queue_status,
        "worker": worker_status,
        "gpu": gpu,
        "storage": storage,
        "models": models_status,
    }


def run() -> None:
    import uvicorn

    settings = get_settings()
    uvicorn.run("restorai.api:app", host=settings.app_host, port=settings.app_port)


web_dist = Path("dist")
if web_dist.is_dir():
    app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")


if __name__ == "__main__":
    run()
