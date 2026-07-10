from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from restorai.db import Base
from restorai.orm import Batch, Job, JobOutbox
from restorai.services.jobs import JobService
from restorai.settings import Settings


def test_job_and_batch_are_persisted(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = Settings(
        model_root=tmp_path,
        storage_root=tmp_path / "storage",
        queue_enabled=False,
        database_url="sqlite:///:memory:",
    )
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    with Session(engine, expire_on_commit=False) as session:
        service = JobService(session, settings)
        batch = service.create_batch("image", "upscale", {"scale": 2})
        session.commit()
        job = service.create_job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=input_path,
            original_filename="input.png",
            batch_id=batch.id,
        )
        assert session.get(Job, job.id).status == "queued"
        assert session.get(Batch, batch.id).total_items == 1
        assert session.query(JobOutbox).filter_by(job_id=job.id).one()
        assert job.events[0].event_type == "queued"


def test_cancel_queued_job_does_not_require_worker(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = Settings(model_root=tmp_path, queue_enabled=False)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")

    class FakeRedis:
        def setex(self, *_args) -> None:  # type: ignore[no-untyped-def]
            return None

    monkeypatch.setattr("restorai.services.jobs.Redis.from_url", lambda *_args: FakeRedis())
    with Session(engine, expire_on_commit=False) as session:
        service = JobService(session, settings)
        job = service.create_job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=input_path,
            original_filename="input.png",
        )
        service.request_cancel(job)
        assert job.status == "cancelled"
        assert job.finished_at is not None


def test_individual_retry_detaches_from_original_batch(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = Settings(model_root=tmp_path, queue_enabled=False)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    with Session(engine, expire_on_commit=False) as session:
        service = JobService(session, settings)
        batch = service.create_batch("image", "upscale", {"scale": 2})
        job = service.create_job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=input_path,
            original_filename="input.png",
            batch_id=batch.id,
        )
        job.status = "failed"
        session.commit()
        retried = service.retry(job)
        assert retried.retry_of_job_id == job.id
        assert retried.batch_id is None
        assert batch.total_items == 1


def test_retry_failed_batch_creates_linked_batch(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = Settings(model_root=tmp_path, queue_enabled=False)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    with Session(engine, expire_on_commit=False) as session:
        service = JobService(session, settings)
        batch = service.create_batch("image", "upscale", {"scale": 2})
        job = service.create_job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=input_path,
            original_filename="input.png",
            batch_id=batch.id,
        )
        job.status = "failed"
        session.commit()
        retried = service.retry_failed_batch(batch)
        assert retried.id != batch.id
        assert retried.retry_of_batch_id == batch.id
        assert retried.total_items == 1
        assert retried.jobs[0].retry_of_job_id == job.id


def test_outbox_recovers_after_queue_failure(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = Settings(model_root=tmp_path, queue_enabled=True)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    with Session(engine, expire_on_commit=False) as session:
        service = JobService(session, settings)
        job = service.create_job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=input_path,
            original_filename="input.png",
        )

        def unavailable_queue(*_args, **_kwargs):  # type: ignore[no-untyped-def]
            raise ConnectionError("redis down")

        monkeypatch.setattr("restorai.services.jobs.Queue", unavailable_queue)
        assert service.dispatch_pending() == 0
        outbox = session.query(JobOutbox).filter_by(job_id=job.id).one()
        assert outbox.dispatched_at is None
        assert outbox.attempts == 1

        class FakeQueue:
            def enqueue(self, *_args, **_kwargs) -> None:  # type: ignore[no-untyped-def]
                return None

        monkeypatch.setattr("restorai.services.jobs.Queue", lambda *_args, **_kwargs: FakeQueue())
        monkeypatch.setattr("restorai.services.jobs.Redis.from_url", lambda *_args: object())
        assert service.dispatch_pending() == 1
        assert outbox.dispatched_at is not None


def test_cancel_batch_updates_aggregate_status(tmp_path: Path, monkeypatch) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    settings = Settings(model_root=tmp_path, queue_enabled=False)
    input_path = tmp_path / "input.png"
    input_path.write_bytes(b"png")
    monkeypatch.setattr("restorai.services.jobs.Redis.from_url", lambda *_args: object())
    with Session(engine, expire_on_commit=False) as session:
        service = JobService(session, settings)
        batch = service.create_batch("image", "upscale", {"scale": 2})
        for name in ("one.png", "two.png"):
            service.create_job(
                kind="image",
                operation="upscale",
                parameters={"scale": 2},
                input_path=input_path,
                original_filename=name,
                batch_id=batch.id,
            )
        assert service.cancel_batch(batch) == 2
        assert batch.status == "cancelled"
        assert batch.failed_items == 2
