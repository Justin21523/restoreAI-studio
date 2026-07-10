from __future__ import annotations

from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from typing import Any

from redis import Redis
from rq import Queue
from sqlalchemy import select
from sqlalchemy.orm import Session

from restorai.orm import Batch, Job, JobEvent, JobOutbox, utcnow
from restorai.settings import Settings, get_settings


class JobService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.session = session
        self.settings = settings or get_settings()

    def create_batch(
        self,
        kind: str,
        operation: str,
        parameters: dict[str, Any],
        *,
        retry_of_batch_id: str | None = None,
    ) -> Batch:
        batch = Batch(
            kind=kind,
            operation=operation,
            parameters=dict(parameters),
            retry_of_batch_id=retry_of_batch_id,
        )
        self.session.add(batch)
        self.session.flush()
        return batch

    def create_job(
        self,
        *,
        kind: str,
        operation: str,
        parameters: dict[str, Any],
        input_path: Path,
        original_filename: str,
        batch_id: str | None = None,
        retry_of_job_id: str | None = None,
        commit: bool = True,
    ) -> Job:
        job = Job(
            kind=kind,
            operation=operation,
            parameters=dict(parameters),
            input_path=str(input_path),
            original_filename=original_filename,
            batch_id=batch_id,
            retry_of_job_id=retry_of_job_id,
        )
        self.session.add(job)
        self.session.flush()
        self.session.add(JobOutbox(job_id=job.id))
        self.add_event(job, "queued", "queued", 0, "Job queued")
        if batch_id:
            batch = self.session.get(Batch, batch_id)
            if batch:
                batch.total_items += 1
        if commit:
            self.session.commit()
        return job

    def add_event(
        self,
        job: Job,
        event_type: str,
        stage: str,
        progress: float,
        message: str,
        payload: dict[str, Any] | None = None,
    ) -> JobEvent:
        event = JobEvent(
            job_id=job.id,
            event_type=event_type,
            stage=stage,
            progress=max(0, min(float(progress), 1)),
            message=message,
            payload=payload or {},
        )
        self.session.add(event)
        job.stage = stage
        job.progress = event.progress
        return event

    def dispatch_pending(self, limit: int = 100) -> int:
        pending = self.session.scalars(
            select(JobOutbox)
            .where(JobOutbox.dispatched_at.is_(None))
            .order_by(JobOutbox.id)
            .limit(limit)
        ).all()
        if not pending or not self.settings.queue_enabled:
            return 0
        try:
            queue = Queue(
                self.settings.queue_name,
                connection=Redis.from_url(self.settings.redis_url),
                default_timeout=24 * 60 * 60,
            )
        except Exception as exc:
            for item in pending:
                item.attempts += 1
                item.last_error = str(exc)
            self.session.commit()
            return 0
        dispatched = 0
        for item in pending:
            try:
                queue.enqueue(
                    "restorai.worker.execute_job",
                    item.job_id,
                    job_id=item.job_id,
                    result_ttl=24 * 60 * 60,
                    failure_ttl=7 * 24 * 60 * 60,
                )
                item.dispatched_at = utcnow()
                item.last_error = None
                dispatched += 1
            except Exception as exc:
                item.attempts += 1
                item.last_error = str(exc)
        self.session.commit()
        return dispatched

    def request_cancel(self, job: Job, *, commit: bool = True) -> None:
        if job.status == "queued":
            job.status = "cancelled"
            job.stage = "cancelled"
            job.finished_at = utcnow()
            self.add_event(job, "cancelled", "cancelled", job.progress, "Job cancelled")
        elif job.status == "running":
            job.status = "cancelling"
            self.add_event(job, "progress", "cancelling", job.progress, "Cancellation requested")
        with suppress(Exception):
            Redis.from_url(self.settings.redis_url).setex(f"restorai:cancel:{job.id}", 86400, "1")
        if commit:
            self.session.commit()

    def retry(self, job: Job) -> Job:
        if job.status not in {"failed", "cancelled"}:
            raise ValueError("only failed or cancelled jobs may be retried")
        return self.create_job(
            kind=job.kind,
            operation=job.operation,
            parameters=job.parameters,
            input_path=Path(job.input_path),
            original_filename=job.original_filename,
            batch_id=None,
            retry_of_job_id=job.id,
        )

    def cancel_batch(self, batch: Batch) -> int:
        cancellable = [
            job for job in batch.jobs if job.status in {"queued", "running", "cancelling"}
        ]
        for job in cancellable:
            self.request_cancel(job, commit=False)
        batch.succeeded_items = sum(job.status == "succeeded" for job in batch.jobs)
        batch.failed_items = sum(
            job.status in {"failed", "cancelled"} for job in batch.jobs
        )
        if cancellable:
            batch.status = "cancelling" if any(
                job.status == "cancelling" for job in cancellable
            ) else "cancelled"
        self.session.commit()
        return len(cancellable)

    def retry_failed_batch(self, batch: Batch) -> Batch:
        failed = [job for job in batch.jobs if job.status == "failed"]
        if not failed:
            raise ValueError("batch has no failed jobs to retry")
        missing = [job.original_filename for job in failed if not Path(job.input_path).is_file()]
        if missing:
            raise ValueError(f"retry input is unavailable: {', '.join(missing)}")
        retried = self.create_batch(
            batch.kind,
            batch.operation,
            batch.parameters,
            retry_of_batch_id=batch.id,
        )
        for job in failed:
            self.create_job(
                kind=job.kind,
                operation=job.operation,
                parameters=job.parameters,
                input_path=Path(job.input_path),
                original_filename=job.original_filename,
                batch_id=retried.id,
                retry_of_job_id=job.id,
                commit=False,
            )
        self.session.commit()
        return retried

    @staticmethod
    def expiry_time(settings: Settings | None = None):  # type: ignore[no-untyped-def]
        current = settings or get_settings()
        return utcnow() + timedelta(hours=current.artifact_retention_hours)
