import uuid
import time
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import logging
import json

from api.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Job data structure"""

    id: str
    task_type: str
    parameters: Dict[str, Any]
    status: JobStatus
    progress: float
    artifact_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class JobManager:
    """Thread-safe job manager with progress tracking"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.futures: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_workers, thread_name_prefix="restorai_job"
        )
        self.progress_callbacks: Dict[str, Callable] = {}
        self.event_callbacks: Dict[str, Callable] = {}

    def create_job(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new job"""
        job_id = str(uuid.uuid4())

        job = Job(
            id=job_id,
            task_type=task_type,
            parameters=parameters,
            status=JobStatus.PENDING,
            progress=0.0,
        )

        self.jobs[job_id] = job
        logger.info(f"Job created: {job_id} for {task_type}")
        return job_id

    def submit_job(
        self,
        job_id: str,
        task_fn: Callable,
        progress_callback: Optional[Callable] = None,
        event_callback: Optional[Callable] = None,
    ) -> bool:
        """Submit job to thread pool"""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        if job.status != JobStatus.PENDING:
            return False

        # Update job status
        job.status = JobStatus.RUNNING
        job.started_at = time.time()

        # Store callbacks
        if progress_callback:
            self.progress_callbacks[job_id] = progress_callback
        if event_callback:
            self.event_callbacks[job_id] = event_callback

        # Submit to executor
        future = self.executor.submit(self._run_job, job_id, task_fn)
        self.futures[job_id] = future

        logger.info(f"Job submitted: {job_id}")
        return True

    def _run_job(self, job_id: str, task_fn: Callable):
        """Run job with progress tracking"""
        job = self.jobs.get(job_id)
        if not job:
            return

        def update_progress(progress: float, message: str = ""):
            if job:
                job.progress = progress
                if job_id in self.progress_callbacks:
                    self.progress_callbacks[job_id](progress, message)

        def emit_event(event_type: str, data: Dict[str, Any]):
            if job_id in self.event_callbacks:
                self.event_callbacks[job_id](event_type, data)

        try:
            emit_event("started", {"message": "Processing started"})
            update_progress(0.1, "Starting processing")

            # Execute the task
            result = task_fn(job.parameters, update_progress, emit_event)

            # Update job with result
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.result = result
            job.completed_at = time.time()
            job.artifact_id = result.get("artifact_id")

            emit_event("completed", {"result": result})
            update_progress(1.0, "Completed")
            logger.info(f"Job completed: {job_id}")

        except Exception as e:
            # Update job with error
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

            emit_event("error", {"error": str(e)})
            update_progress(1.0, f"Failed: {str(e)}")
            logger.error(f"Job failed: {job_id} - {e}")

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        if job_id not in self.jobs:
            return False

        job = self.jobs[job_id]
        future = self.futures.get(job_id)

        if future and not future.done():
            future.cancel()
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()

            if job_id in self.event_callbacks:
                self.event_callbacks[job_id](
                    "cancelled", {"message": "Job was cancelled"}
                )

            logger.info(f"Job cancelled: {job_id}")
            return True

        return False

    def list_jobs(
        self, status: Optional[JobStatus] = None, limit: int = 100
    ) -> List[Job]:
        """List jobs with optional status filter"""
        jobs = list(self.jobs.values())

        if status:
            jobs = [job for job in jobs if job.status == status]

        # Sort by creation time (newest first)
        jobs.sort(key=lambda x: x.created_at, reverse=True)

        return jobs[:limit]

    def cleanup_old_jobs(self, older_than_hours: int = 24):
        """Remove old completed jobs"""
        current_time = time.time()
        cutoff = current_time - (older_than_hours * 3600)

        jobs_to_remove = []
        for job_id, job in self.jobs.items():
            if job.completed_at and job.completed_at < cutoff:
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            if job_id in self.futures:
                del self.futures[job_id]
            if job_id in self.progress_callbacks:
                del self.progress_callbacks[job_id]
            if job_id in self.event_callbacks:
                del self.event_callbacks[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# Global job manager instance
job_manager = JobManager()
