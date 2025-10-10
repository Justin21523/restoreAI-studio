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
from utils.logger import logger

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
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: float = 0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


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


class JobRegistry:
    """Thread-safe job registry with disk journaling"""

    def __init__(self):
        self.jobs: Dict[str, Job] = {}
        self.futures: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(
            max_workers=settings.max_workers, thread_name_prefix="restorai_job"
        )
        self.journal_path = Path("jobs_journal.jsonl")
        self.lock = threading.RLock()

        # Load existing jobs from journal
        self._load_journal()

    def _load_journal(self):
        """Load jobs from journal file"""
        if not self.journal_path.exists():
            return

        try:
            with open(self.journal_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    job_data = json.loads(line)
                    job = Job(**job_data)
                    # Only load recent or active jobs
                    if job.status in [JobStatus.PENDING, JobStatus.RUNNING] or (
                        job.completed_at and time.time() - job.completed_at < 3600
                    ):
                        self.jobs[job.id] = job
        except Exception as e:
            logger.error(f"Failed to load job journal: {e}")

    def _journal_job(self, job: Job):
        """Append job state to journal"""
        try:
            with open(self.journal_path, "a") as f:
                f.write(json.dumps(job.__dict__) + "\n")
        except Exception as e:
            logger.error(f"Failed to journal job {job.id}: {e}")

    def create_job(self, task_type: str, parameters: Dict[str, Any]) -> str:
        """Create a new job"""
        with self.lock:
            job_id = str(uuid.uuid4())
            job = Job(
                id=job_id,
                task_type=task_type,
                parameters=parameters,
                status=JobStatus.PENDING,
                progress=0.0,
            )
            self.jobs[job_id] = job
            self._journal_job(job)
            logger.info(f"Created job: {job_id} for {task_type}")
            return job_id

    def submit_job(
        self,
        job_id: str,
        task_fn: Callable,
        progress_callback: Optional[Callable] = None,
    ) -> bool:
        """Submit job to thread pool"""
        with self.lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            if job.status != JobStatus.PENDING:
                return False

            # Update job status
            job.status = JobStatus.RUNNING
            job.started_at = time.time()
            self._journal_job(job)

            # Submit to executor
            future = self.executor.submit(
                self._run_job, job_id, task_fn, progress_callback
            )
            self.futures[job_id] = future

            logger.info(f"Submitted job: {job_id}")
            return True

    def _run_job(
        self, job_id: str, task_fn: Callable, progress_callback: Optional[Callable]
    ):
        """Run job with progress tracking"""
        job = self.jobs.get(job_id)
        if not job:
            return

        def update_progress(progress: float, message: str = ""):
            if job:
                job.progress = progress
                if progress_callback:
                    progress_callback(job_id, progress, message)
                self._journal_job(job)

        try:
            update_progress(0.1, "Starting processing")

            # Execute the task
            result = task_fn(job.parameters, update_progress)

            # Update job with result
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.result = result
            job.completed_at = time.time()
            job.artifact_id = result.get("artifact_id")
            self._journal_job(job)

            update_progress(1.0, "Completed")
            logger.info(f"Job completed: {job_id}")

        except Exception as e:
            # Update job with error
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()
            self._journal_job(job)

            update_progress(1.0, f"Failed: {str(e)}")
            logger.error(f"Job failed: {job_id} - {e}")

    def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID"""
        return self.jobs.get(job_id)

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job"""
        with self.lock:
            if job_id not in self.jobs:
                return False

            job = self.jobs[job_id]
            future = self.futures.get(job_id)

            if future and not future.done():
                future.cancel()
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                self._journal_job(job)
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

        with self.lock:
            jobs_to_remove = []
            for job_id, job in self.jobs.items():
                if job.completed_at and job.completed_at < cutoff:
                    jobs_to_remove.append(job_id)

            for job_id in jobs_to_remove:
                del self.jobs[job_id]
                if job_id in self.futures:
                    del self.futures[job_id]

            if jobs_to_remove:
                logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")


# Global job registry instance
job_registry = JobRegistry()


class JobRunner:
    """ThreadPool-based job runner with progress tracking"""

    def __init__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=settings.jobs_max_workers, thread_name_prefix="restorai_job"
        )
        self.jobs: Dict[str, Job] = {}
        self.futures: Dict[str, Future] = {}
        self.progress_callbacks: Dict[str, Callable] = {}

    def submit(
        self,
        task_type: str,
        parameters: Dict[str, Any],
        task_fn: Callable,
        progress_callback: Optional[Callable] = None,
    ) -> str:
        """Submit a new job to the queue"""
        # Check queue limits
        if len(self.jobs) >= settings.jobs_max_queue:
            raise RuntimeError("Job queue is full")

        job_id = str(uuid.uuid4())

        job = Job(
            id=job_id,
            task_type=task_type,
            parameters=parameters,
            status=JobStatus.QUEUED,
            progress=0.0,
        )

        self.jobs[job_id] = job

        # Submit to thread pool
        future = self.executor.submit(self._run_task, job_id, task_fn, parameters)
        self.futures[job_id] = future

        if progress_callback:
            self.progress_callbacks[job_id] = progress_callback

        logger.info(f"Job {job_id} submitted for {task_type}")
        return job_id

    def _run_task(self, job_id: str, task_fn: Callable, parameters: Dict[str, Any]):
        """Run task with progress tracking and error handling"""
        job = self.jobs[job_id]

        try:
            # Update status to running
            job.status = JobStatus.RUNNING
            job.started_at = time.time()

            # Define progress callback for the task
            def update_progress(progress: float):
                job.progress = progress
                if job_id in self.progress_callbacks:
                    self.progress_callbacks[job_id](progress)

            # Run the actual task
            result = task_fn(**parameters, progress_callback=update_progress)

            # Mark as completed
            job.status = JobStatus.COMPLETED
            job.progress = 1.0
            job.result = result
            job.completed_at = time.time()

            logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            # Mark as failed
            job.status = JobStatus.FAILED
            job.error = str(e)
            job.completed_at = time.time()

            logger.error(f"Job {job_id} failed: {str(e)}")

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
            logger.info(f"Job {job_id} cancelled")
            return True

        return False

    def cleanup_old_jobs(self):
        """Remove old completed jobs to free memory"""
        current_time = time.time()
        jobs_to_remove = []

        for job_id, job in self.jobs.items():
            if (
                job.completed_at
                and current_time - job.completed_at > settings.jobs_ttl_seconds
            ):
                jobs_to_remove.append(job_id)

        for job_id in jobs_to_remove:
            del self.jobs[job_id]
            if job_id in self.futures:
                del self.futures[job_id]
            if job_id in self.progress_callbacks:
                del self.progress_callbacks[job_id]

        if jobs_to_remove:
            logger.info(f"Cleaned up {len(jobs_to_remove)} old jobs")

    def get_queue_status(self) -> Dict[str, Any]:
        """Get overall job queue status"""
        status_counts = {status: 0 for status in JobStatus}
        for job in self.jobs.values():
            status_counts[job.status] += 1

        return {
            "total_jobs": len(self.jobs),
            "status_counts": status_counts,
            "max_workers": settings.jobs_max_workers,
            "max_queue": settings.jobs_max_queue,
        }


# Global job runner instance
job_runner = JobRunner()
