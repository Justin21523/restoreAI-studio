from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class JobRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    batch_id: str | None
    retry_of_job_id: str | None
    artifact_id: str | None
    kind: Literal["image", "video"]
    operation: str
    status: str
    stage: str
    progress: float
    parameters: dict[str, Any]
    original_filename: str
    error_code: str | None
    error_message: str | None
    queued_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    updated_at: datetime


class BatchRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    retry_of_batch_id: str | None
    kind: str
    operation: str
    status: str
    parameters: dict[str, Any]
    total_items: int
    succeeded_items: int
    failed_items: int
    created_at: datetime
    updated_at: datetime


class ArtifactRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    job_id: str
    input_sha256: str
    output_sha256: str
    mime_type: str
    size_bytes: int
    metadata_json: dict[str, Any]
    model_snapshot: dict[str, Any]
    created_at: datetime
    expires_at: datetime
    deleted_at: datetime | None


class JobEventRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    stage: str
    progress: float
    message: str
    payload: dict[str, Any]
    created_at: datetime


class JobDetailRead(JobRead):
    artifact: ArtifactRead | None
    events: list[JobEventRead]


class BatchDetailRead(BatchRead):
    jobs: list[JobRead]


class SystemStatusRead(BaseModel):
    status: Literal["ready", "degraded"]
    database: dict[str, Any]
    redis: dict[str, Any]
    queue: dict[str, Any]
    worker: dict[str, Any]
    gpu: dict[str, Any] | None
    storage: dict[str, Any]
    models: dict[str, Any]


class ModelStatusRead(BaseModel):
    model_id: str
    family: str
    path: str
    available: bool
    valid: bool
    size_bytes: int
    sha256: str | None
    loaded: bool = False
    error: str | None = None
