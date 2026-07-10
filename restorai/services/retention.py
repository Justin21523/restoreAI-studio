from __future__ import annotations

from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from restorai.orm import Artifact, Job, utcnow


def cleanup_expired(session: Session) -> int:
    artifacts = session.scalars(
        select(Artifact).where(Artifact.deleted_at.is_(None), Artifact.expires_at <= utcnow())
    ).all()
    deleted = 0
    for artifact in artifacts:
        Path(artifact.output_path).unlink(missing_ok=True)
        job = session.get(Job, artifact.job_id)
        if job:
            Path(job.input_path).unlink(missing_ok=True)
            job.status = "expired"
            job.stage = "expired"
        artifact.deleted_at = utcnow()
        deleted += 1
    session.commit()
    return deleted
