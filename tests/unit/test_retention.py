from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from restorai.db import Base
from restorai.orm import Artifact, Job, utcnow
from restorai.services.retention import cleanup_expired


def test_cleanup_removes_files_but_retains_metadata(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    input_path = tmp_path / "input.png"
    output_path = tmp_path / "output.png"
    input_path.write_bytes(b"input")
    output_path.write_bytes(b"output")

    with Session(engine, expire_on_commit=False) as session:
        job = Job(
            kind="image",
            operation="upscale",
            status="succeeded",
            parameters={},
            input_path=str(input_path),
            original_filename="input.png",
        )
        session.add(job)
        session.flush()
        artifact = Artifact(
            job_id=job.id,
            input_sha256="a" * 64,
            output_sha256="b" * 64,
            output_path=str(output_path),
            mime_type="image/png",
            size_bytes=6,
            expires_at=utcnow() - timedelta(seconds=1),
        )
        session.add(artifact)
        session.commit()

        assert cleanup_expired(session) == 1
        assert not input_path.exists()
        assert not output_path.exists()
        assert session.get(Job, job.id).status == "expired"
        assert session.get(Artifact, artifact.id).deleted_at is not None
