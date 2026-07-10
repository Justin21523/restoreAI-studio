from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from restorai.api import app
from restorai.db import Base, get_session
from restorai.orm import Job
from restorai.settings import Settings


def _client(tmp_path: Path) -> tuple[AsyncClient, Session]:
    engine = create_engine(f"sqlite:///{tmp_path / 'api.db'}")
    Base.metadata.create_all(engine)
    session = Session(engine, expire_on_commit=False)

    def override_session() -> Iterator[Session]:
        yield session

    app.dependency_overrides[get_session] = override_session
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test"), session


@pytest.mark.asyncio
async def test_live_health_contract() -> None:
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        response = await client.get("/api/v1/health/live")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_jobs_list_and_missing_detail(tmp_path: Path) -> None:
    client, session = _client(tmp_path)
    try:
        job = Job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=str(tmp_path / "input.png"),
            original_filename="input.png",
        )
        session.add(job)
        session.commit()

        listed = await client.get("/api/v1/jobs")
        assert listed.status_code == 200
        assert listed.json()[0]["id"] == job.id
        assert listed.json()[0]["artifact_id"] is None
        assert (await client.get("/api/v1/jobs/does-not-exist")).status_code == 404
    finally:
        app.dependency_overrides.clear()
        await client.aclose()
        session.close()


@pytest.mark.asyncio
async def test_invalid_scale_is_rejected_before_upload(tmp_path: Path) -> None:
    client, session = _client(tmp_path)
    try:
        response = await client.post(
            "/api/v1/jobs/images",
            files={"file": ("input.png", b"not-needed", "image/png")},
            data={"operation": "upscale", "scale": "3"},
        )
        assert response.status_code == 422
        assert "scale must be 2 or 4" in response.text
    finally:
        app.dependency_overrides.clear()
        await client.aclose()
        session.close()


@pytest.mark.asyncio
async def test_upload_limit_returns_413(tmp_path: Path, monkeypatch) -> None:
    client, session = _client(tmp_path)
    settings = Settings(
        model_root=Path("/mnt/c/ai_models"),
        storage_root=tmp_path / "storage",
        max_image_bytes=4,
        queue_enabled=False,
    )
    monkeypatch.setattr("restorai.api.get_settings", lambda: settings)
    try:
        response = await client.post(
            "/api/v1/jobs/images",
            files={"file": ("input.png", b"too-large", "image/png")},
            data={"operation": "upscale", "scale": "2"},
        )
        assert response.status_code == 413
        assert "configured size limit" in response.text
    finally:
        app.dependency_overrides.clear()
        await client.aclose()
        session.close()


@pytest.mark.asyncio
async def test_job_detail_and_input_download(tmp_path: Path, monkeypatch) -> None:
    client, session = _client(tmp_path)
    settings = Settings(
        model_root=Path("/mnt/c/ai_models"),
        storage_root=tmp_path / "storage",
        queue_enabled=False,
    )
    source = settings.storage_root / "uploads" / "job" / "portrait.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"input")
    monkeypatch.setattr("restorai.api.get_settings", lambda: settings)
    try:
        job = Job(
            kind="image",
            operation="upscale",
            parameters={"scale": 2},
            input_path=str(source),
            original_filename="portrait.png",
        )
        session.add(job)
        session.commit()
        detail = await client.get(f"/api/v1/jobs/{job.id}")
        assert detail.status_code == 200
        assert detail.json()["artifact"] is None
        assert detail.json()["events"] == []
        downloaded = await client.get(f"/api/v1/jobs/{job.id}/input")
        assert downloaded.status_code == 200
        assert downloaded.content == b"input"
    finally:
        app.dependency_overrides.clear()
        await client.aclose()
        session.close()
