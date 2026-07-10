from __future__ import annotations

import hashlib
import re
from pathlib import Path

from fastapi import UploadFile

from restorai.settings import Settings, get_settings

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(value: str) -> str:
    cleaned = SAFE_NAME.sub("_", Path(value).name).strip("._")
    return cleaned[:160] or "upload.bin"


class StorageService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    def job_upload_path(self, job_id: str, filename: str) -> Path:
        return self.settings.storage_root / "uploads" / job_id / safe_filename(filename)

    def job_work_dir(self, job_id: str) -> Path:
        return self.settings.storage_root / "work" / job_id

    def artifact_path(self, job_id: str, suffix: str) -> Path:
        return self.settings.storage_root / "artifacts" / job_id / f"result{suffix}"

    async def save_upload(self, upload: UploadFile, target: Path, max_bytes: int) -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        try:
            with target.open("wb") as handle:
                while chunk := await upload.read(1024 * 1024):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("upload exceeds configured size limit")
                    handle.write(chunk)
        except Exception:
            target.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()
        return total

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
