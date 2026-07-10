from __future__ import annotations

import hashlib
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from restorai.settings import Settings, get_settings


class ModelRegistryError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    relative_path: Path
    source: str
    required: bool
    min_bytes: int
    sha256: str | None = None
    native_scale: int | None = None
    source_revision: str | None = None


@dataclass(frozen=True)
class ModelStatus:
    model_id: str
    path: Path
    available: bool
    valid: bool
    size_bytes: int
    sha256: str | None
    error: str | None = None


class ModelRegistry:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.root = self.settings.model_root.resolve()
        self.specs = self._load_manifest(self.settings.model_manifest)

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, ModelSpec]:
        data: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
        specs: dict[str, ModelSpec] = {}
        for model_id, raw in data.get("models", {}).items():
            specs[model_id] = ModelSpec(
                model_id=model_id,
                family=str(raw["family"]),
                relative_path=Path(raw["relative_path"]),
                source=str(raw["source"]),
                required=bool(raw.get("required", True)),
                min_bytes=int(raw.get("min_bytes", 1)),
                sha256=raw.get("sha256"),
                native_scale=raw.get("native_scale"),
                source_revision=raw.get("source_revision"),
            )
        return specs

    def resolve(self, model_id: str) -> Path:
        try:
            spec = self.specs[model_id]
        except KeyError as exc:
            raise ModelRegistryError(f"Unknown model: {model_id}") from exc
        path = (self.root / spec.relative_path).resolve()
        if path != self.root and self.root not in path.parents:
            raise ModelRegistryError(f"Model path escapes MODEL_ROOT: {spec.relative_path}")
        return path

    @staticmethod
    def hash_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def inspect(self, model_id: str, *, verify_hash: bool = True) -> ModelStatus:
        spec = self.specs[model_id]
        path = self.resolve(model_id)
        if not path.is_file():
            return ModelStatus(model_id, path, False, False, 0, None, "missing")
        size = path.stat().st_size
        if size < spec.min_bytes:
            return ModelStatus(model_id, path, True, False, size, None, "file is too small")
        actual_hash = self.hash_file(path) if verify_hash else None
        if verify_hash and spec.sha256 and actual_hash != spec.sha256:
            return ModelStatus(model_id, path, True, False, size, actual_hash, "sha256 mismatch")
        if not os.access(path, os.R_OK):
            return ModelStatus(model_id, path, True, False, size, actual_hash, "not readable")
        return ModelStatus(model_id, path, True, True, size, actual_hash)

    def verify_all(self, *, verify_hash: bool = True) -> list[ModelStatus]:
        return [self.inspect(model_id, verify_hash=verify_hash) for model_id in self.specs]

    def require(self, model_id: str, *, verify_hash: bool = False) -> Path:
        """Return a usable model path.

        Workers perform one full manifest verification before accepting work. Model
        loading then uses the inexpensive size/readability check so every forked
        RQ job does not re-hash several gigabytes of immutable weights.
        """
        status = self.inspect(model_id, verify_hash=verify_hash)
        if not status.valid:
            raise ModelRegistryError(f"Model {model_id} is invalid: {status.error} ({status.path})")
        return status.path

    def install_missing(self) -> list[Path]:
        installed: list[Path] = []
        for model_id, spec in self.specs.items():
            status = self.inspect(model_id, verify_hash=False)
            if status.valid:
                continue
            target = self.resolve(model_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as temporary:
                temp_path = Path(temporary.name)
            try:
                urllib.request.urlretrieve(spec.source, temp_path)
                if temp_path.stat().st_size < spec.min_bytes:
                    raise ModelRegistryError(f"Downloaded model is too small: {model_id}")
                if spec.sha256 and self.hash_file(temp_path) != spec.sha256:
                    raise ModelRegistryError(f"Downloaded model checksum mismatch: {model_id}")
                temp_path.replace(target)
                installed.append(target)
            finally:
                temp_path.unlink(missing_ok=True)
        return installed
