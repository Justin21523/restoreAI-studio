from __future__ import annotations

from pathlib import Path

import pytest

from restorai.model_registry import ModelRegistry, ModelRegistryError
from restorai.settings import Settings


def test_model_root_must_be_absolute() -> None:
    with pytest.raises(ValueError, match="absolute"):
        Settings(model_root=Path("models"))


def test_default_manifest_is_packaged_and_absolute() -> None:
    settings = Settings(model_root=Path("/mnt/c/ai_models"))
    assert settings.model_manifest.is_absolute()
    assert settings.model_manifest.name == "model_manifest.yaml"
    assert settings.model_manifest.is_file()


def test_registry_rejects_path_escape(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
models:
  escaped:
    family: test
    relative_path: ../secret.pth
    source: https://example.invalid/secret.pth
    required: true
    min_bytes: 1
""",
        encoding="utf-8",
    )
    settings = Settings(model_root=tmp_path / "models", model_manifest=manifest)
    registry = ModelRegistry(settings)
    with pytest.raises(ModelRegistryError, match="escapes"):
        registry.resolve("escaped")


def test_registry_validates_size_and_hash(tmp_path: Path) -> None:
    root = tmp_path / "models"
    root.mkdir()
    model = root / "model.bin"
    model.write_bytes(b"model-data")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        """
version: 1
models:
  test-model:
    family: test
    relative_path: model.bin
    source: https://example.invalid/model.bin
    required: true
    min_bytes: 5
    sha256: 78af4dc485eb32a809e795bfca78a4f5313401b58d9bc81cc4d1b89445b97333
""",
        encoding="utf-8",
    )
    registry = ModelRegistry(Settings(model_root=root, model_manifest=manifest))
    status = registry.inspect("test-model")
    assert status.available is True
    assert status.valid is True
    assert registry.require("test-model") == model
