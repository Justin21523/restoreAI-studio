from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from restorai.adapters.base import ProcessingResult
from restorai.pipelines.image import ImageOperation, ImagePipeline
from restorai.settings import Settings


class FakeUpscaler:
    def process(self, image, *, scale, progress, cancel):  # type: ignore[no-untyped-def]
        output = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return ProcessingResult(image=output, metadata={"model_id": "fake", "scale": scale})


def test_image_pipeline_writes_expected_dimensions(tmp_path: Path, monkeypatch) -> None:
    settings = Settings(
        model_root=Path("/mnt/c/ai_models"),
        storage_root=tmp_path / "storage",
        max_image_pixels=100_000,
        max_output_pixels=1_000_000,
    )
    source = tmp_path / "source.png"
    output = tmp_path / "output.png"
    cv2.imwrite(str(source), np.full((12, 16, 3), 120, dtype=np.uint8))
    pipeline = ImagePipeline(settings=settings)
    monkeypatch.setattr(pipeline, "_upscaler", lambda _scale: FakeUpscaler())
    result = pipeline.process(source, output, operation=ImageOperation.UPSCALE, scale=4)
    assert result.output_path == output
    assert result.metadata["output_width"] == 64
    assert result.metadata["output_height"] == 48


def test_image_pipeline_rejects_excessive_output(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    cv2.imwrite(str(source), np.zeros((100, 100, 3), dtype=np.uint8))
    pipeline = ImagePipeline(
        settings=Settings(
            model_root=Path("/mnt/c/ai_models"),
            max_image_pixels=20_000,
            max_output_pixels=20_000,
        )
    )
    with pytest.raises(ValueError, match="MAX_OUTPUT_PIXELS"):
        pipeline.validate(source, scale=4)
