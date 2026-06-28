#!/usr/bin/env python3
"""Smoke test the mock-safe FastAPI demo API."""

from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from webapp.main import app


def make_image() -> bytes:
    image = Image.new("RGB", (64, 48), "#4263eb")
    for x in range(0, 64, 4):
        for y in range(0, 48, 4):
            if (x + y) % 16 == 0:
                image.putpixel((x, y), (255, 255, 255))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def main() -> None:
    client = TestClient(app)
    health = client.get("/api/v1/health")
    assert health.status_code == 200, health.text
    assert health.json()["status"] == "ok"

    response = client.post(
        "/api/v1/upscale",
        files={"file": ("sample.png", make_image(), "image/png")},
        data={"scale": "2", "method": "bicubic", "sharpen": "true"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-input-size"] == "64x48"
    assert response.headers["x-output-size"] == "128x96"
    assert response.content.startswith(b"\x89PNG")

    print(
        {
            "health": health.json(),
            "output_size": response.headers["x-output-size"],
            "bytes": len(response.content),
        }
    )


if __name__ == "__main__":
    main()
