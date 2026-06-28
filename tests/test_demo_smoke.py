from __future__ import annotations

import io
import subprocess
import sys

from fastapi.testclient import TestClient
from PIL import Image

from webapp.main import app


def _png(width: int = 32, height: int = 24) -> bytes:
    image = Image.new("RGB", (width, height), "#111827")
    for x in range(width):
        for y in range(height):
            if x % 8 == 0 or y % 8 == 0:
                image.putpixel((x, y), (96, 165, 250))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def test_demo_health_contract():
    client = TestClient(app)
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["mode"] == "demo"


def test_demo_upscale_contract():
    client = TestClient(app)
    response = client.post(
        "/api/v1/upscale",
        files={"file": ("sample.png", _png(), "image/png")},
        data={"scale": "4", "method": "lanczos", "sharpen": "true"},
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-input-size"] == "32x24"
    assert response.headers["x-output-size"] == "128x96"
    assert response.content.startswith(b"\x89PNG")


def test_static_demo_build_check():
    result = subprocess.run(
        [sys.executable, "scripts/build_static_demo.py", "--check"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr + result.stdout
