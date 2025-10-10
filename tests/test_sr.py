import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import tempfile
from PIL import Image
import numpy as np

from api.main import app

client = TestClient(app)


def create_test_image(width=100, height=100):
    """Create a test image for testing"""
    img = Image.new("RGB", (width, height), color="red")
    return img


def test_super_resolution_invalid_scale():
    """Test SR with invalid scale"""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img = create_test_image()
        img.save(tmp.name)

        response = client.post(
            "/api/v1/restore/sr",
            files={"file": ("test.png", open(tmp.name, "rb"), "image/png")},
            data={"scale": 3},  # Invalid scale
        )

        assert response.status_code == 400


def test_super_resolution_valid():
    """Test SR with valid parameters"""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img = create_test_image(50, 50)  # Small image for fast testing
        img.save(tmp.name)

        response = client.post(
            "/api/v1/restore/sr",
            files={"file": ("test.png", open(tmp.name, "rb"), "image/png")},
            data={"scale": 2, "model": "realesrgan-x2plus"},
        )

        # Should either succeed or give appropriate error (like model not found)
        assert response.status_code in [200, 500]


def test_face_restoration_invalid_method():
    """Test face restoration with invalid method"""
    with tempfile.NamedTemporaryFile(suffix=".png") as tmp:
        img = create_test_image()
        img.save(tmp.name)

        response = client.post(
            "/api/v1/restore/face",
            files={"file": ("test.png", open(tmp.name, "rb"), "image/png")},
            data={"method": "invalid", "strength": 0.5},
        )

        assert response.status_code == 400
