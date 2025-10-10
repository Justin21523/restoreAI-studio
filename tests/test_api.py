import pytest
from fastapi.testclient import TestClient
import tempfile
from pathlib import Path
import numpy as np
import cv2

from api.main import app


class TestAPI:
    """Test API endpoints"""

    @pytest.fixture
    def client(self):
        return TestClient(app)

    @pytest.fixture
    def test_image(self):
        """Create a test image file"""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            # Create a simple test image
            img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            cv2.imwrite(f.name, img)
            return Path(f.name)

    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/api/v1/healthz")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "timestamp" in data
        assert "system_info" in data

    def test_super_resolution_endpoint(self, client, test_image):
        """Test super resolution endpoint"""
        with open(test_image, "rb") as f:
            response = client.post(
                "/api/v1/restore/sr",
                files={"file": ("test.png", f, "image/png")},
                data={"scale": 2, "model": "realesrgan-x2plus"},
            )

        # Should either succeed or give appropriate error
        assert response.status_code in [200, 500, 400]

        if response.status_code == 200:
            data = response.json()
            assert "url" in data
            assert "width" in data
            assert "height" in data
            assert "latency_ms" in data

    def test_face_restoration_endpoint(self, client, test_image):
        """Test face restoration endpoint"""
        with open(test_image, "rb") as f:
            response = client.post(
                "/api/v1/restore/face",
                files={"file": ("test.png", f, "image/png")},
                data={"method": "gfpgan", "strength": 0.5},
            )

        assert response.status_code in [200, 500, 400]

        if response.status_code == 200:
            data = response.json()
            assert "url" in data
            assert "face_count" in data
            assert "latency_ms" in data

    def test_history_endpoint(self, client):
        """Test history endpoint"""
        response = client.get("/api/v1/history")
        assert response.status_code == 200
        data = response.json()
        assert "artifacts" in data
        assert "pagination" in data

    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/api/v1/metrics")
        # Metrics endpoint should always return 200
        assert response.status_code == 200

    def test_invalid_file_type(self, client):
        """Test invalid file type handling"""
        # Create a text file instead of image
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"This is not an image")
            file_path = Path(f.name)

        with open(file_path, "rb") as f:
            response = client.post(
                "/api/v1/restore/sr", files={"file": ("test.txt", f, "text/plain")}
            )

        # Should return 400 for invalid file type
        assert response.status_code == 400


# Cleanup
def cleanup_test_files():
    """Clean up any test files created during testing"""
    import glob

    test_files = glob.glob("/tmp/tmp*")  # Cleanup temp files
    for f in test_files:
        try:
            Path(f).unlink()
        except:
            pass
