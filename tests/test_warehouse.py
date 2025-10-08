#!/usr/bin/env python3
"""
AI Warehouse Path Testing
Tests centralized AI warehouse paths, model discovery, and fallback behaviors.
"""

import os
import tempfile
import shutil
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

# Test imports
from api.config import Settings, get_settings
from core.esrgan import ESRGANProcessor
from scripts.download_models import ModelDownloader
from api.main import app
from fastapi.testclient import TestClient


class TestWarehousePaths:
    """Test AI warehouse path configuration and validation"""

    def setup_method(self):
        """Setup test environment with temporary warehouse"""
        self.temp_warehouse = tempfile.mkdtemp(prefix="ai_warehouse_test_")
        self.temp_models = Path(self.temp_warehouse) / "models"
        self.temp_cache = Path(self.temp_warehouse) / "cache"

        # Create test directory structure
        self.temp_models.mkdir(parents=True, exist_ok=True)
        self.temp_cache.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        """Cleanup test environment"""
        if hasattr(self, "temp_warehouse") and Path(self.temp_warehouse).exists():
            shutil.rmtree(self.temp_warehouse)

    def test_settings_paths_configuration(self):
        """Test settings loads warehouse paths correctly"""
        with patch.dict(
            os.environ,
            {
                "AI_WAREHOUSE_ROOT": self.temp_warehouse,
                "MODELS_DIR": str(self.temp_models),
                "CACHE_DIR": str(self.temp_cache),
            },
        ):
            settings = Settings()

            assert settings.PATHS["warehouse_root"] == Path(self.temp_warehouse)
            assert settings.PATHS["models"] == self.temp_models
            assert settings.PATHS["cache"] == self.temp_cache
            assert all(path.exists() for path in settings.PATHS.values())

    def test_warehouse_directory_creation(self):
        """Test warehouse directories are created if missing"""
        new_warehouse = tempfile.mkdtemp(prefix="new_warehouse_test_")
        shutil.rmtree(new_warehouse)  # Remove to test creation

        with patch.dict(os.environ, {"AI_WAREHOUSE_ROOT": new_warehouse}):
            settings = Settings()

            # Directories should be auto-created
            assert settings.PATHS["warehouse_root"].exists()
            assert settings.PATHS["models"].exists()
            assert settings.PATHS["cache"].exists()

        # Cleanup
        shutil.rmtree(new_warehouse)

    def test_model_path_resolution(self):
        """Test model paths resolve correctly in warehouse"""
        with patch.dict(os.environ, {"MODELS_DIR": str(self.temp_models)}):
            settings = Settings()

            # Test Real-ESRGAN path resolution
            esrgan_path = settings.PATHS["models"] / "RealESRGAN_x4plus.pth"
            expected_path = self.temp_models / "RealESRGAN_x4plus.pth"

            assert esrgan_path == expected_path
            assert esrgan_path.parent == self.temp_models


class TestModelDiscovery:
    """Test model discovery and validation in warehouse"""

    def setup_method(self):
        self.temp_warehouse = tempfile.mkdtemp(prefix="model_discovery_test_")
        self.temp_models = Path(self.temp_warehouse) / "models"
        self.temp_models.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        if hasattr(self, "temp_warehouse") and Path(self.temp_warehouse).exists():
            shutil.rmtree(self.temp_warehouse)

    def test_esrgan_model_discovery(self):
        """Test ESRGAN processor finds models in warehouse"""
        # Create dummy model file
        dummy_model = self.temp_models / "RealESRGAN_x4plus.pth"
        dummy_model.write_bytes(b"dummy_model_data")

        with patch.dict(
            os.environ,
            {
                "MODELS_DIR": str(self.temp_models),
                "DEVICE": "cpu",  # Force CPU for testing
            },
        ):
            processor = ESRGANProcessor()

            # Should find the model in warehouse
            assert processor.model_path == dummy_model
            assert processor.model_path.exists()

    def test_missing_model_handling(self):
        """Test behavior when models are missing from warehouse"""
        with patch.dict(
            os.environ, {"MODELS_DIR": str(self.temp_models), "DEVICE": "cpu"}
        ):
            # No models in empty warehouse
            processor = ESRGANProcessor()

            # Should handle missing model gracefully
            with pytest.raises(FileNotFoundError, match="Model not found"):
                processor.load_model()


class TestModelDownloader:
    """Test model downloader works with warehouse paths"""

    def setup_method(self):
        self.temp_warehouse = tempfile.mkdtemp(prefix="downloader_test_")
        self.temp_models = Path(self.temp_warehouse) / "models"
        self.temp_models.mkdir(parents=True, exist_ok=True)

    def teardown_method(self):
        if hasattr(self, "temp_warehouse") and Path(self.temp_warehouse).exists():
            shutil.rmtree(self.temp_warehouse)

    def test_downloader_uses_warehouse_paths(self):
        """Test downloader saves to warehouse models directory"""
        with patch.dict(os.environ, {"MODELS_DIR": str(self.temp_models)}):
            downloader = ModelDownloader()

            # Should use warehouse models directory
            assert downloader.models_dir == self.temp_models

    @patch("scripts.download_models.download_file")
    def test_download_to_warehouse(self, mock_download):
        """Test models download to correct warehouse structure"""
        mock_download.return_value = True

        with patch.dict(os.environ, {"MODELS_DIR": str(self.temp_models)}):
            downloader = ModelDownloader()

            # Mock download essential models
            downloader.download_essential()

            # Should attempt download to warehouse
            mock_download.assert_called()
            call_args = mock_download.call_args_list[0]
            download_path = Path(call_args[0][1])  # Second argument is path

            assert download_path.parent == self.temp_models


class TestAPIWarehouseIntegration:
    """Test API integration with warehouse paths"""

    def setup_method(self):
        self.temp_warehouse = tempfile.mkdtemp(prefix="api_warehouse_test_")
        self.temp_models = Path(self.temp_warehouse) / "models"
        self.temp_cache = Path(self.temp_warehouse) / "cache"
        self.temp_models.mkdir(parents=True, exist_ok=True)
        self.temp_cache.mkdir(parents=True, exist_ok=True)

        # Create test client
        self.client = TestClient(app)

    def teardown_method(self):
        if hasattr(self, "temp_warehouse") and Path(self.temp_warehouse).exists():
            shutil.rmtree(self.temp_warehouse)

    def test_api_warehouse_health_check(self):
        """Test API returns warehouse status in health check"""
        with patch.dict(
            os.environ,
            {"MODELS_DIR": str(self.temp_models), "CACHE_DIR": str(self.temp_cache)},
        ):
            response = self.client.get("/api/v1/healthz")

            assert response.status_code == 200
            data = response.json()

            # Should include warehouse information
            assert "warehouse" in data
            assert data["warehouse"]["models_dir"] == str(self.temp_models)
            assert data["warehouse"]["cache_dir"] == str(self.temp_cache)

    def test_api_missing_warehouse_error(self):
        """Test API returns 500 when warehouse is missing"""
        nonexistent_path = "/nonexistent/warehouse/path"

        with patch.dict(os.environ, {"MODELS_DIR": nonexistent_path}):
            response = self.client.get("/api/v1/healthz")

            # Should return error for missing warehouse
            assert response.status_code == 500
            data = response.json()
            assert "warehouse" in data["detail"].lower()

    def test_api_models_endpoint_warehouse(self):
        """Test /models endpoint respects warehouse paths"""
        # Create dummy model
        dummy_model = self.temp_models / "RealESRGAN_x4plus.pth"
        dummy_model.write_bytes(b"dummy")

        with patch.dict(os.environ, {"MODELS_DIR": str(self.temp_models)}):
            response = self.client.get("/api/v1/models")

            assert response.status_code == 200
            data = response.json()

            # Should find model in warehouse
            assert any(
                model["name"] == "RealESRGAN_x4plus.pth"
                for model in data["available_models"]
            )


class TestWarehouseFallbacks:
    """Test fallback behaviors and error handling"""

    def test_default_warehouse_creation(self):
        """Test default warehouse is created in home directory"""
        with patch.dict(os.environ, {}, clear=True):
            # Clear all warehouse env vars
            for key in ["AI_WAREHOUSE_ROOT", "MODELS_DIR", "CACHE_DIR"]:
                os.environ.pop(key, None)

            settings = Settings()

            # Should default to ~/ai-warehouse
            expected_root = Path.home() / "ai-warehouse"
            assert settings.PATHS["warehouse_root"] == expected_root

    def test_partial_warehouse_config(self):
        """Test behavior with partial warehouse configuration"""
        temp_root = tempfile.mkdtemp(prefix="partial_warehouse_")

        with patch.dict(
            os.environ,
            {
                "AI_WAREHOUSE_ROOT": temp_root
                # Missing MODELS_DIR and CACHE_DIR
            },
        ):
            settings = Settings()

            # Should create default subdirectories
            assert settings.PATHS["models"] == Path(temp_root) / "models"
            assert settings.PATHS["cache"] == Path(temp_root) / "cache"
            assert all(path.exists() for path in settings.PATHS.values())

        shutil.rmtree(temp_root)

    def test_permission_error_handling(self):
        """Test handling of permission errors in warehouse"""
        with patch("pathlib.Path.mkdir") as mock_mkdir:
            mock_mkdir.side_effect = PermissionError("Access denied")

            with pytest.raises(PermissionError):
                Settings()


# Benchmarks and Integration Tests
class TestWarehousePerformance:
    """Test warehouse performance characteristics"""

    def test_path_resolution_performance(self):
        """Test path resolution is fast enough for production"""
        import time

        temp_warehouse = tempfile.mkdtemp(prefix="perf_test_")

        with patch.dict(os.environ, {"AI_WAREHOUSE_ROOT": temp_warehouse}):
            start_time = time.time()

            # Create settings 100 times
            for _ in range(100):
                settings = Settings()
                _ = settings.PATHS["models"] / "test_model.pth"

            elapsed = time.time() - start_time

            # Should be very fast (< 0.1s for 100 iterations)
            assert elapsed < 0.1, f"Path resolution too slow: {elapsed:.3f}s"

        shutil.rmtree(temp_warehouse)

    def test_concurrent_warehouse_access(self):
        """Test concurrent access to warehouse doesn't cause issues"""
        import threading
        import concurrent.futures

        temp_warehouse = tempfile.mkdtemp(prefix="concurrent_test_")

        def create_settings():
            with patch.dict(os.environ, {"AI_WAREHOUSE_ROOT": temp_warehouse}):
                settings = Settings()
                return settings.PATHS["models"].exists()

        # Test concurrent access
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(create_settings) for _ in range(20)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert all(results), "Concurrent warehouse access failed"

        shutil.rmtree(temp_warehouse)


if __name__ == "__main__":
    # Run basic smoke tests
    print("Running AI Warehouse smoke tests...")

    # Test 1: Basic path configuration
    test_paths = TestWarehousePaths()
    test_paths.setup_method()
    try:
        test_paths.test_settings_paths_configuration()
        print("✅ Path configuration test passed")
    finally:
        test_paths.teardown_method()

    # Test 2: Model discovery
    test_discovery = TestModelDiscovery()
    test_discovery.setup_method()
    try:
        test_discovery.test_missing_model_handling()
        print("✅ Model discovery test passed")
    finally:
        test_discovery.teardown_method()

    # Test 3: API integration
    test_api = TestAPIWarehouseIntegration()
    test_api.setup_method()
    try:
        test_api.test_api_warehouse_health_check()
        print("✅ API integration test passed")
    finally:
        test_api.teardown_method()

    print("🎉 All smoke tests passed! Warehouse integration is working.")
