import pytest
import tempfile
from pathlib import Path
import numpy as np
import cv2

from core.esrgan import ESRGAN
from core.common.io import ImageIO
from api.config import get_settings


class TestESRGAN:
    """Test ESRGAN adapter functionality"""

    @pytest.fixture
    def test_image(self):
        """Create a test image"""
        # Create a simple test image
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img

    @pytest.fixture
    def esrgan_model(self):
        """Create ESRGAN model instance"""
        return ESRGAN(model_name="realesrgan-x2plus", device="cpu")

    def test_model_initialization(self, esrgan_model):
        """Test model initialization"""
        assert esrgan_model.model_name == "realesrgan-x2plus"
        assert esrgan_model.device == "cpu"
        assert esrgan_model.scale == 2

    def test_tile_processor_initialization(self, esrgan_model):
        """Test tile processor setup"""
        assert esrgan_model.tile_processor is not None
        assert esrgan_model.tile_processor.tile_size == get_settings().tile_size

    def test_image_processing(self, esrgan_model, test_image):
        """Test basic image processing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save test image
            input_path = Path(temp_dir) / "test_input.png"
            output_path = Path(temp_dir) / "test_output.png"

            cv2.imwrite(str(input_path), test_image)

            # Process image
            result = esrgan_model.process_image(input_path, output_path)

            # Verify results
            assert output_path.exists()
            assert result["original_width"] == 100
            assert result["original_height"] == 100
            assert result["processed_width"] == 200  # 2x scale
            assert result["processed_height"] == 200
            assert result["scale"] == 2

    def test_quality_metrics(self, esrgan_model, test_image):
        """Test quality metrics calculation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            input_path = Path(temp_dir) / "test_input.png"
            output_path = Path(temp_dir) / "test_output.png"

            cv2.imwrite(str(input_path), test_image)

            # Process with quality metrics
            result = esrgan_model.process_image(
                input_path, output_path, quality_metrics=True
            )

            # Check if metrics are calculated
            assert "quality_metrics" in result
            metrics = result["quality_metrics"]
            assert "sharpness" in metrics

    def test_different_scales(self):
        """Test different scale factors"""
        scales = [2, 4]

        for scale in scales:
            model_name = f"realesrgan-x{scale}plus"
            model = ESRGAN(model_name=model_name, device="cpu")
            assert model.scale == scale


class TestImageIO:
    """Test image I/O functionality"""

    def test_image_read_write(self):
        """Test image reading and writing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test image
            test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
            input_path = Path(temp_dir) / "test.png"
            output_path = Path(temp_dir) / "test_out.png"

            # Write image
            cv2.imwrite(str(input_path), test_image)

            # Read image
            img, metadata = ImageIO.read_image(input_path)

            # Verify read image
            assert img.shape == (50, 50, 3)
            assert "file_size" in metadata
            assert "file_hash" in metadata

            # Write image back
            ImageIO.write_image(img, output_path)
            assert output_path.exists()

    def test_image_resize(self):
        """Test image resizing"""
        test_image = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)

        # Resize to smaller dimensions
        resized = ImageIO.resize_image(test_image, 50, 50)
        assert resized.shape[0] <= 50
        assert resized.shape[1] <= 50

        # Aspect ratio should be preserved
        original_ratio = 200 / 100
        new_ratio = resized.shape[1] / resized.shape[0]
        assert abs(original_ratio - new_ratio) < 0.1
