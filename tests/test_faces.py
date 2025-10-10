import pytest
import numpy as np
import cv2

from core.common.faces import FaceProcessor


class TestFaceProcessor:
    """Test face processing functionality"""

    @pytest.fixture
    def face_processor(self):
        return FaceProcessor(min_face_size=30)

    @pytest.fixture
    def test_image_with_faces(self):
        """Create a test image with synthetic faces"""
        img = np.ones((200, 200, 3), dtype=np.uint8) * 128

        # Draw simple face-like rectangles
        cv2.rectangle(img, (50, 50), (80, 80), (255, 255, 255), -1)  # Face 1
        cv2.rectangle(img, (120, 70), (150, 100), (255, 255, 255), -1)  # Face 2

        return img

    @pytest.fixture
    def test_image_no_faces(self):
        """Create a test image without faces"""
        return np.ones((100, 100, 3), dtype=np.uint8) * 128

    def test_face_detection(self, face_processor, test_image_with_faces):
        """Test face detection"""
        faces = face_processor.detect_faces(test_image_with_faces)

        # Should detect at least one face
        assert len(faces) >= 1

        # Check face rectangle format
        for face in faces:
            assert len(face) == 4  # x, y, w, h
            assert all(isinstance(coord, int) for coord in face)

    def test_no_face_detection(self, face_processor, test_image_no_faces):
        """Test case with no faces"""
        faces = face_processor.detect_faces(test_image_no_faces)
        assert len(faces) == 0

    def test_face_counting(self, face_processor, test_image_with_faces):
        """Test face counting"""
        count = face_processor.count_faces(test_image_with_faces)
        assert count >= 1

    def test_face_alignment(self, face_processor, test_image_with_faces):
        """Test face alignment"""
        faces = face_processor.detect_faces(test_image_with_faces)

        if faces:
            aligned = face_processor.align_face(test_image_with_faces, faces[0])
            assert aligned is not None
            assert aligned.shape == (512, 512, 3)  # Default output size

    def test_face_quality_metrics(self, face_processor, test_image_with_faces):
        """Test face quality metrics calculation"""
        faces = face_processor.detect_faces(test_image_with_faces)

        if faces:
            metrics = face_processor.get_face_quality_metrics(
                test_image_with_faces, faces[0]
            )

            assert "sharpness" in metrics
            assert "brightness" in metrics
            assert "contrast" in metrics

            # Metrics should be numeric
            for value in metrics.values():
                assert isinstance(value, (int, float))
