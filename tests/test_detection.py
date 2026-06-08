"""Unit tests for yolo_detect."""

from pathlib import Path

import numpy as np
import pytest

from yolo_detect import YoloDetector


class TestYoloDetectorInternals:
    """Tests for internal/static helper methods (no model loading required)."""

    def test_tensor_to_scalar_with_number(self):
        assert YoloDetector._tensor_to_scalar(42) == 42
        assert YoloDetector._tensor_to_scalar(3.14) == 3.14

    def test_tensor_to_scalar_with_array(self):
        arr = np.array([5.0])
        assert YoloDetector._tensor_to_scalar(arr) == 5.0

    def test_polygon_area_square(self):
        pts = [(0, 0), (4, 0), (4, 4), (0, 4)]
        area = YoloDetector._polygon_area(pts)
        assert area == pytest.approx(16.0)

    def test_polygon_area_triangle(self):
        pts = [(0, 0), (4, 0), (0, 3)]
        area = YoloDetector._polygon_area(pts)
        assert area == pytest.approx(6.0)

    def test_polygon_area_zero(self):
        pts = [(0, 0), (0, 0), (0, 0)]
        area = YoloDetector._polygon_area(pts)
        assert area == pytest.approx(0.0)

    def test_get_available_models_returns_dict(self):
        models = YoloDetector.get_available_models()
        assert isinstance(models, dict)
        assert len(models) > 0
        assert "v8-large" in models


class TestYoloDetectorResolution:
    """Tests for model name resolution logic."""

    def test_resolve_known_model(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.__init__("v8-large")
        expected = Path("models") / "yolov8l.pt"
        assert str(detector._resolve_model_path("v8-large")).endswith(str(expected))

    def test_resolve_unknown_model_uses_name_as_is(self):
        detector = YoloDetector.__new__(YoloDetector)
        detector.__init__("v8-large")
        path = detector._resolve_model_path("nonexistent.pt")
        assert str(path).endswith("nonexistent.pt")

