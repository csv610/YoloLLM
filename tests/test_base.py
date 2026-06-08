"""Unit tests for base model class."""


import pytest

from base import BaseYoloModel
from exceptions import ImageLoadError


class TestBaseYoloModel:
    def test_load_image_missing_file(self):
        model = BaseYoloModel.__new__(BaseYoloModel)
        with pytest.raises(ImageLoadError):
            model._load_image("/nonexistent/path.jpg")

    def test_get_available_models_returns_copy(self):
        assert BaseYoloModel.get_available_models() == {}

