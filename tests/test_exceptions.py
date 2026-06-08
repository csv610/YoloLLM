"""Unit tests for custom exceptions."""

import pytest

from exceptions import (
    ImageLoadError,
    ModelFileNotFoundError,
    ModelNotFoundError,
    NoResultsError,
    YoloLLMError,
)


class TestExceptions:
    def test_yolollm_error(self):
        with pytest.raises(YoloLLMError):
            raise ModelNotFoundError("test")

    def test_model_not_found(self):
        err = ModelNotFoundError("model x not found")
        assert "model x not found" in str(err)

    def test_no_results_error(self):
        err = NoResultsError("no results")
        assert "no results" in str(err)

    def test_image_load_error(self):
        err = ImageLoadError("cannot load")
        assert "cannot load" in str(err)

    def test_model_file_not_found(self):
        err = ModelFileNotFoundError("file missing")
        assert "file missing" in str(err)

