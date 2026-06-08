"""Pytest fixtures and shared configuration."""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def mock_torch_load():
    """Prevent actual torch model loading during tests that import YOLO."""
    with patch("torch.load", MagicMock()):
        yield


@pytest.fixture
def mock_yolo():
    """Return a MagicMock that stands in for ultralytics.YOLO."""
    mock = MagicMock()
    mock.return_value = mock
    return mock

