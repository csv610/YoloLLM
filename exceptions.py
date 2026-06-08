class YoloLLMError(Exception):
    """Base exception for all YoloLLM errors."""


class ModelNotFoundError(YoloLLMError):
    """Raised when a requested model is not found in available models."""


class ModelFileNotFoundError(YoloLLMError):
    """Raised when a model file does not exist on disk."""


class NoResultsError(YoloLLMError):
    """Raised when attempting to access results before running inference."""


class ImageLoadError(YoloLLMError):
    """Raised when an image cannot be loaded from a given path or URL."""

