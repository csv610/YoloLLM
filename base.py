import logging
from abc import ABC
from pathlib import Path
from typing import Any, ClassVar, Optional

import cv2
import numpy as np

from config import MODELS_DIR
from exceptions import ImageLoadError, NoResultsError

logger = logging.getLogger(__name__)


class BaseYoloModel(ABC):
    """Abstract base class for all YOLO/SAM model wrappers.

    Subclasses must set :attr:`model_class` and :attr:`available_models`.
    """

    model_class: ClassVar[type]
    available_models: ClassVar[dict[str, str]] = {}

    def __init__(
        self,
        model_name: Optional[str] = None,
        default_model: str = "",
    ):
        MODELS_DIR.mkdir(exist_ok=True)

        if model_name is None:
            model_name = default_model

        model_path = self._resolve_model_path(model_name)
        logger.info("Loading %s model (%s)...", model_name, model_path)
        self.model = self.model_class(str(model_path))
        self.results: Any = None

    def _resolve_model_path(self, name: str) -> Path:
        """Resolve a model name key to a full filesystem path.

        If *name* is a known model key it is replaced with the corresponding
        filename first.  The result is always joined to the shared *models/*
        directory.
        """
        if name in self.available_models:
            name = self.available_models[name]
        return MODELS_DIR / name

    def _check_results(self) -> None:
        """Raise :exc:`NoResultsError` if inference has not been run yet."""
        if self.results is None:
            raise NoResultsError("No results available. Run inference first.")

    def _load_image(self, image_path: str) -> np.ndarray:
        """Read an image from disk and return it as a numpy array.

        Raises :exc:`ImageLoadError` if the image cannot be read.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ImageLoadError(f"Failed to load image: {image_path}")
        return image

    @classmethod
    def get_available_models(cls) -> dict[str, str]:
        """Return a copy of the available models dictionary."""
        return dict(cls.available_models)

