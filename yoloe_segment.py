import argparse
import logging
from typing import Dict, List, Optional

import cv2
import numpy as np
from ultralytics import YOLOE

from base import BaseYoloModel
from exceptions import ImageLoadError, NoResultsError

logger = logging.getLogger(__name__)


class YoloESegment(BaseYoloModel):
    model_class = YOLOE

    MODELS: Dict[str, str] = {
        "yoloe11s": "yoloe-11s-seg.pt",
        "yoloe11m": "yoloe-11m-seg.pt",
        "yoloe11l": "yoloe-11l-seg.pt",
        "yoloev8s": "yoloe-v8s-seg.pt",
        "yoloev8m": "yoloe-v8m-seg.pt",
        "yoloev8l": "yoloe-v8l-seg.pt",
    }

    PROMPT_FREE_MODELS: Dict[str, str] = {
        "yoloe11s-pf": "yoloe-11s-seg-pf.pt",
        "yoloe11m-pf": "yoloe-11m-seg-pf.pt",
        "yoloe11l-pf": "yoloe-11l-seg-pf.pt",
        "yoloev8s-pf": "yoloe-v8s-seg-pf.pt",
        "yoloev8m-pf": "yoloe-v8m-seg-pf.pt",
        "yoloev8l-pf": "yoloe-v8l-seg-pf.pt",
    }

    available_models: Dict[str, str] = {
        **MODELS,
        **PROMPT_FREE_MODELS,
    }

    def __init__(
        self,
        model_name: Optional[str] = None,
        classes: Optional[List[str]] = None,
    ):
        super().__init__(model_name, default_model="yoloe11l")

        self.is_prompt_free = model_name is not None and model_name.endswith("-pf")

        if not self.is_prompt_free:
            self.classes: List[str] = classes if classes else ["person", "bus"]
            self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))
        else:
            self.classes = None

        self.image: Optional[np.ndarray] = None
        self.results = None

    def segment(self, image_path: str):
        """Run segmentation on an image.

        Args:
            image_path: Path to the image file.

        Returns:
            YOLOE inference results.
        """
        self._load_image(image_path)
        self._infer()
        return self.results

    def _load_image(self, image_path: str) -> np.ndarray:
        image = cv2.imread(image_path)
        if image is None:
            raise ImageLoadError(f"Failed to load image: {image_path}")
        self.image = image
        return self.image

    def _infer(self, image_path: Optional[str] = None):
        """Run YOLOE inference.

        Args:
            image_path: Load this image first if provided.
        """
        if image_path is not None:
            self._load_image(image_path)

        if self.image is None:
            raise NoResultsError("No image loaded. Call segment() with an image_path.")

        self.results = self.model.predict(self.image)
        return self.results

    def show(self) -> None:
        """Display the segmentation results."""
        if self.results is None:
            raise NoResultsError("No results to show. Run segment() first.")
        self.results[0].show()

    def save(self, output_path: str) -> None:
        """Save the segmentation results to a file."""
        if self.results is None:
            raise NoResultsError("No results to save. Run segment() first.")
        self.results[0].save(filename=output_path)


def main():
    parser = argparse.ArgumentParser(
        description="YOLOE Instance Segmentation (Text Prompt & Prompt-Free models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  Text Prompt Models: {', '.join(YoloESegment.MODELS.keys())}
  Prompt-Free Models: {', '.join(YoloESegment.PROMPT_FREE_MODELS.keys())}

Examples:
  python yoloe_segment.py image.jpg -m yoloe11l -p person car --show
  python yoloe_segment.py image.jpg -m yoloe11l-pf --show
  python yoloe_segment.py image.jpg -m yoloev8s -p dog cat bird -o output.jpg
        """,
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "-m", "--model", type=str, default="yoloe11l",
        choices=list(YoloESegment.available_models.keys()),
        help="Model to use (default: yoloe11l). Use -pf suffix for prompt-free models",
    )
    parser.add_argument(
        "-p", "--prompt", type=str, nargs="+",
        help="Text prompts for classes (only for non-PF models, default: person bus)",
    )
    parser.add_argument("-o", "--output", type=str, default="image_segment.png", help="Path to save output image")
    parser.add_argument("--show", action="store_true", help="Display the result")

    args = parser.parse_args()

    if args.model not in YoloESegment.available_models:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available models: {', '.join(YoloESegment.available_models.keys())}")
        return

    logger.info("Loading model: %s", YoloESegment.available_models[args.model])
    yoloe = YoloESegment(model_name=args.model, classes=args.prompt)

    if yoloe.is_prompt_free:
        logger.info("Using prompt-free model (detects all COCO classes)")
    else:
        logger.info("Using text prompts: %s", yoloe.classes)

    logger.info("Running segmentation on: %s", args.image)
    yoloe.segment(args.image)

    if args.output:
        yoloe.save(args.output)
        logger.info("Results saved to: %s", args.output)

    if args.show:
        yoloe.show()

    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

