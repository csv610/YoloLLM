import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import FastSAM

from base import BaseYoloModel
from config import SEG_RESULTS_DIR
from exceptions import NoResultsError
from mask_processing import Segment, overlay_masks, save_masks

logger = logging.getLogger(__name__)


class YoloFastSAM(BaseYoloModel):
    model_class = FastSAM

    FASTSAM_MODELS: Dict[str, str] = {
        "fastsam-small": "FastSAM-s.pt",
        "fastsam-large": "FastSAM-x.pt",
    }

    available_models: Dict[str, str] = {
        **FASTSAM_MODELS,
    }

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name, default_model="fastsam-small")
        self.image: Optional[np.ndarray] = None
        self.original_image: Optional[np.ndarray] = None

    def segment(self, image_path: str) -> List[Segment]:
        """Run the full segmentation pipeline on an image.

        Args:
            image_path: Path to the image file.

        Returns:
            List of Segment objects sorted by mask area (largest first).
        """
        self._load_image(image_path)
        self._infer()
        return self._get_masks()

    def _load_image(self, image_path: str) -> np.ndarray:
        self.image = super()._load_image(image_path)
        self.original_image = self.image.copy()
        return self.image

    def _infer(self, image_path: Optional[str] = None) -> List:
        """Run FastSAM inference.

        Args:
            image_path: Load this image first if provided.

        Returns:
            Inference results.
        """
        if image_path is not None:
            self._load_image(image_path)

        if self.image is None:
            raise NoResultsError("No image loaded. Call segment() with an image_path.")

        self.results = self.model(self.image)
        return self.results

    def _get_masks(self) -> List[Segment]:
        """Extract masks from results, sorted by area descending."""
        if self.results is None:
            raise NoResultsError("No inference results. Call segment() first.")

        segments: List[Segment] = []
        mask_count = 0

        for r in self.results:
            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()
            for i, mask in enumerate(masks):
                mask_count += 1
                segments.append(Segment(mask=mask, label=f"mask_{mask_count}"))

        segments.sort(key=lambda s: np.sum(s.mask > 0), reverse=True)
        return segments


def main():
    parser = argparse.ArgumentParser(
        description="FastSAM for image segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(YoloFastSAM.available_models.keys())}

Examples:
  python yolo_fastsam.py -i image.jpg
  python yolo_fastsam.py -i image.jpg -m fastsam-large
  python yolo_fastsam.py -i image.jpg -m fastsam-small -o result.jpg -t 0.7
  python yolo_fastsam.py -i image.jpg -s -d my_masks
        """,
    )

    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "-m", "--model", type=str, default="fastsam-small",
        help="Model to use (default: fastsam-small). Options: " + ", ".join(YoloFastSAM.available_models.keys()),
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(SEG_RESULTS_DIR / "fastsam_segment.png"),
        help="Path to save output image (default: results/seg/fastsam_segment.png)",
    )
    parser.add_argument(
        "-t", "--transparency", type=float, default=0.25,
        help="Mask transparency 0 (opaque) to 1 (invisible) (default: 0.25)",
    )
    parser.add_argument("-s", "--save-masks", action="store_true", help="Save individual masks as PNG files")
    parser.add_argument(
        "-d", "--masks-dir", type=str, default=str(SEG_RESULTS_DIR / "masks"),
        help="Directory for individual masks (default: results/seg/masks)",
    )

    args = parser.parse_args()

    if args.model not in YoloFastSAM.available_models:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available models: {', '.join(YoloFastSAM.available_models.keys())}")
        return

    yolo_fastsam = YoloFastSAM(args.model)
    logger.info("Using model: %s", YoloFastSAM.available_models[args.model])
    logger.info("Segmenting image: %s", args.image)
    segments = yolo_fastsam.segment(args.image)
    logger.info("Found %d segments", len(segments))

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        save_masks(segments, output_dir=args.masks_dir)

    overlay_masks(segments, args.image, output_path=args.output, alpha=args.transparency)
    print("Done!")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

