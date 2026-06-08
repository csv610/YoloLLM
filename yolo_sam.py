import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ultralytics import SAM

from base import BaseYoloModel
from config import SEG_RESULTS_DIR
from exceptions import NoResultsError
from mask_processing import Segment, overlay_masks, save_masks

logger = logging.getLogger(__name__)


class YoloSAM(BaseYoloModel):
    model_class = SAM

    SAM2_MODELS: Dict[str, str] = {
        "sam2-tiny": "sam2_t.pt",
        "sam2-small": "sam2_s.pt",
        "sam2-base": "sam2_b.pt",
        "sam2-large": "sam2_l.pt",
    }

    SAM2_1_MODELS: Dict[str, str] = {
        "sam2.1-tiny": "sam2.1_t.pt",
        "sam2.1-small": "sam2.1_s.pt",
        "sam2.1-base": "sam2.1_b.pt",
        "sam2.1-large": "sam2.1_l.pt",
    }

    MOBILE_SAM_MODELS: Dict[str, str] = {
        "mobile-sam": "mobile_sam.pt",
    }

    available_models: Dict[str, str] = {
        **SAM2_MODELS,
        **SAM2_1_MODELS,
        **MOBILE_SAM_MODELS,
    }

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name, default_model="sam2.1-base")
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
        """Run SAM inference.

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
        description="Segment Anything Model (SAM) for image segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(YoloSAM.available_models.keys())}

Examples:
  python yolo_sam.py -i image.jpg
  python yolo_sam.py -i image.jpg -m sam2.1-large
  python yolo_sam.py -i image.jpg -m sam2.1-base -o result.jpg -t 0.7
  python yolo_sam.py -i image.jpg -s -d my_masks
        """,
    )

    parser.add_argument("-i", "--image", type=str, required=True, help="Path to input image")
    parser.add_argument(
        "-m", "--model", type=str, default="sam2.1-base",
        help="Model to use (default: sam2.1-base). Options: " + ", ".join(YoloSAM.available_models.keys()),
    )
    parser.add_argument(
        "-o", "--output", type=str, default=str(SEG_RESULTS_DIR / "sam_segment.png"),
        help="Path to save output image (default: results/seg/sam_segment.png)",
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

    if args.model not in YoloSAM.available_models:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available models: {', '.join(YoloSAM.available_models.keys())}")
        return

    yolo_sam = YoloSAM(args.model)
    logger.info("Using model: %s", YoloSAM.available_models[args.model])
    logger.info("Segmenting image: %s", args.image)
    segments = yolo_sam.segment(args.image)
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

