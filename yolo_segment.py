import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ultralytics import YOLO

from base import BaseYoloModel
from config import SEG_RESULTS_DIR
from exceptions import ModelNotFoundError
from mask_processing import Segment, overlay_masks, save_masks

logger = logging.getLogger(__name__)


class YoloSegmentation(BaseYoloModel):
    model_class = YOLO

    YOLOV8_MODELS: Dict[str, str] = {
        "v8-nano": "yolov8n-seg.pt",
        "v8-small": "yolov8s-seg.pt",
        "v8-medium": "yolov8m-seg.pt",
        "v8-large": "yolov8l-seg.pt",
        "v8-xlarge": "yolov8x-seg.pt",
    }

    YOLOV11_MODELS: Dict[str, str] = {
        "v11-nano": "yolo11n-seg.pt",
        "v11-small": "yolo11s-seg.pt",
        "v11-medium": "yolo11m-seg.pt",
        "v11-large": "yolo11l-seg.pt",
        "v11-xlarge": "yolo11x-seg.pt",
    }

    YOLOV26_MODELS: Dict[str, str] = {
        "v26-nano": "yolo26n-seg.pt",
        "v26-small": "yolo26s-seg.pt",
        "v26-medium": "yolo26m-seg.pt",
        "v26-large": "yolo26l-seg.pt",
        "v26-xlarge": "yolo26x-seg.pt",
    }

    YOLOE26_MODELS: Dict[str, str] = {
        "ve26-nano": "yoloe26n-seg.pt",
        "ve26-small": "yoloe26s-seg.pt",
        "ve26-medium": "yoloe26m-seg.pt",
        "ve26-large": "yoloe26l-seg.pt",
        "ve26-xlarge": "yoloe26x-seg.pt",
    }

    available_models: Dict[str, str] = {
        **YOLOV8_MODELS,
        **YOLOV11_MODELS,
        **YOLOV26_MODELS,
        **YOLOE26_MODELS,
    }

    def __init__(self, model_name: Optional[str] = None):
        if model_name is not None and model_name not in self.available_models:
            raise ModelNotFoundError(
                f"Model '{model_name}' not found. Available models: {list(self.available_models.keys())}"
            )
        super().__init__(model_name, default_model="v11-nano")

    def segment(self, image_path: str) -> List[Segment]:
        """Run segmentation on an image.

        Args:
            image_path: Path or URL to the image.

        Returns:
            List of Segment objects containing binary mask images.
        """
        logger.info("Running segmentation on: %s", image_path)
        results = self.model(image_path)
        return self._extract_segments(results)

    @staticmethod
    def _extract_segments(results) -> List[Segment]:
        """Extract Segment objects from YOLO inference results."""
        segments: List[Segment] = []

        for result in results:
            if result.masks is None:
                continue

            masks = result.masks.data
            class_indices = result.boxes.cls.int().tolist()

            if hasattr(masks, "cpu"):
                masks = masks.cpu().numpy()

            for mask, cls_idx in zip(masks, class_indices):
                label = result.names[int(cls_idx)]
                segments.append(Segment(mask=mask, label=label))

        return segments

    @staticmethod
    def print_results(segments: List[Segment]) -> None:
        """Print segmentation results to the console."""
        print("\nSegmentation Results:")
        if not segments:
            print("  No objects detected")
            return

        print(f"  Total objects detected: {len(segments)}")
        for i, segment in enumerate(segments):
            print(f"  Object {i}: {segment.label} (mask shape: {segment.mask.shape})")


def main():
    parser = argparse.ArgumentParser(description="YOLO Segmentation (v8, v11, v26, YOLOE-26)")
    parser.add_argument(
        "-m", "--model",
        choices=list(YoloSegmentation.available_models.keys()),
        default="v11-nano",
        help="Model version and size (default: v11-nano)",
    )
    parser.add_argument(
        "-i", "--image",
        default="https://ultralytics.com/images/bus.jpg",
        help="Image path or URL (default: bus.jpg from ultralytics)",
    )
    parser.add_argument(
        "-o", "--output-dir",
        metavar="OUTPUT_DIR",
        default=str(SEG_RESULTS_DIR),
        help=f"Output directory (default: {SEG_RESULTS_DIR})",
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.6,
        help="Transparency of masks in overlay (0-1, default: 0.6)",
    )
    parser.add_argument(
        "--overlay-filename",
        default="overlay.png",
        help="Filename for overlay image (default: overlay.png)",
    )

    args = parser.parse_args()

    segmentation = YoloSegmentation(args.model)
    results = segmentation.segment(args.image)
    segmentation.print_results(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    save_masks(results, output_dir=str(output_dir))
    overlay_path = output_dir / args.overlay_filename
    overlay_masks(results, args.image, output_path=str(overlay_path), alpha=args.alpha)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

