import argparse
import os
import shutil
import numpy as np
from ultralytics import YOLO
from mask_processing import save_masks, overlay_masks

# YOLOv8 segmentation models
YOLOV8_MODELS = {
    "v8-nano": "yolov8n-seg.pt",
    "v8-small": "yolov8s-seg.pt",
    "v8-medium": "yolov8m-seg.pt",
    "v8-large": "yolov8l-seg.pt",
    "v8-xlarge": "yolov8x-seg.pt",
}

# YOLOv11 segmentation models
YOLOV11_MODELS = {
    "v11-nano": "yolo11n-seg.pt",
    "v11-small": "yolo11s-seg.pt",
    "v11-medium": "yolo11m-seg.pt",
    "v11-large": "yolo11l-seg.pt",
    "v11-xlarge": "yolo11x-seg.pt",
}

# Combined models
MODELS = {**YOLOV8_MODELS, **YOLOV11_MODELS}


class Segment:
    """Represents a single segmented object with its label and binary mask."""

    def __init__(self, label, mask):
        """
        Initialize a Segment.

        Args:
            label: Class name of the detected object
            mask: Binary segmentation mask image (numpy array with 0s and 1s)
        """
        self.label = label
        self.mask = mask


class YoloSegmentation:
    """YOLO Segmentation class for running image segmentation."""

    def __init__(self, model_name="v11-nano"):
        """
        Initialize the YoloSegmentation model.

        Args:
            model_name: Key from MODELS dictionary (default: v11-nano)

        Raises:
            ValueError: If model_name is not in MODELS or model file not found in models directory
        """
        if model_name not in MODELS:
            raise ValueError(
                f"Model '{model_name}' not found. Available models: {list(MODELS.keys())}"
            )

        self.model_name = model_name
        model_filename = MODELS[model_name]
        self.model_file = os.path.join("models", model_filename)

        if not os.path.exists(self.model_file):
            raise ValueError(
                f"Model file not found at {self.model_file}. "
                f"Please download {model_filename} and place it in the models/ directory."
            )

        print(f"Loading {model_name} model ({self.model_file})...")
        self.model = YOLO(self.model_file)

    def segment(self, image_path):
        """
        Run segmentation on an image.

        Args:
            image_path: Path or URL to the image

        Returns:
            List of Segment objects, each containing a label and binary mask image
        """
        print(f"Running segmentation on: {image_path}")
        results = self.model(image_path)
        return self._extract_segments(results)

    def _extract_segments(self, results):
        """
        Extract Segment objects from YOLO results.

        Args:
            results: Results from YOLO inference

        Returns:
            List of Segment objects with binary mask images
        """
        segments = []

        for result in results:
            if result.masks is not None:
                masks = result.masks.data  # (num_objects, H, W)
                class_indices = result.boxes.cls.int().tolist()

                # Convert to numpy if needed
                if hasattr(masks, 'cpu'):
                    masks = masks.cpu().numpy()

                # Create Segment object for each detected object
                for mask, cls_idx in zip(masks, class_indices):
                    label = result.names[int(cls_idx)]
                    segments.append(Segment(label, mask))

        return segments

    def print_results(self, segments):
        """
        Print segmentation results to console.

        Args:
            segments: List of Segment objects from segment() method
        """
        print("\nSegmentation Results:")
        if not segments:
            print("  No objects detected")
            return

        print(f"  Total objects detected: {len(segments)}")
        for i, segment in enumerate(segments):
            print(f"  Object {i}: {segment.label} (mask shape: {segment.mask.shape})")


    @staticmethod
    def get_available_models():
        """
        Get list of available models.

        Returns:
            Dictionary of available models
        """
        return MODELS


def main():
    parser = argparse.ArgumentParser(description="YOLO Segmentation (v8 and v11)")
    parser.add_argument(
        "-m", "--model",
        choices=MODELS.keys(),
        default="v11-nano",
        help="Model version and size (default: v11-nano)"
    )
    parser.add_argument(
        "-i", "--image",
        default="https://ultralytics.com/images/bus.jpg",
        help="Image path or URL (default: bus.jpg from ultralytics)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        metavar="OUTPUT_DIR",
        default="seg_results",
        help="Output directory for masks and overlay images (default: seg_results)"
    )
    parser.add_argument(
        "-a", "--alpha",
        type=float,
        default=0.6,
        help="Transparency of masks in overlay (0-1, default: 0.6)"
    )
    parser.add_argument(
        "--overlay-filename",
        default="overlay.png",
        help="Filename for overlay image (default: overlay.png)"
    )

    args = parser.parse_args()

    # Create segmentation instance and run
    segmentation = YoloSegmentation(args.model)
    results = segmentation.segment(args.image)
    segmentation.print_results(results)

    # Save masks and overlay if output directory is provided
    if args.output_dir:
        # Clear output directory if it exists
        if os.path.exists(args.output_dir):
            shutil.rmtree(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        save_masks(results, output_dir=args.output_dir)
        overlay_path = os.path.join(args.output_dir, args.overlay_filename)
        overlay_masks(results, args.image, output_path=overlay_path, alpha=args.alpha)


if __name__ == "__main__":
    main()
