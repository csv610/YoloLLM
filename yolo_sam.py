import cv2
import numpy as np
import argparse
from pathlib import Path
from ultralytics import SAM
from mask_processing import save_masks, overlay_masks


# SAM 2 models
SAM2_MODELS = {
    "sam2-tiny": "sam2_t.pt",
    "sam2-small": "sam2_s.pt",
    "sam2-base": "sam2_b.pt",
    "sam2-large": "sam2_l.pt",
}

# SAM 2.1 models
SAM2_1_MODELS = {
    "sam2.1-tiny": "sam2.1_t.pt",
    "sam2.1-small": "sam2.1_s.pt",
    "sam2.1-base": "sam2.1_b.pt",
    "sam2.1-large": "sam2.1_l.pt",
}

# Mobile SAM model
MOBILE_SAM_MODELS = {
    "mobile-sam": "mobile_sam.pt",
}

# All available models
AVAILABLE_MODELS = {**SAM2_MODELS, **SAM2_1_MODELS, **MOBILE_SAM_MODELS}


class Segment:
    """Simple class to hold segmentation mask data."""
    def __init__(self, mask, label="segment"):
        self.mask = mask
        self.label = label


class YoloSAM:
    """Segment Anything Model (SAM) wrapper for image segmentation and visualization."""

    def __init__(self, model_name: str = "sam2.1-base"):
        """
        Initialize YoloSAM with a specified model.

        Args:
            model_name: Model name key (default: "sam2.1-base")

        Examples:
            yolo_sam = YoloSAM("sam2.1-base")
            yolo_sam = YoloSAM("sam2-large")
        """
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Look up the actual filename from the model name key
        if model_name in AVAILABLE_MODELS:
            model_name = AVAILABLE_MODELS[model_name]

        model_path = self.models_dir / model_name
        self.model = SAM(str(model_path))
        self.image = None
        self.original_image = None
        self.results = None

    def segment(self, image_path: str) -> list:
        """
        Run complete segmentation pipeline on an image.

        Args:
            image_path: Path to the image file

        Returns:
            List of Segment objects containing mask data
        """
        self._load_image(image_path)
        self._infer()
        return self._get_masks()

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array
        """
        self.image = cv2.imread(image_path)
        self.original_image = self.image.copy()
        return self.image

    def _infer(self, image_path: str = None) -> list:
        """
        Run SAM inference on an image.

        Args:
            image_path: Path to image file. If None, uses previously loaded image.

        Returns:
            List of inference results
        """
        if image_path is not None:
            self._load_image(image_path)

        if self.image is None:
            raise ValueError("No image loaded. Call segment() with an image_path.")

        self.results = self.model(self.image)
        return self.results

    def _get_masks(self) -> list:
        """
        Get list of segmentation masks as Segment objects, sorted by mask area (white pixel count).

        Returns:
            List of Segment objects containing mask data, sorted from largest to smallest
        """
        if self.results is None:
            raise ValueError("No inference results. Call infer() first.")

        segments = []
        mask_count = 0

        for r in self.results:
            if r.masks is None:
                continue

            masks = r.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                mask_count += 1
                segment = Segment(mask=mask, label=f"mask_{mask_count}")
                segments.append(segment)

        # Sort by count of white pixels (mask area) in descending order
        segments.sort(key=lambda s: np.sum(s.mask > 0), reverse=True)

        return segments


def main():
    """Main function to run YoloSAM from command line."""
    parser = argparse.ArgumentParser(
        description="Segment Anything Model (SAM) for image segmentation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  {', '.join(AVAILABLE_MODELS.keys())}

Examples:
  python yolo_sam.py -i image.jpg
  python yolo_sam.py -i image.jpg -m sam2.1-large
  python yolo_sam.py -i image.jpg -m sam2.1-base -o result.jpg -t 0.7
  python yolo_sam.py -i image.jpg -s -d my_masks
        """
    )

    parser.add_argument(
        "-i", "--image",
        type=str,
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "-m", "--model",
        type=str,
        default="sam2.1-base",
        help=f"Model to use (default: sam2.1-base). Options: {', '.join(AVAILABLE_MODELS.keys())}"
    )

    parser.add_argument(
        "-o", "--output",
        type=str,
        default="sam_segment.png",
        help="Path to save output image (default: sam_segment.png)"
    )

    parser.add_argument(
        "-t", "--transparency",
        type=float,
        default=0.25,
        help="Mask transparency from 0.0 (no transparency, opaque masks) to 1.0 (full transparency, invisible masks) (default: 0.25)"
    )

    parser.add_argument(
        "-s", "--save-masks",
        action="store_true",
        help="Save individual masks as separate PNG files"
    )

    parser.add_argument(
        "-d", "--masks-dir",
        type=str,
        default="masks",
        help="Directory to save individual masks (default: masks)"
    )

    args = parser.parse_args()

    # Validate model name
    if args.model not in AVAILABLE_MODELS:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
        return

    # Run segmentation
    yolo_sam = YoloSAM(args.model)

    print(f"Using model: {AVAILABLE_MODELS[args.model]}")
    print(f"Segmenting image: {args.image}")
    segments = yolo_sam.segment(args.image)
    print(f"Found {len(segments)} segments")

    # Save individual masks if requested
    if args.save_masks:
        print(f"Saving individual masks to {args.masks_dir}...")
        save_masks(segments, output_dir=args.masks_dir)

    # Create overlay using mask_processing
    print(f"Overlaying masks with transparency {args.transparency}...")
    overlay_masks(segments, args.image, output_path=args.output, alpha=args.transparency)

    print("Done!")


if __name__ == "__main__":
    main()

