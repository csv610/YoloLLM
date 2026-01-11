import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLOE


# Available YOLOE segmentation models (text prompt-based)
MODELS = {
    "11s": "yoloe-11s-seg.pt",
    "11m": "yoloe-11m-seg.pt",
    "11l": "yoloe-11l-seg.pt",
    "v8s": "yoloe-v8s-seg.pt",
    "v8m": "yoloe-v8m-seg.pt",
    "v8l": "yoloe-v8l-seg.pt",
}

# Prompt-free YOLOE segmentation models
PROMPT_FREE_MODELS = {
    "11s-pf": "yoloe-11s-seg-pf.pt",
    "11m-pf": "yoloe-11m-seg-pf.pt",
    "11l-pf": "yoloe-11l-seg-pf.pt",
    "v8s-pf": "yoloe-v8s-seg-pf.pt",
    "v8m-pf": "yoloe-v8m-seg-pf.pt",
    "v8l-pf": "yoloe-v8l-seg-pf.pt",
}

# Combine all models
ALL_MODELS = {**MODELS, **PROMPT_FREE_MODELS}


class YoloESegment:
    """YOLOE Instance Segmentation wrapper for text prompt-based and prompt-free models."""

    def __init__(self, model_name: str = "11l", classes: list = None):
        """
        Initialize YoloESegment with a specified model.

        Args:
            model_name: Model name key (default: "11l")
            classes: List of class names for text prompt-based models (optional for PF models)

        Examples:
            # Text prompt-based model
            yoloe = YoloESegment("11l", classes=["person", "car"])

            # Prompt-free model
            yoloe = YoloESegment("11l-pf")
        """
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Check if using prompt-free model
        self.is_prompt_free = model_name.endswith("-pf")

        # Look up the actual filename from the model name key
        if model_name in ALL_MODELS:
            model_file = ALL_MODELS[model_name]
        else:
            model_file = model_name

        model_path = self.models_dir / model_file
        self.model = YOLOE(str(model_path))

        # Set classes for text prompt-based models
        if not self.is_prompt_free:
            self.classes = classes if classes else ["person", "bus"]
            self.model.set_classes(self.classes, self.model.get_text_pe(self.classes))
        else:
            self.classes = None

        self.image = None
        self.results = None

    def segment(self, image_path: str):
        """
        Run segmentation on an image.

        Args:
            image_path: Path to the image file

        Returns:
            Segmentation results
        """
        self._load_image(image_path)
        self._infer()
        return self.results

    def _load_image(self, image_path: str) -> np.ndarray:
        """
        Load an image from file.

        Args:
            image_path: Path to the image file

        Returns:
            Loaded image as numpy array
        """
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        return self.image

    def _infer(self, image_path: str = None):
        """
        Run YOLOE inference on an image.

        Args:
            image_path: Path to image file. If None, uses previously loaded image.

        Returns:
            Inference results
        """
        if image_path is not None:
            self._load_image(image_path)

        if self.image is None:
            raise ValueError("No image loaded. Call segment() with an image_path.")

        self.results = self.model.predict(self.image)
        return self.results

    def show(self):
        """Display the segmentation results."""
        if self.results is None:
            raise ValueError("No results to show. Run segment() first.")
        self.results[0].show()

    def save(self, output_path: str):
        """
        Save the segmentation results to a file.

        Args:
            output_path: Path to save the output image
        """
        if self.results is None:
            raise ValueError("No results to save. Run segment() first.")
        self.results[0].save(filename=output_path)


def main():
    """Main function to run YoloESegment from command line."""
    parser = argparse.ArgumentParser(
        description="YOLOE Instance Segmentation (Text Prompt & Prompt-Free models)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available models:
  Text Prompt Models: {', '.join(MODELS.keys())}
  Prompt-Free Models: {', '.join(PROMPT_FREE_MODELS.keys())}

Examples:
  # Text prompt-based models (use --prompt to specify classes)
  python yoloe_segment.py image.jpg -m 11l -p person car --show

  # Prompt-free models (detect all COCO classes automatically)
  python yoloe_segment.py image.jpg -m 11l-pf --show

  # Save output with custom prompts
  python yoloe_segment.py image.jpg -m v8s -p dog cat bird -o output.jpg
        """
    )
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="11l",
        choices=list(ALL_MODELS.keys()),
        help="Model to use (default: 11l). Use -pf suffix for prompt-free models"
    )
    parser.add_argument(
        "-p", "--prompt",
        type=str,
        nargs="+",
        help="Text prompts for classes to detect (only for non-PF models, default: person bus)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="image_segment.png",
        help="Path to save the output image (optional)"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the result"
    )

    args = parser.parse_args()

    # Validate model name
    if args.model not in ALL_MODELS:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available models: {', '.join(ALL_MODELS.keys())}")
        return

    # Initialize YoloESegment
    print(f"Loading model: {ALL_MODELS[args.model]}")
    yoloe = YoloESegment(model_name=args.model, classes=args.prompt)

    if yoloe.is_prompt_free:
        print("Using prompt-free model (detects all COCO classes)")
    else:
        print(f"Using text prompts: {yoloe.classes}")

    # Run segmentation
    print(f"Running segmentation on: {args.image}")
    yoloe.segment(args.image)

    # Save results if output path provided
    if args.output:
        yoloe.save(args.output)
        print(f"Results saved to: {args.output}")

    # Show results if requested
    if args.show:
        yoloe.show()

    print("Done!")

if __name__ == "__main__":
    main()
