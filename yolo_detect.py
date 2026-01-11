
import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
from tabulate import tabulate


class YoloDetector:
    # Output format constants
    FORMAT_TABLE = "table"
    FORMAT_JSON = "json"

    # YOLOv10 Models
    YOLOV10_MODELS = {
        "v10-nano": "yolov10n.pt",
        "v10-small": "yolov10s.pt",
        "v10-medium": "yolov10m.pt",
        "v10-base": "yolov10b.pt",
        "v10-large": "yolov10l.pt",
        "v10-xlarge": "yolov10x.pt",
    }

    # YOLOv9 Models
    YOLOV9_MODELS = {
        "v9-tiny": "yolov9t.pt",
        "v9-small": "yolov9s.pt",
        "v9-medium": "yolov9m.pt",
        "v9-compact": "yolov9c.pt",
        "v9-extra": "yolov9e.pt",
    }

    # YOLOv8 Models
    YOLOV8_MODELS = {
        "v8-nano": "yolov8n.pt",
        "v8-small": "yolov8s.pt",
        "v8-medium": "yolov8m.pt",
        "v8-large": "yolov8l.pt",
        "v8-xlarge": "yolov8x.pt",
    }

    # YOLOv8 World Models
    YOLOV8_WORLD_MODELS = {
        "v8s-world": "yolov8s-world.pt",
        "v8s-worldv2": "yolov8s-worldv2.pt",
        "v8m-world": "yolov8m-world.pt",
        "v8m-worldv2": "yolov8m-worldv2.pt",
        "v8l-world": "yolov8l-world.pt",
        "v8l-worldv2": "yolov8l-worldv2.pt",
        "v8x-world": "yolov8x-world.pt",
        "v8x-worldv2": "yolov8x-worldv2.pt",
    }

    # RT-DETR Models
    RTDETR_MODELS = {
        "rtdetr-large": "rtdetr-l.pt",
        "rtdetr-xlarge": "rtdetr-x.pt",
    }

    # YOLO11 Models
    YOLO11_MODELS = {
        "v11-nano": "yolo11n.pt",
        "v11-small": "yolo11s.pt",
        "v11-medium": "yolo11m.pt",
        "v11-large": "yolo11l.pt",
        "v11-xlarge": "yolo11x.pt",
    }

    # YOLO12 Models
    YOLO12_MODELS = {
        "v12-nano": "yolo12n.pt",
        "v12-small": "yolo12s.pt",
        "v12-medium": "yolo12m.pt",
        "v12-large": "yolo12l.pt",
        "v12-xlarge": "yolo12x.pt",
    }

    # Combined: All Available Models
    AVAILABLE_MODELS = {
        **YOLOV10_MODELS,
        **YOLOV9_MODELS,
        **YOLOV8_MODELS,
        **YOLOV8_WORLD_MODELS,
        **RTDETR_MODELS,
        **YOLO11_MODELS,
        **YOLO12_MODELS,
    }

    def __init__(self, model_name=None):
        """Initialize YoloDetector with a model."""
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            model_name = self.select_model()
        else:
            # Look up the actual filename from the model name key
            if model_name in self.AVAILABLE_MODELS:
                model_name = self.AVAILABLE_MODELS[model_name]

        model_path = self.models_dir / model_name
        self.model = YOLO(str(model_path))
        self.results = None

    def select_model(self):
        """Prompt user to select a model."""
        print("Available YOLO models:")
        for name in self.AVAILABLE_MODELS.keys():
            print(f"  {name}")

        while True:
            choice = input("\nSelect a model: ").strip()
            if choice in self.AVAILABLE_MODELS:
                return self.AVAILABLE_MODELS[choice]
            else:
                print(f"Invalid model. Available models: {', '.join(self.AVAILABLE_MODELS.keys())}")

    def detect(self, source):
        """Run detection on an image or video source and return extracted detections."""
        self.results = self.model(source)
        return self._extract_detections()

    def _check_results(self):
        """Check if detection results are available."""
        if self.results is None:
            raise ValueError("No detection results. Run detect() first.")

    def _to_scalar(self, value):
        """Convert tensor to scalar value."""
        return value.item() if hasattr(value, 'item') else value

    def _extract_detections(self):
        """Extract detection data from results."""
        self._check_results()

        detections = []
        for result in self.results:
            for i in range(len(result.boxes)):
                detection = {
                    "xyxy": result.boxes.xyxy[i],
                    "class_name": result.names[int(result.boxes.cls[i].item())],
                    "confidence": result.boxes.conf[i].item(),
                }
                detections.append(detection)
        return detections

    def select_output_format(self):
        """Prompt user to select output format."""
        print("\nOutput format:")
        print("1. Table")
        print("2. JSON file")

        while True:
            try:
                choice = int(input("Select output format (1-2): "))
                if choice == 1:
                    return self.FORMAT_TABLE
                elif choice == 2:
                    return self.FORMAT_JSON
                else:
                    print("Please enter 1 or 2")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def _prepare_detections(self):
        """Prepare detection data with area percentage calculation."""
        detections = self._extract_detections()
        if not detections:
            return []

        # Get image dimensions from the first result
        img_height = self.results[0].orig_shape[0]
        img_width = self.results[0].orig_shape[1]
        total_area = img_height * img_width

        # Calculate area percentage for each detection
        for det in detections:
            x1, y1, x2, y2 = det['xyxy']
            box_area = (x2 - x1) * (y2 - y1)
            det['area_percent'] = (box_area / total_area) * 100

        # Sort by area percentage (descending)
        return sorted(detections, key=lambda d: d['area_percent'], reverse=True)

    def print_detections(self):
        """Print detection results in table format sorted by box area % (descending)."""
        self._check_results()

        sorted_detections = self._prepare_detections()
        if not sorted_detections:
            print("No objects detected.")
            return

        table_data = []
        for i, det in enumerate(sorted_detections, 1):
            x1, y1, x2, y2 = det['xyxy']
            box = f"{{{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}}}"
            table_data.append([
                i,
                det['class_name'],
                box,
                f"{det['confidence']:.2%}",
                f"{det['area_percent']:.2f}%",
            ])

        headers = ["ID", "Class", "Box (xmin, ymin, xmax, ymax)", "Confidence", "Area %"]
        print(f"\nDetected {len(sorted_detections)} object(s):\n")
        print(tabulate(table_data, headers=headers, tablefmt="grid"))

    def save_to_json(self, output_file="detections.json"):
        """Save detection results to JSON file."""
        self._check_results()

        sorted_detections = self._prepare_detections()
        if not sorted_detections:
            print("No objects detected.")
            return

        json_data = []
        for i, det in enumerate(sorted_detections, 1):
            x1, y1, x2, y2 = det['xyxy']
            json_data.append({
                "id": i,
                "class": det['class_name'],
                "box": {
                    "xmin": int(self._to_scalar(x1)),
                    "ymin": int(self._to_scalar(y1)),
                    "xmax": int(self._to_scalar(x2)),
                    "ymax": int(self._to_scalar(y2))
                },
                "confidence": float(det['confidence']),
                "area_percent": round(float(det['area_percent']), 2)
            })

        output_path = Path(__file__).parent / output_file
        with open(output_path, 'w') as f:
            json.dump({"detections": json_data}, f, indent=2)

        print(f"\nResults saved to {output_file}")

    def output_results(self, format_type=None):
        """Output results in selected format."""
        if format_type is None:
            format_type = self.select_output_format()

        if format_type == self.FORMAT_TABLE:
            self.print_detections()
        elif format_type == self.FORMAT_JSON:
            self.save_to_json()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Path to image or video file, or URL"
    )
    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model name (e.g., v12-nano, v11-large). If not provided, will prompt for selection."
    )
    parser.add_argument(
        "-f", "--format",
        choices=["table", "json"],
        default="table",
        help="Output format: table or json (default: table)"
    )
    parser.add_argument(
        "-o", "--output",
        default="detections.json",
        help="Output file path for JSON results (default: detections.json)"
    )

    args = parser.parse_args()

    detector = YoloDetector(model_name=args.model)
    detector.detect(args.input)
    detector.output_results(format_type=args.format)

    if args.format == "json":
        detector.save_to_json(output_file=args.output)
