
import os
import json
from pathlib import Path
from ultralytics import YOLO
from tabulate import tabulate


class YoloDetector:
    # Output format constants
    FORMAT_TABLE = "table"
    FORMAT_JSON = "json"

    AVAILABLE_MODELS = {
        "YOLOv10n": "yolov10n.pt",
        "YOLOv10s": "yolov10s.pt",
        "YOLOv10m": "yolov10m.pt",
        "YOLOv10b": "yolov10b.pt",
        "YOLOv10l": "yolov10l.pt",
        "YOLOv10x": "yolov10x.pt",
        "YOLOv9t": "yolov9t.pt",
        "YOLOv9s": "yolov9s.pt",
        "YOLOv9m": "yolov9m.pt",
        "YOLOv9c": "yolov9c.pt",
        "YOLOv9e": "yolov9e.pt",
        "YOLOv8n": "yolov8n.pt",
        "YOLOv8s": "yolov8s.pt",
        "YOLOv8m": "yolov8m.pt",
        "YOLOv8l": "yolov8l.pt",
        "YOLOv8x": "yolov8x.pt",
        "YOLOv8s-world": "yolov8s-world.pt",
        "YOLOv8s-worldv2": "yolov8s-worldv2.pt",
        "YOLOv8m-world": "yolov8m-world.pt",
        "YOLOv8m-worldv2": "yolov8m-worldv2.pt",
        "YOLOv8l-world": "yolov8l-world.pt",
        "YOLOv8l-worldv2": "yolov8l-worldv2.pt",
        "YOLOv8x-world": "yolov8x-world.pt",
        "YOLOv8x-worldv2": "yolov8x-worldv2.pt",
        "RT-DETR-l": "rtdetr-l.pt",
        "RT-DETR-x": "rtdetr-x.pt",
        "YOLO11n": "yolo11n.pt",
        "YOLO11s": "yolo11s.pt",
        "YOLO11m": "yolo11m.pt",
        "YOLO11l": "yolo11l.pt",
        "YOLO11x": "yolo11x.pt",
        "YOLO12n": "yolo12n.pt",
        "YOLO12s": "yolo12s.pt",
        "YOLO12m": "yolo12m.pt",
        "YOLO12l": "yolo12l.pt",
        "YOLO12x": "yolo12x.pt",
    }

    def __init__(self, model_name=None):
        """Initialize YoloDetector with a model."""
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            model_name = self.select_model()

        model_path = self.models_dir / model_name
        self.model = YOLO(str(model_path))
        self.results = None

    def select_model(self):
        """Prompt user to select a model."""
        models_list = list(self.AVAILABLE_MODELS.items())
        print("Available YOLO models:")
        for i, (display_name, _) in enumerate(models_list, 1):
            print(f"{i}. {display_name}")

        while True:
            try:
                choice = int(input(f"Select a model (1-{len(models_list)}): "))
                if 1 <= choice <= len(models_list):
                    return models_list[choice - 1][1]
                else:
                    print(f"Please enter a number between 1 and {len(models_list)}")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def detect(self, source):
        """Run detection on an image or video source."""
        self.results = self.model(source)
        return self.results

    def _check_results(self):
        """Check if detection results are available."""
        if self.results is None:
            raise ValueError("No detection results. Run detect() first.")

    def _to_scalar(self, value):
        """Convert tensor to scalar value."""
        return value.item() if hasattr(value, 'item') else value

    def get_detections(self):
        """Extract and return detection data from results."""
        self._check_results()

        detections = []
        for result in self.results:
            for i in range(len(result.boxes)):
                detection = {
                    "xywh": result.boxes.xywh[i],
                    "xywhn": result.boxes.xywhn[i],
                    "xyxy": result.boxes.xyxy[i],
                    "xyxyn": result.boxes.xyxyn[i],
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
        detections = self.get_detections()
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
    detector = YoloDetector()
    input_source = input("\nEnter image path or URL: ")
    detector.detect(input_source)
    detector.output_results()
