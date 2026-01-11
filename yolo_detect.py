
import os
import json
import argparse
from pathlib import Path
from ultralytics import YOLO
import supervision as sv
import cv2


class YoloDetector:
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

    # YOLO11 OBB Models (Oriented Bounding Box)
    YOLO11_OBB_MODELS = {
        "v11n-obb": "yolo11n-obb.pt",
        "v11s-obb": "yolo11s-obb.pt",
        "v11m-obb": "yolo11m-obb.pt",
        "v11l-obb": "yolo11l-obb.pt",
        "v11x-obb": "yolo11x-obb.pt",
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
        **YOLO11_OBB_MODELS,
        **YOLO12_MODELS,
    }

    def __init__(self, model_name=None):
        """Initialize YoloDetector with a model."""
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        if model_name is None:
            model_name = "v8-large"

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
        self.source = source
        self.results = self.model(source)
        return self._extract_detections()

    def _check_results(self):
        """Check if detection results are available."""
        if self.results is None:
            raise ValueError("No detection results. Run detect() first.")

    def _to_scalar(self, value):
        """Convert tensor to scalar value."""
        return value.item() if hasattr(value, 'item') else value

    def _polygon_area(self, points):
        """Calculate polygon area using the Shoelace formula."""
        points = [self._to_scalar(p) for p in points]
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _extract_detections(self):
        """Extract detection data from results."""
        self._check_results()

        detections = []
        for result in self.results:
            # Check if this is an OBB result
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                # OBB (Oriented Bounding Box) detection
                for i in range(len(result.obb)):
                    detection = {
                        "type": "obb",
                        "xyxyxyxy": result.obb.xyxyxyxy[i],  # 4 polygon points
                        "class_name": result.names[int(result.obb.cls[i].item())],
                        "confidence": result.obb.conf[i].item(),
                    }
                    detections.append(detection)
            else:
                # Regular box detection
                for i in range(len(result.boxes)):
                    detection = {
                        "type": "box",
                        "xyxy": result.boxes.xyxy[i],
                        "class_name": result.names[int(result.boxes.cls[i].item())],
                        "confidence": result.boxes.conf[i].item(),
                    }
                    detections.append(detection)
        return detections

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
            if det['type'] == 'obb':
                # For OBB, calculate polygon area using the 4 vertices
                box_area = self._polygon_area(det['xyxyxyxy'])
            else:
                # For regular boxes, use xyxy coordinates
                x1, y1, x2, y2 = det['xyxy']
                box_area = (self._to_scalar(x2) - self._to_scalar(x1)) * (self._to_scalar(y2) - self._to_scalar(y1))

            det['area_percent'] = (box_area / total_area) * 100

        # Sort by area percentage (descending)
        return sorted(detections, key=lambda d: d['area_percent'], reverse=True)

    def save_to_json(self, output_file=None):
        """Save detection results to JSON file."""
        self._check_results()

        sorted_detections = self._prepare_detections()
        if not sorted_detections:
            print("No objects detected.")
            return

        # Generate output filename from source if not provided
        if output_file is None:
            input_filename = Path(self.source).stem
            output_file = f"{input_filename}_detection.json"

        json_data = []
        for i, det in enumerate(sorted_detections, 1):
            detection_obj = {
                "id": i,
                "class": det['class_name'],
                "confidence": round(float(det['confidence']), 3),
                "area_percent": round(float(det['area_percent']), 2)
            }

            if det['type'] == 'obb':
                # For OBB detections, include polygon points
                detection_obj["polygon_points"] = [
                    {"x": int(self._to_scalar(p[0])), "y": int(self._to_scalar(p[1]))}
                    for p in det['xyxyxyxy']
                ]
            else:
                # For regular box detections
                x1, y1, x2, y2 = det['xyxy']
                detection_obj["box"] = {
                    "xmin": int(self._to_scalar(x1)),
                    "ymin": int(self._to_scalar(y1)),
                    "xmax": int(self._to_scalar(x2)),
                    "ymax": int(self._to_scalar(y2))
                }

            json_data.append(detection_obj)

        output_dir = Path(__file__).parent / "results" / "det"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / output_file
        with open(output_path, 'w') as f:
            json.dump({"detections": json_data}, f, indent=2)

        print(f"\nResults saved to results/det/{output_file}")

    def save_annotated_image(self, source, output_file=None):
        """Save annotated image with bounding boxes using supervision."""
        self._check_results()

        if not self.results:
            print("No detection results to annotate.")
            return

        # Load the image
        image = cv2.imread(source)
        if image is None:
            print(f"Could not read image from {source}")
            return

        # Generate output filename if not provided
        if output_file is None:
            input_filename = Path(source).stem
            output_file = f"{input_filename}_annotated.png"

        # Convert YOLO results to supervision Detections
        detections_list = []
        class_names = []

        for result in self.results:
            if hasattr(result, 'obb') and result.obb is not None and len(result.obb) > 0:
                # OBB detections - convert to regular boxes for visualization
                for i in range(len(result.obb)):
                    xyxyxyxy = result.obb.xyxyxyxy[i]
                    # Get bounding box from polygon points
                    xs = [self._to_scalar(p[0]) for p in xyxyxyxy]
                    ys = [self._to_scalar(p[1]) for p in xyxyxyxy]
                    x1, x2 = min(xs), max(xs)
                    y1, y2 = min(ys), max(ys)

                    detections_list.append({
                        'xyxy': [x1, y1, x2, y2],
                        'confidence': result.obb.conf[i].item(),
                        'class_id': int(result.obb.cls[i].item())
                    })
                    class_names.append(result.names[int(result.obb.cls[i].item())])
            else:
                # Regular box detections
                for i in range(len(result.boxes)):
                    xyxy = result.boxes.xyxy[i].tolist()
                    detections_list.append({
                        'xyxy': xyxy,
                        'confidence': result.boxes.conf[i].item(),
                        'class_id': int(result.boxes.cls[i].item())
                    })
                    class_names.append(result.names[int(result.boxes.cls[i].item())])

        # Create supervision Detections object
        if detections_list:
            xyxy = [d['xyxy'] for d in detections_list]
            confidence = [d['confidence'] for d in detections_list]
            class_ids = [d['class_id'] for d in detections_list]

            import numpy as np
            detections = sv.Detections(
                xyxy=np.array(xyxy),
                confidence=np.array(confidence),
                class_id=np.array(class_ids)
            )

            # Create annotator and draw detections
            box_annotator = sv.BoxAnnotator()
            label_annotator = sv.LabelAnnotator()

            annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections)
            labels = [f"{class_names[i]} {confidence[i]:.2f}" for i in range(len(class_names))]
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            # Save annotated image
            output_dir = Path(__file__).parent / "results" / "det"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / output_file
            cv2.imwrite(str(output_path), annotated_image)
            print(f"Annotated image saved to results/det/{output_file}")
        else:
            print("No detections to annotate.")


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

    args = parser.parse_args()

    detector = YoloDetector(model_name=args.model)
    detector.detect(args.input)
    detector.save_to_json()
    detector.save_annotated_image(source=args.input)
