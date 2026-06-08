import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import supervision as sv
from tabulate import tabulate
from ultralytics import YOLO

from base import BaseYoloModel
from config import DET_RESULTS_DIR
from exceptions import NoResultsError

logger = logging.getLogger(__name__)


class YoloDetector(BaseYoloModel):
    model_class = YOLO

    YOLOV10_MODELS: Dict[str, str] = {
        "v10-nano": "yolov10n.pt",
        "v10-small": "yolov10s.pt",
        "v10-medium": "yolov10m.pt",
        "v10-base": "yolov10b.pt",
        "v10-large": "yolov10l.pt",
        "v10-xlarge": "yolov10x.pt",
    }

    YOLOV9_MODELS: Dict[str, str] = {
        "v9-tiny": "yolov9t.pt",
        "v9-small": "yolov9s.pt",
        "v9-medium": "yolov9m.pt",
        "v9-compact": "yolov9c.pt",
        "v9-extra": "yolov9e.pt",
    }

    YOLOV8_MODELS: Dict[str, str] = {
        "v8-nano": "yolov8n.pt",
        "v8-small": "yolov8s.pt",
        "v8-medium": "yolov8m.pt",
        "v8-large": "yolov8l.pt",
        "v8-xlarge": "yolov8x.pt",
    }

    YOLOV8_WORLD_MODELS: Dict[str, str] = {
        "v8s-world": "yolov8s-world.pt",
        "v8s-worldv2": "yolov8s-worldv2.pt",
        "v8m-world": "yolov8m-world.pt",
        "v8m-worldv2": "yolov8m-worldv2.pt",
        "v8l-world": "yolov8l-world.pt",
        "v8l-worldv2": "yolov8l-worldv2.pt",
        "v8x-world": "yolov8x-world.pt",
        "v8x-worldv2": "yolov8x-worldv2.pt",
    }

    RTDETR_MODELS: Dict[str, str] = {
        "rtdetr-large": "rtdetr-l.pt",
        "rtdetr-xlarge": "rtdetr-x.pt",
    }

    YOLO11_MODELS: Dict[str, str] = {
        "v11-nano": "yolo11n.pt",
        "v11-small": "yolo11s.pt",
        "v11-medium": "yolo11m.pt",
        "v11-large": "yolo11l.pt",
        "v11-xlarge": "yolo11x.pt",
    }

    YOLO11_OBB_MODELS: Dict[str, str] = {
        "v11n-obb": "yolo11n-obb.pt",
        "v11s-obb": "yolo11s-obb.pt",
        "v11m-obb": "yolo11m-obb.pt",
        "v11l-obb": "yolo11l-obb.pt",
        "v11x-obb": "yolo11x-obb.pt",
    }

    YOLO12_MODELS: Dict[str, str] = {
        "v12-nano": "yolo12n.pt",
        "v12-small": "yolo12s.pt",
        "v12-medium": "yolo12m.pt",
        "v12-large": "yolo12l.pt",
        "v12-xlarge": "yolo12x.pt",
    }

    YOLO26_MODELS: Dict[str, str] = {
        "v26-nano": "yolo26n.pt",
        "v26-small": "yolo26s.pt",
        "v26-medium": "yolo26m.pt",
        "v26-large": "yolo26l.pt",
        "v26-xlarge": "yolo26x.pt",
    }

    YOLO26_OBB_MODELS: Dict[str, str] = {
        "v26n-obb": "yolo26n-obb.pt",
        "v26s-obb": "yolo26s-obb.pt",
        "v26m-obb": "yolo26m-obb.pt",
        "v26l-obb": "yolo26l-obb.pt",
        "v26x-obb": "yolo26x-obb.pt",
    }

    available_models: Dict[str, str] = {
        **YOLOV10_MODELS,
        **YOLOV9_MODELS,
        **YOLOV8_MODELS,
        **YOLOV8_WORLD_MODELS,
        **RTDETR_MODELS,
        **YOLO11_MODELS,
        **YOLO11_OBB_MODELS,
        **YOLO12_MODELS,
        **YOLO26_MODELS,
        **YOLO26_OBB_MODELS,
    }

    FORMAT_TABLE = "table"
    FORMAT_JSON = "json"

    def __init__(self, model_name: Optional[str] = None):
        super().__init__(model_name, default_model="v8-large")
        self._source: Optional[str] = None

    def detect(self, source: str) -> List[Dict[str, Any]]:
        """Run detection on an image or video source.

        Returns:
            List of detection dictionaries with keys ``type``, ``xyxy``
            (or ``xyxyxyxy`` for OBB), ``class_name``, and ``confidence``.
        """
        self._source = source
        self.results = self.model(source)
        return self._extract_detections()

    def get_detections(self) -> List[Dict[str, Any]]:
        """Return prepared detection data with area percentages."""
        return self._prepare_detections()

    def print_detections(self) -> None:
        """Display results as a formatted table via ``tabulate``."""
        detections = self._prepare_detections()
        if not detections:
            logger.info("No objects detected.")
            return

        table_data: List[List[Union[int, str]]] = []
        for i, det in enumerate(detections, 1):
            if det["type"] == "obb":
                box_str = f"polygon({len(det['xyxyxyxy'])} pts)"
            else:
                x1, y1, x2, y2 = det["xyxy"]
                box_str = (
                    f"({self._tensor_to_scalar(x1):.0f}, "
                    f"{self._tensor_to_scalar(y1):.0f}, "
                    f"{self._tensor_to_scalar(x2):.0f}, "
                    f"{self._tensor_to_scalar(y2):.0f})"
                )
            table_data.append([
                i,
                det["class_name"],
                box_str,
                f"{det['confidence']:.2%}",
                f"{det['area_percent']:.2f}%",
            ])

        headers = ["ID", "Class", "Box (xmin, ymin, xmax, ymax)", "Confidence", "Area %"]
        print("\n" + tabulate(table_data, headers=headers, tablefmt="grid"))

    def select_output_format(self) -> str:
        """Interactive format selection menu (blocking)."""
        print("\nOutput format:")
        print(f"  1. {self.FORMAT_TABLE}")
        print(f"  2. {self.FORMAT_JSON}")
        while True:
            choice = input("\nSelect format: ").strip()
            if choice == "1":
                return self.FORMAT_TABLE
            if choice == "2":
                return self.FORMAT_JSON
            print("Invalid choice. Enter 1 or 2.")

    def output_results(self, format_type: Optional[str] = None) -> None:
        """Output results in the specified format."""
        if format_type is None:
            format_type = self.select_output_format()
        if format_type == self.FORMAT_TABLE:
            self.print_detections()
        elif format_type == self.FORMAT_JSON:
            self.save_to_json()
        else:
            logger.warning("Unknown format: %s", format_type)

    @staticmethod
    def _tensor_to_scalar(value: Any) -> Union[float, int, Any]:
        """Convert a tensor/array element to a Python scalar."""
        return value.item() if hasattr(value, "item") else value

    @staticmethod
    def _polygon_area(points: List[Tuple[float, float]]) -> float:
        """Calculate polygon area using the Shoelace formula.

        Args:
            points: List of ``(x, y)`` tuples.
        """
        n = len(points)
        area = 0.0
        for i in range(n):
            x1, y1 = points[i]
            x2, y2 = points[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) / 2.0

    def _extract_detections(self) -> List[Dict[str, Any]]:
        """Extract raw detection data from the latest inference results."""
        self._check_results()

        detections: List[Dict[str, Any]] = []
        for result in self.results:
            if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
                for i in range(len(result.obb)):
                    detections.append({
                        "type": "obb",
                        "xyxyxyxy": result.obb.xyxyxyxy[i],
                        "class_name": result.names[int(result.obb.cls[i].item())],
                        "confidence": result.obb.conf[i].item(),
                    })
            else:
                for i in range(len(result.boxes)):
                    detections.append({
                        "type": "box",
                        "xyxy": result.boxes.xyxy[i],
                        "class_name": result.names[int(result.boxes.cls[i].item())],
                        "confidence": result.boxes.conf[i].item(),
                    })
        return detections

    def _prepare_detections(self) -> List[Dict[str, Any]]:
        """Augment detections with area-percentage and sort descending."""
        detections = self._extract_detections()
        if not detections:
            return []

        img_height = self.results[0].orig_shape[0]
        img_width = self.results[0].orig_shape[1]
        total_area = img_height * img_width

        for det in detections:
            if det["type"] == "obb":
                pts = [(self._tensor_to_scalar(p[0]), self._tensor_to_scalar(p[1]))
                       for p in det["xyxyxyxy"]]
                box_area = self._polygon_area(pts)
            else:
                x1, y1, x2, y2 = det["xyxy"]
                box_area = (
                    (self._tensor_to_scalar(x2) - self._tensor_to_scalar(x1))
                    * (self._tensor_to_scalar(y2) - self._tensor_to_scalar(y1))
                )
            det["area_percent"] = (box_area / total_area) * 100

        return sorted(detections, key=lambda d: d["area_percent"], reverse=True)

    def save_to_json(self, output_file: Optional[str] = None) -> None:
        """Save detection results to a JSON file.

        Args:
            output_file: Filename (without path).  If omitted, derives from
                the source image name.
        """
        self._check_results()

        sorted_detections = self._prepare_detections()
        if not sorted_detections:
            logger.info("No objects detected.")
            return

        if output_file is None:
            if self._source is None:
                raise NoResultsError("No source image set; cannot derive output filename.")
            input_filename = Path(self._source).stem
            output_file = f"{input_filename}_detection.json"

        json_data: List[Dict[str, Any]] = []
        for i, det in enumerate(sorted_detections, 1):
            entry: Dict[str, Any] = {
                "id": i,
                "class": det["class_name"],
                "confidence": round(float(det["confidence"]), 3),
                "area_percent": round(float(det["area_percent"]), 2),
            }

            if det["type"] == "obb":
                entry["polygon_points"] = [
                    {"x": int(self._tensor_to_scalar(p[0])),
                     "y": int(self._tensor_to_scalar(p[1]))}
                    for p in det["xyxyxyxy"]
                ]
            else:
                x1, y1, x2, y2 = det["xyxy"]
                entry["box"] = {
                    "xmin": int(self._tensor_to_scalar(x1)),
                    "ymin": int(self._tensor_to_scalar(y1)),
                    "xmax": int(self._tensor_to_scalar(x2)),
                    "ymax": int(self._tensor_to_scalar(y2)),
                }

            json_data.append(entry)

        DET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DET_RESULTS_DIR / output_file
        with open(output_path, "w") as f:
            json.dump({"detections": json_data}, f, indent=2)
        logger.info("Results saved to %s", output_path)

    def save_annotated_image(self, source: str, output_file: Optional[str] = None) -> None:
        """Save an annotated image with bounding boxes via supervision.

        Args:
            source: Path to the original image.
            output_file: Output filename.  Derived from *source* if omitted.
        """
        self._check_results()

        image = cv2.imread(source)
        if image is None:
            raise FileNotFoundError(f"Could not read image from {source}")

        if output_file is None:
            input_filename = Path(source).stem
            output_file = f"{input_filename}_annotated.png"

        detections_list: List[Dict[str, Any]] = []
        class_names: List[str] = []

        for result in self.results:
            if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
                for i in range(len(result.obb)):
                    xyxyxyxy = result.obb.xyxyxyxy[i]
                    xs = [self._tensor_to_scalar(p[0]) for p in xyxyxyxy]
                    ys = [self._tensor_to_scalar(p[1]) for p in xyxyxyxy]
                    detections_list.append({
                        "xyxy": [min(xs), min(ys), max(xs), max(ys)],
                        "confidence": result.obb.conf[i].item(),
                        "class_id": int(result.obb.cls[i].item()),
                    })
                    class_names.append(result.names[int(result.obb.cls[i].item())])
            else:
                for i in range(len(result.boxes)):
                    xyxy = result.boxes.xyxy[i].tolist()
                    detections_list.append({
                        "xyxy": xyxy,
                        "confidence": result.boxes.conf[i].item(),
                        "class_id": int(result.boxes.cls[i].item()),
                    })
                    class_names.append(result.names[int(result.boxes.cls[i].item())])

        if not detections_list:
            logger.info("No detections to annotate.")
            return

        xyxy = [d["xyxy"] for d in detections_list]
        confidence = [d["confidence"] for d in detections_list]
        class_ids = [d["class_id"] for d in detections_list]

        detections_sv = sv.Detections(
            xyxy=np.array(xyxy),
            confidence=np.array(confidence),
            class_id=np.array(class_ids),
        )

        box_annotator = sv.BoxAnnotator()
        label_annotator = sv.LabelAnnotator()

        annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections_sv)
        labels = [f"{class_names[i]} {confidence[i]:.2f}" for i in range(len(class_names))]
        annotated_image = label_annotator.annotate(
            scene=annotated_image, detections=detections_sv, labels=labels
        )

        DET_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DET_RESULTS_DIR / output_file
        cv2.imwrite(str(output_path), annotated_image)
        logger.info("Annotated image saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import argparse

    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("-i", "--input", required=True, help="Path to image or video file, or URL")
    parser.add_argument(
        "-m", "--model", default=None,
        help="Model name (e.g., v12-nano, v11-large). If not provided, will prompt for selection.",
    )
    args = parser.parse_args()

    detector = YoloDetector(model_name=args.model)
    detector.detect(args.input)
    detector.save_to_json()
    detector.save_annotated_image(source=args.input)

