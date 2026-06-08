import argparse
import logging
from typing import Dict, List, Optional, Union

import numpy as np
from ultralytics import YOLO

from base import BaseYoloModel

logger = logging.getLogger(__name__)


class YoloPose(BaseYoloModel):
    model_class = YOLO

    YOLO11_MODELS: Dict[str, str] = {
        "yolo11n": "yolo11n-pose.pt",
        "yolo11s": "yolo11s-pose.pt",
        "yolo11m": "yolo11m-pose.pt",
        "yolo11l": "yolo11l-pose.pt",
        "yolo11x": "yolo11x-pose.pt",
    }

    YOLOV8_MODELS: Dict[str, str] = {
        "yolov8n": "yolov8n-pose.pt",
        "yolov8s": "yolov8s-pose.pt",
        "yolov8m": "yolov8m-pose.pt",
        "yolov8l": "yolov8l-pose.pt",
        "yolov8x": "yolov8x-pose.pt",
        "yolov8x-p6": "yolov8x-pose-p6.pt",
    }

    YOLO26_MODELS: Dict[str, str] = {
        "yolo26n": "yolo26n-pose.pt",
        "yolo26s": "yolo26s-pose.pt",
        "yolo26m": "yolo26m-pose.pt",
        "yolo26l": "yolo26l-pose.pt",
        "yolo26x": "yolo26x-pose.pt",
    }

    _ALIASES: Dict[str, str] = {
        "v11-nano": "yolo11n-pose.pt", "v11-small": "yolo11s-pose.pt",
        "v11-medium": "yolo11m-pose.pt", "v11-large": "yolo11l-pose.pt", "v11-xlarge": "yolo11x-pose.pt",
        "v8-nano": "yolov8n-pose.pt", "v8-small": "yolov8s-pose.pt",
        "v8-medium": "yolov8m-pose.pt", "v8-large": "yolov8l-pose.pt", "v8-xlarge": "yolov8x-pose.pt",
        "v26-nano": "yolo26n-pose.pt", "v26-small": "yolo26s-pose.pt",
        "v26-medium": "yolo26m-pose.pt", "v26-large": "yolo26l-pose.pt", "v26-xlarge": "yolo26x-pose.pt",
    }

    available_models: Dict[str, str] = {
        **YOLO11_MODELS,
        **YOLOV8_MODELS,
        **YOLO26_MODELS,
        **_ALIASES,
    }

    def __init__(self, model: str = "yolo11n", device: Optional[Union[int, str]] = None):
        super().__init__(model, default_model="yolo11n")
        if device is not None:
            self.model.to(device)

    def predict(self, source: Union[str, np.ndarray], conf: float = 0.5, **kwargs):
        """Run pose estimation on an image.

        Args:
            source: Image path, URL, or numpy array.
            conf: Confidence threshold.
            **kwargs: Additional arguments passed to ``YOLO.predict()``.

        Returns:
            YOLO inference results.
        """
        return self.model.predict(source, conf=conf, **kwargs)

    @staticmethod
    def get_keypoints(results) -> List[Dict[str, object]]:
        """Extract keypoints from inference results.

        Args:
            results: Output of :meth:`predict`.

        Returns:
            List of dicts with keys ``xy`` and ``confidence`` per person.
        """
        keypoints_list: List[Dict[str, object]] = []

        for result in results:
            if result.keypoints is None:
                continue

            xy = result.keypoints.xy
            conf = result.keypoints.conf

            if hasattr(xy, "cpu"):
                xy = xy.cpu().numpy()
            if conf is not None and hasattr(conf, "cpu"):
                conf = conf.cpu().numpy()

            for person_idx in range(len(xy)):
                keypoints_list.append({
                    "xy": xy[person_idx],
                    "confidence": conf[person_idx] if conf is not None else None,
                })

        return keypoints_list

    @staticmethod
    def get_all_data(results) -> List[Dict]:
        """Extract bounding boxes, confidence, and keypoints from results.

        Args:
            results: Output of :meth:`predict`.

        Returns:
            List of detection dicts, each optionally containing ``keypoints``.
        """
        detections: List[Dict] = []

        for result in results:
            if result.boxes is None:
                continue

            for idx in range(len(result.boxes)):
                detection: Dict = {
                    "bbox": result.boxes.xyxy[idx].cpu().numpy()
                    if hasattr(result.boxes.xyxy[idx], "cpu")
                    else result.boxes.xyxy[idx],
                    "confidence": result.boxes.conf[idx].item()
                    if hasattr(result.boxes.conf[idx], "item")
                    else float(result.boxes.conf[idx]),
                    "class_id": int(result.boxes.cls[idx]),
                }

                if result.keypoints is not None and idx < len(result.keypoints.xy):
                    xy = result.keypoints.xy[idx]
                    conf = result.keypoints.conf[idx]
                    if hasattr(xy, "cpu"):
                        xy = xy.cpu().numpy()
                    if conf is not None and hasattr(conf, "cpu"):
                        conf = conf.cpu().numpy()
                    detection["keypoints"] = {"xy": xy, "confidence": conf}

                detections.append(detection)

        return detections


def main():
    parser = argparse.ArgumentParser(description="YOLO Pose Estimation")
    parser.add_argument("--version", action="store_true", help="Show version and exit")
    parser.add_argument(
        "--model", type=str, default="yolo11n",
        help="Model to use (default: yolo11n). Available: " + ", ".join(YoloPose.available_models.keys()),
    )
    parser.add_argument(
        "--source", type=str, default="https://ultralytics.com/images/bus.jpg",
        help="Image source (path or URL, default: bus.jpg from ultralytics)",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    parser.add_argument(
        "--show-all", action="store_true",
        help="Show all detection data (bboxes + keypoints) instead of just keypoints",
    )
    parser.add_argument("--list-models", action="store_true", help="List all available models and exit")

    args = parser.parse_args()

    if args.version:
        from version import __version__
        print(f"YoloLLM {__version__}")
        return

    if args.list_models:
        print("Available models:")
        for name in YoloPose.available_models:
            print(f"  - {name}")
        return

    logger.info("Loading model: %s", args.model)
    pose_estimator = YoloPose(model=args.model)

    logger.info("Running inference on: %s", args.source)
    results = pose_estimator.predict(args.source, conf=args.conf)

    if args.show_all:
        detections = pose_estimator.get_all_data(results)
        print(f"\nTotal detections: {len(detections)}")
        for i, detection in enumerate(detections):
            print(f"\nDetection {i}:")
            print(f"  Bounding Box: {detection['bbox']}")
            print(f"  Confidence: {detection['confidence']:.2f}")
            print(f"  Class ID: {detection['class_id']}")
            if "keypoints" in detection:
                print(f"  Keypoints: {len(detection['keypoints']['xy'])} points")
    else:
        keypoints = pose_estimator.get_keypoints(results)
        print(f"\nDetected {len(keypoints)} people")
        for i, kpt in enumerate(keypoints):
            print(f"Person {i}: {len(kpt['xy'])} keypoints")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

