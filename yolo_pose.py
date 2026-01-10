from ultralytics import YOLO
from typing import Union, List, Optional
import numpy as np
import argparse
import os


class YoloPose:
    """YOLO Pose estimation wrapper for easy model loading and inference."""

    # YOLO11 Pose Models
    YOLO11_MODELS = {
        "yolo11n": "yolo11n-pose.pt",
        "yolo11s": "yolo11s-pose.pt",
        "yolo11m": "yolo11m-pose.pt",
        "yolo11l": "yolo11l-pose.pt",
        "yolo11x": "yolo11x-pose.pt",
    }

    # YOLOv8 Pose Models
    YOLOV8_MODELS = {
        "yolov8n": "yolov8n-pose.pt",
        "yolov8s": "yolov8s-pose.pt",
        "yolov8m": "yolov8m-pose.pt",
        "yolov8l": "yolov8l-pose.pt",
        "yolov8x": "yolov8x-pose.pt",
        "yolov8x-p6": "yolov8x-pose-p6.pt",
    }

    # Combined all available models
    ALL_MODELS = {**YOLO11_MODELS, **YOLOV8_MODELS}

    def __init__(self, model: str = "yolo11n", device: Optional[Union[int, str]] = None):
        """
        Initialize YoloPose with a specified model.

        Args:
            model: Model name (e.g., 'yolo11n', 'yolov8m'). Defaults to 'yolo11n'.
            device: Device to run inference on (e.g., 0 for GPU, 'cpu'). Defaults to auto-detection.
        """
        self.model_name = model
        self.model_path = self._get_model_path(model)
        self.model = YOLO(self.model_path)

        if device is not None:
            self.model.to(device)

    def _get_model_path(self, model: str) -> str:
        """Get the model file path from model name."""
        if model in self.ALL_MODELS:
            return os.path.join("models", self.ALL_MODELS[model])
        else:
            # Assume it's a direct path or model identifier
            return model

    def predict(self, source: Union[str, np.ndarray], conf: float = 0.5, **kwargs):
        """
        Run pose estimation on an image.

        Args:
            source: Image path, URL, or numpy array.
            conf: Confidence threshold for detections.
            **kwargs: Additional arguments to pass to YOLO.predict().

        Returns:
            Results object with pose keypoints.
        """
        return self.model.predict(source, conf=conf, **kwargs)

    def get_keypoints(self, results) -> List[dict]:
        """
        Extract keypoints from results.

        Args:
            results: Results from predict() method.

        Returns:
            List of dictionaries with keypoint data for each detected person.
        """
        keypoints_list = []

        for result in results:
            if result.keypoints is None:
                continue

            xy = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, 'cpu') else result.keypoints.xy
            conf = result.keypoints.conf.cpu().numpy() if hasattr(result.keypoints.conf, 'cpu') else result.keypoints.conf

            for person_idx in range(len(xy)):
                keypoints_dict = {
                    "xy": xy[person_idx],  # (num_keypoints, 2)
                    "confidence": conf[person_idx] if conf is not None else None,  # (num_keypoints,)
                }
                keypoints_list.append(keypoints_dict)

        return keypoints_list

    def get_all_data(self, results) -> List[dict]:
        """
        Extract all detection data including bounding boxes and keypoints.

        Args:
            results: Results from predict() method.

        Returns:
            List of dictionaries with detection and keypoint data.
        """
        detections = []

        for result in results:
            for idx in range(len(result.boxes)):
                detection = {
                    "bbox": result.boxes.xyxy[idx].cpu().numpy() if hasattr(result.boxes.xyxy[idx], 'cpu') else result.boxes.xyxy[idx],
                    "confidence": result.boxes.conf[idx].item() if hasattr(result.boxes.conf[idx], 'item') else float(result.boxes.conf[idx]),
                    "class_id": int(result.boxes.cls[idx]),
                }

                # Add keypoints if available
                if result.keypoints is not None and idx < len(result.keypoints.xy):
                    xy = result.keypoints.xy[idx].cpu().numpy() if hasattr(result.keypoints.xy[idx], 'cpu') else result.keypoints.xy[idx]
                    conf = result.keypoints.conf[idx].cpu().numpy() if hasattr(result.keypoints.conf[idx], 'cpu') else result.keypoints.conf[idx]
                    detection["keypoints"] = {
                        "xy": xy,
                        "confidence": conf,
                    }

                detections.append(detection)

        return detections


def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="YOLO Pose Estimation - Detect human poses in images"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n",
        help="Model to use for pose estimation. Available models: " +
             ", ".join(YoloPose.ALL_MODELS.keys()) +
             " (default: yolo11n)"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="https://ultralytics.com/images/bus.jpg",
        help="Image source (path, URL, or directory). (default: bus.jpg from ultralytics)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detections (0.0-1.0). (default: 0.5)"
    )

    parser.add_argument(
        "--show-all",
        action="store_true",
        help="Show all detection data (bboxes + keypoints) instead of just keypoints"
    )

    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all available models and exit"
    )

    args = parser.parse_args()

    # List available models if requested
    if args.list_models:
        print("Available YOLO11 Models:")
        for model_name in YoloPose.YOLO11_MODELS.keys():
            print(f"  - {model_name}")
        print("\nAvailable YOLOv8 Models:")
        for model_name in YoloPose.YOLOV8_MODELS.keys():
            print(f"  - {model_name}")
        return

    print(f"Loading model: {args.model}")
    pose_estimator = YoloPose(model=args.model)

    print(f"Running inference on: {args.source}")
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
    main()
