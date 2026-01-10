from yolo_detect import YoloDetector

# Test with bicycle.jpg using YOLOv10n model
detector = YoloDetector(model_name="yolov10n.pt")
detections = detector.detect("bicycle.jpg")

print(f"\nDetected {len(detections)} object(s)")
for i, det in enumerate(detections, 1):
    print(f"{i}. {det['class_name']} - Confidence: {det['confidence']:.2%}")

# Test output formats
detector.output_results("table")
