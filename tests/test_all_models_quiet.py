from yolo_detect import YoloDetector
import time
import sys

models = list(YoloDetector.AVAILABLE_MODELS.items())

print(f"Testing {len(models)} models on bicycle.jpg\n")
print(f"{'Model':<20} {'Detections':<12} {'Time (ms)':<12}")
print("-" * 45)

for idx, (model_name, model_file) in enumerate(models, 1):
    try:
        start = time.time()
        detector = YoloDetector(model_name=model_file)
        detections = detector.detect("bicycle.jpg")
        elapsed = (time.time() - start) * 1000
        
        print(f"{model_name:<20} {len(detections):<12} {elapsed:<12.1f}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"{model_name:<20} {'ERROR':<12} {str(e)[:30]}", flush=True)
        sys.stdout.flush()

print("\nTest completed!")
