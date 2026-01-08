# YoloLLM - YOLO Object Detection

A simple and modular Python tool for object detection using multiple YOLO model families. Detect objects in images or videos with support for 36 different pre-trained models.

## Features

- **36 Pre-trained Models** across 7 families:
  - YOLOv8 (standard + world variants)
  - YOLOv9
  - YOLOv10
  - YOLO11
  - YOLO12
  - RT-DETR

- **Interactive Model Selection** - Choose from available models via command-line menu

- **Multiple Output Formats**:
  - Table format (ASCII grid with detailed statistics)
  - JSON file export

- **Advanced Detection Metrics**:
  - Bounding box coordinates (multiple formats: xyxy, xywh, normalized variants)
  - Confidence scores
  - Object area as percentage of total image area
  - Results sorted by detection size (descending)

- **Clean, Modular Code** - Well-organized class with helper methods and constants

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to the project directory
cd YoloLLM

# Install dependencies
pip install ultralytics tabulate

# Create models directory (auto-created on first run)
mkdir models
```

## Usage

### Basic Usage

```bash
python yolo_detect.py
```

This will:
1. Prompt you to select a model (1-36)
2. Ask for an image path or URL
3. Run detection
4. Ask for output format (table or JSON)
5. Display or save results

### Python API

```python
from yolo_detect import YoloDetector

# Initialize detector with specific model
detector = YoloDetector(model_name="yolov8n.pt")

# Run detection
detector.detect("image.jpg")

# Get detection data
detections = detector.get_detections()

# Output results
detector.output_results(format_type="table")  # or "json"
```

### Programmatic Usage

```python
from yolo_detect import YoloDetector

# Create detector
detector = YoloDetector(model_name="yolov10n.pt")

# Detect objects
detector.detect("path/to/image.jpg")

# Access raw detections
detections = detector.get_detections()
for det in detections:
    print(f"{det['class_name']}: {det['confidence']:.2%} confidence")

# Save to JSON
detector.save_to_json("results.json")

# Print table
detector.print_detections()
```

## Available Models

### YOLOv8 (13 models)
- Standard: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- World: YOLOv8s-world, YOLOv8s-worldv2, YOLOv8m-world, YOLOv8m-worldv2, YOLOv8l-world, YOLOv8l-worldv2, YOLOv8x-world, YOLOv8x-worldv2

### YOLOv9 (5 models)
- YOLOv9t, YOLOv9s, YOLOv9m, YOLOv9c, YOLOv9e

### YOLOv10 (6 models)
- YOLOv10n, YOLOv10s, YOLOv10m, YOLOv10b, YOLOv10l, YOLOv10x

### YOLO11 (5 models)
- YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x

### YOLO12 (5 models)
- YOLO12n, YOLO12s, YOLO12m, YOLO12l, YOLO12x

### RT-DETR (2 models)
- RT-DETR-l, RT-DETR-x

## Output Formats

### Table Format
ASCII grid table with columns:
- ID: Detection index
- Class: Object class name
- Box: Bounding box coordinates (xmin, ymin, xmax, ymax)
- Confidence: Detection confidence percentage
- Area %: Percentage of image area occupied by object

```
+------+---------+--------------------------------+--------------+----------+
|   ID | Class   | Box (xmin, ymin, xmax, ymax)   | Confidence   | Area %   |
+======+=========+================================+==============+==========+
|    1 | bicycle | {360, 285, 799, 628}           | 96.19%       | 20.50%   |
+------+---------+--------------------------------+--------------+----------+
|    2 | person  | {464, 80, 706, 595}            | 88.23%       | 16.95%   |
+------+---------+--------------------------------+--------------+----------+
```

### JSON Format
Structured JSON file (`detections.json`):
```json
{
  "detections": [
    {
      "id": 1,
      "class": "bicycle",
      "box": {
        "xmin": 360,
        "ymin": 285,
        "xmax": 799,
        "ymax": 628
      },
      "confidence": 0.9619,
      "area_percent": 20.50
    }
  ]
}
```

## Project Structure

```
YoloLLM/
├── yolo_detect.py          # Main detection module
├── models/                 # Model storage (auto-created)
│   ├── yolov8n.pt
│   ├── yolo11x.pt
│   └── ... (34 more models)
├── detections.json         # Output file (generated)
├── README.md               # This file
└── .gitignore              # Git ignore file
```

## Class Reference

### YoloDetector

#### Constants
- `FORMAT_TABLE` - "table" output format
- `FORMAT_JSON` - "json" output format
- `AVAILABLE_MODELS` - Dictionary of available models

#### Methods

**Initialization**
- `__init__(model_name=None)` - Initialize detector with optional model name

**Core Detection**
- `detect(source)` - Run detection on image or video
- `get_detections()` - Extract detection data from results

**Model Selection**
- `select_model()` - Interactive model selection menu

**Output**
- `output_results(format_type=None)` - Output in selected format
- `print_detections()` - Display results as formatted table
- `save_to_json(output_file="detections.json")` - Save to JSON file
- `select_output_format()` - Interactive format selection menu

**Utilities**
- `_check_results()` - Validate detection results exist
- `_to_scalar(value)` - Convert tensor to scalar value
- `_prepare_detections()` - Prepare and sort detection data

## Examples

### Example 1: Simple Detection

```bash
python yolo_detect.py
# Select model: 1 (YOLOv10n)
# Enter image path: 1.jpg
# Select output format: 1 (Table)
```

### Example 2: Batch Processing

```python
from yolo_detect import YoloDetector
import os

detector = YoloDetector(model_name="yolov8s.pt")

for image in os.listdir("images/"):
    if image.endswith((".jpg", ".png")):
        print(f"Processing {image}...")
        detector.detect(f"images/{image}")
        detector.save_to_json(f"results/{image}.json")
```

### Example 3: Model Comparison

```python
from yolo_detect import YoloDetector

models = ["yolov8n.pt", "yolov10n.pt", "yolo11n.pt"]

for model in models:
    detector = YoloDetector(model_name=model)
    detector.detect("test.jpg")
    detections = detector.get_detections()
    print(f"{model}: {len(detections)} objects detected")
```

## Performance Notes

- Model sizes range from 4.7 MB (YOLOv9t) to 141 MB (YOLOv8x-world)
- Total storage: ~2.0 GB for all 36 models
- Models are auto-downloaded on first use
- Inference speed depends on model size and hardware

## Requirements

```
ultralytics>=8.0.0
tabulate>=0.9.0
Pillow>=8.0.0
numpy>=1.20.0
torch>=1.9.0
torchvision>=0.10.0
```

## License

MIT License - Feel free to use and modify as needed.
