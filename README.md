# YoloLLM - YOLO Object Detection, Segmentation & Pose Estimation

A comprehensive Python toolkit for object detection, instance segmentation, and pose estimation using multiple YOLO and SAM model families. Detect objects, segment images, and estimate human poses with support for 70+ pre-trained models.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
  - [Object Detection](#basic-usage)
  - [Instance Segmentation](#segmentation-usage)
  - [Pose Estimation](#pose-estimation)
  - [FastSAM Segmentation](#fastsam-segmentation)
  - [SAM 2/2.1 Segmentation](#sam-221-segmentation)
  - [YOLOE Segmentation](#yoloe-segmentation)
- [Examples](#examples)
- [Available Models](#available-models)
- [Performance Notes](#performance-notes)
- [Requirements](#requirements)

## Quick Start

```bash
# Install dependencies
pip install ultralytics tabulate opencv-python numpy

# Object Detection
python yolo_detect.py

# Pose Estimation
python yolo_pose.py --model yolo11n --source image.jpg

# SAM Segmentation
python yolo_sam.py -i image.jpg -m sam2.1-base

# FastSAM Segmentation
python yolo_fastsam.py -i image.jpg -m small

# YOLOE Text Prompt Segmentation
python yoloe_segment.py image.jpg -m 11l -p person car
```

## Features

- **80+ Pre-trained Models** across multiple families:
  - YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO12, YOLO26, RT-DETR (46 detection models including OBB variants)
  - YOLO Pose models (16 models)
  - SAM 2 & SAM 2.1 models (8 models)
  - Mobile SAM (1 model)
  - FastSAM (2 models)
  - YOLOE Segmentation (12 models with text prompt support)
  - YOLO26 & YOLOE-26 Segmentation (10 models)

- **Object Detection** with interactive model selection via command-line menu

- **Instance Segmentation** - Multiple approaches:
  - Standard YOLO segmentation
  - FastSAM - Fast Segment Anything
  - SAM 2/2.1 - Segment Anything Model v2 and v2.1
  - Mobile SAM - Lightweight SAM for mobile devices
  - YOLOE - Text prompt-based and prompt-free segmentation

- **Pose Estimation** - Detect human keypoints and poses using YOLO11 and YOLOv8 pose models

- **Multiple Output Formats**:
  - Table format (ASCII grid with detailed statistics)
  - JSON file export
  - Segmentation masks (PNG format)
  - Annotated images with keypoints

- **Advanced Detection Metrics**:
  - Bounding box coordinates (multiple formats: xyxy, xywh, normalized variants)
  - Confidence scores
  - Object area as percentage of total image area
  - Keypoint coordinates and confidence scores
  - Results sorted by detection size (descending)

- **Mask Processing** - Utilities for post-processing segmentation masks with overlay and transparency controls

- **Clean, Modular Code** - Well-organized classes with helper methods and constants

## Installation

### Requirements
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to the project directory
cd YoloLLM

# Install dependencies
pip install -r requirements.txt
# Or manually install:
pip install ultralytics tabulate opencv-python numpy

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

### Object Detection Models (46 total)

**YOLOv8 (13 models)**
- Standard: YOLOv8n, YOLOv8s, YOLOv8m, YOLOv8l, YOLOv8x
- World: YOLOv8s-world, YOLOv8s-worldv2, YOLOv8m-world, YOLOv8m-worldv2, YOLOv8l-world, YOLOv8l-worldv2, YOLOv8x-world, YOLOv8x-worldv2

**YOLOv9 (5 models)**
- YOLOv9t, YOLOv9s, YOLOv9m, YOLOv9c, YOLOv9e

**YOLOv10 (6 models)**
- YOLOv10n, YOLOv10s, YOLOv10m, YOLOv10b, YOLOv10l, YOLOv10x

**YOLO11 (5 models)**
- YOLO11n, YOLO11s, YOLO11m, YOLO11l, YOLO11x

**YOLO12 (5 models)**
- YOLO12n, YOLO12s, YOLO12m, YOLO12l, YOLO12x

**YOLO26 (10 models)**
- Standard: YOLO26n, YOLO26s, YOLO26m, YOLO26l, YOLO26x
- OBB (Oriented Bounding Box): YOLO26n-obb, YOLO26s-obb, YOLO26m-obb, YOLO26l-obb, YOLO26x-obb

**RT-DETR (2 models)**
- RT-DETR-l, RT-DETR-x

### Pose Estimation Models (16 total)

**YOLO11 Pose (5 models)**
- yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose

**YOLOv8 Pose (6 models)**
- yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose, yolov8x-pose-p6

**YOLO26 Pose (5 models)**
- yolo26n-pose, yolo26s-pose, yolo26m-pose, yolo26l-pose, yolo26x-pose

### Segmentation Models (33 total)

**SAM 2 (4 models)**
- sam2-tiny, sam2-small, sam2-base, sam2-large

**SAM 2.1 (4 models)**
- sam2.1-tiny, sam2.1-small, sam2.1-base, sam2.1-large

**Mobile SAM (1 model)**
- mobile-sam

**FastSAM (2 models)**
- small (FastSAM-s), large (FastSAM-x)

**YOLOE Text Prompt (6 models)**
- 11s, 11m, 11l, v8s, v8m, v8l

**YOLOE Prompt-Free (6 models)**
- 11s-pf, 11m-pf, 11l-pf, v8s-pf, v8m-pf, v8l-pf

**YOLO26 Segmentation (5 models)**
- v26-nano, v26-small, v26-medium, v26-large, v26-xlarge

**YOLOE-26 Segmentation (5 models)**
- ve26-nano, ve26-small, ve26-medium, ve26-large, ve26-xlarge

### Model Naming Convention
- **n** = Nano (smallest, fastest)
- **s** = Small
- **m** = Medium
- **l** = Large
- **x** = X-Large (largest, most accurate)
- **-pf** = Prompt-Free (for YOLOE models)

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
├── yolo_detect.py          # Object detection module (36 models)
├── yolo_segment.py         # Standard instance segmentation
├── yolo_pose.py            # Pose estimation module (11 models)
├── yolo_fastsam.py         # FastSAM segmentation (2 models)
├── yolo_sam.py             # SAM 2/2.1 & Mobile SAM (9 models)
├── yoloe_segment.py        # YOLOE text prompt-based segmentation (12 models)
├── mask_processing.py      # Mask post-processing utilities
├── models/                 # Model storage (auto-created)
│   ├── yolov8n.pt
│   ├── yolo11x.pt
│   ├── yolo11n-pose.pt
│   ├── sam2_b.pt
│   ├── FastSAM-s.pt
│   ├── yoloe-11l-seg.pt
│   └── ... (60+ more models)
├── images/                 # Sample images for testing
├── data/                   # Data directory
├── seg_results/            # Segmentation output (generated)
├── masks/                  # Individual mask outputs (generated)
├── detections.json         # Detection output (generated)
├── requirements.txt        # Python dependencies
├── tests/                  # Test suite
│   ├── test_detection.py
│   ├── test_all_models.py
│   └── test_all_models_quiet.py
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

## Segmentation Usage

### Basic Segmentation

```bash
python yolo_segment.py
```

This will:
1. Prompt you to select a model
2. Ask for an image path
3. Generate segmentation masks
4. Save results to `seg_results/`

### Python API

```python
from yolo_segment import YoloSegmenter

# Initialize segmenter
segmenter = YoloSegmenter(model_name="yolov8n-seg.pt")

# Run segmentation
segmenter.segment("image.jpg")

# Get segmentation masks
masks = segmenter.get_masks()

# Save results
segmenter.save_results("output_dir/")
```

## Pose Estimation

### Basic Pose Estimation

```bash
# Interactive mode
python yolo_pose.py

# Command-line mode
python yolo_pose.py --model yolo11n --source image.jpg --conf 0.5

# List available models
python yolo_pose.py --list-models
```

### Python API

```python
from yolo_pose import YoloPose

# Initialize pose estimator
pose = YoloPose(model="yolo11n")

# Run pose estimation
results = pose.predict("image.jpg", conf=0.5)

# Get keypoints
keypoints = pose.get_keypoints(results)
for i, kpt in enumerate(keypoints):
    print(f"Person {i}: {len(kpt['xy'])} keypoints")

# Get all data (bboxes + keypoints)
detections = pose.get_all_data(results)
```

### Available Pose Models

**YOLO11 Pose Models (5)**
- yolo11n-pose, yolo11s-pose, yolo11m-pose, yolo11l-pose, yolo11x-pose

**YOLOv8 Pose Models (6)**
- yolov8n-pose, yolov8s-pose, yolov8m-pose, yolov8l-pose, yolov8x-pose, yolov8x-pose-p6

## FastSAM Segmentation

### Basic Usage

```bash
# Segment an image with default settings
python yolo_fastsam.py -i image.jpg

# Use large model with custom output
python yolo_fastsam.py -i image.jpg -m large -o result.jpg

# Save individual masks and adjust transparency
python yolo_fastsam.py -i image.jpg -s -d my_masks -t 0.5
```

### Python API

```python
from yolo_fastsam import YoloFastSAM

# Initialize with small or large model
fastsam = YoloFastSAM(model_name="small")

# Run segmentation
segments = fastsam.segment("image.jpg")
print(f"Found {len(segments)} segments")

# Access individual masks
for segment in segments:
    print(f"{segment.label}: {segment.mask.shape}")
```

### Available FastSAM Models
- **small**: FastSAM-s.pt (faster, lower accuracy)
- **large**: FastSAM-x.pt (slower, higher accuracy)

## SAM 2/2.1 Segmentation

### Basic Usage

```bash
# Segment with SAM 2.1 base model (default)
python yolo_sam.py -i image.jpg

# Use SAM 2.1 large model
python yolo_sam.py -i image.jpg -m sam2.1-large

# Mobile SAM for lightweight processing
python yolo_sam.py -i image.jpg -m mobile-sam

# Save individual masks
python yolo_sam.py -i image.jpg -s -d masks_output -t 0.3
```

### Python API

```python
from yolo_sam import YoloSAM

# Initialize with SAM 2.1 base model
sam = YoloSAM(model_name="sam2.1-base")

# Run segmentation
segments = sam.segment("image.jpg")
print(f"Found {len(segments)} segments")

# Access masks (sorted by area)
for i, segment in enumerate(segments[:5]):
    print(f"Segment {i}: {segment.label}")
```

### Available SAM Models

**SAM 2 Models (4)**
- sam2-tiny, sam2-small, sam2-base, sam2-large

**SAM 2.1 Models (4)**
- sam2.1-tiny, sam2.1-small, sam2.1-base, sam2.1-large

**Mobile SAM (1)**
- mobile-sam (lightweight for mobile/edge devices)

## YOLOE Segmentation

YOLOE provides text prompt-based and prompt-free instance segmentation.

### Text Prompt-Based Segmentation

```bash
# Segment specific objects using text prompts
python yoloe_segment.py image.jpg -m 11l -p person car dog --show

# Save output with custom prompts
python yoloe_segment.py image.jpg -m v8s -p cat bird -o output.jpg
```

### Prompt-Free Segmentation

```bash
# Detect all COCO classes automatically
python yoloe_segment.py image.jpg -m 11l-pf --show

# Use larger prompt-free model
python yoloe_segment.py image.jpg -m v8l-pf -o result.jpg
```

### Python API

```python
from yoloe_segment import YoloESegment

# Text prompt-based model
yoloe = YoloESegment(model_name="11l", classes=["person", "car", "dog"])
results = yoloe.segment("image.jpg")
yoloe.save("output.jpg")

# Prompt-free model
yoloe_pf = YoloESegment(model_name="11l-pf")
results = yoloe_pf.segment("image.jpg")
yoloe_pf.show()
```

### Available YOLOE Models

**Text Prompt Models (6)**
- 11s, 11m, 11l, v8s, v8m, v8l

**Prompt-Free Models (6)**
- 11s-pf, 11m-pf, 11l-pf, v8s-pf, v8m-pf, v8l-pf

## Mask Processing

```python
from mask_processing import save_masks, overlay_masks

# Save individual masks to directory
save_masks(segments, output_dir="masks")

# Overlay masks on original image with transparency
overlay_masks(segments, image_path="image.jpg", output_path="result.jpg", alpha=0.25)
```

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

### Example 4: Pose Estimation

```python
from yolo_pose import YoloPose

# Detect human poses
pose = YoloPose(model="yolo11m")
results = pose.predict("people.jpg", conf=0.5)

# Extract keypoints
keypoints = pose.get_keypoints(results)
print(f"Detected {len(keypoints)} people")

for i, kpt in enumerate(keypoints):
    xy_coords = kpt['xy']
    confidences = kpt['confidence']
    print(f"Person {i}: {len(xy_coords)} keypoints detected")
```

### Example 5: SAM Segmentation

```python
from yolo_sam import YoloSAM
from mask_processing import save_masks, overlay_masks

# Use SAM 2.1 large for high-quality segmentation
sam = YoloSAM(model_name="sam2.1-large")
segments = sam.segment("scene.jpg")

print(f"Found {len(segments)} segments")

# Save individual masks
save_masks(segments, output_dir="scene_masks")

# Create overlay visualization
overlay_masks(segments, "scene.jpg", "scene_segmented.jpg", alpha=0.3)
```

### Example 6: YOLOE Text Prompt Segmentation

```python
from yoloe_segment import YoloESegment

# Segment only specific objects mentioned in prompts
yoloe = YoloESegment(model_name="11l", classes=["person", "bicycle", "car"])
results = yoloe.segment("street.jpg")
yoloe.save("street_segmented.jpg")

# Use prompt-free model for all COCO classes
yoloe_pf = YoloESegment(model_name="11l-pf")
results = yoloe_pf.segment("street.jpg")
yoloe_pf.save("street_all_objects.jpg")
```

## Performance Notes

### Model Sizes
- Detection models: 4.7 MB (YOLOv9t) to 141 MB (YOLOv8x-world)
- Pose models: 4.5 MB (YOLO11n-pose) to 194 MB (YOLOv8x-pose-p6)
- SAM models: 40 MB (Mobile SAM) to 224 MB (SAM 2.1 Large)
- FastSAM models: 23.7 MB (small) to 138.8 MB (large)
- YOLOE models: 25 MB to 115 MB

### Storage
- All detection models (~36): ~2.0 GB
- All pose models (11): ~480 MB
- All SAM models (9): ~1.2 GB
- All FastSAM models (2): ~163 MB
- All YOLOE models (12): ~760 MB
- **Total for all 70+ models**: ~4.6 GB

### Performance
- Models are auto-downloaded on first use
- Inference speed depends on model size and hardware
- Nano/Tiny models: Fastest, suitable for real-time applications
- Small/Medium models: Balanced speed and accuracy
- Large/X-Large models: Highest accuracy, slower inference
- Mobile models: Optimized for edge devices and mobile platforms

## Requirements

```
ultralytics>=8.0.0
tabulate>=0.9.0
opencv-python>=4.5.0
Pillow>=8.0.0
numpy>=1.20.0
torch>=1.9.0
torchvision>=0.10.0
```

Note: Install all dependencies using `pip install -r requirements.txt`

## License

MIT License - Feel free to use and modify as needed.
