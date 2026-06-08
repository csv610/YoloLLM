"""Unified command-line entry point for all YoloLLM model types."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from config import SEG_RESULTS_DIR
from mask_processing import overlay_masks, save_masks
from version import __version__

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _handle_version(args: argparse.Namespace) -> bool:
    if getattr(args, "version", False):
        print(f"YoloLLM {__version__}")
        return True
    return False


def cmd_detect(args: argparse.Namespace) -> None:
    if _handle_version(args):
        return
    from yolo_detect import YoloDetector

    detector = YoloDetector(model_name=args.model)
    detector.detect(args.input)
    detector.save_to_json()
    detector.save_annotated_image(source=args.input)


def cmd_segment(args: argparse.Namespace) -> None:
    if _handle_version(args):
        return
    from yolo_segment import YoloSegmentation

    seg = YoloSegmentation(args.model)
    results = seg.segment(args.image)
    seg.print_results(results)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_masks(results, output_dir=str(output_dir))
    overlay_path = output_dir / args.overlay_filename
    overlay_masks(results, args.image, output_path=str(overlay_path), alpha=args.alpha)


def cmd_pose(args: argparse.Namespace) -> None:
    if _handle_version(args):
        return
    from yolo_pose import YoloPose

    if args.list_models:
        print("Available models:")
        for name in YoloPose.available_models:
            print(f"  - {name}")
        return

    pose = YoloPose(model=args.model)
    logger.info("Running inference on: %s", args.source)
    results = pose.predict(args.source, conf=args.conf)

    if args.show_all:
        detections = pose.get_all_data(results)
        print(f"\nTotal detections: {len(detections)}")
        for i, d in enumerate(detections):
            print(f"\nDetection {i}:")
            for k, v in d.items():
                if isinstance(v, dict):
                    for kk, vv in v.items():
                        print(f"  {k}.{kk}: {vv}")
                else:
                    print(f"  {k}: {v}")
    else:
        keypoints = pose.get_keypoints(results)
        print(f"\nDetected {len(keypoints)} people")
        for i, kpt in enumerate(keypoints):
            xy = kpt.get("xy", [])
            assert isinstance(xy, (list, np.ndarray))
            print(f"Person {i}: {len(xy)} keypoints")


def cmd_sam(args: argparse.Namespace) -> None:
    if _handle_version(args):
        return
    from yolo_sam import YoloSAM

    if args.model not in YoloSAM.available_models:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available: {', '.join(YoloSAM.available_models.keys())}")
        sys.exit(1)

    sam = YoloSAM(args.model)
    logger.info("Segmenting image: %s", args.image)
    segments = sam.segment(args.image)
    logger.info("Found %d segments", len(segments))

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        save_masks(segments, output_dir=args.masks_dir)
    overlay_masks(segments, args.image, output_path=args.output, alpha=args.transparency)
    print("Done!")


def cmd_fastsam(args: argparse.Namespace) -> None:
    if _handle_version(args):
        return
    from yolo_fastsam import YoloFastSAM

    if args.model not in YoloFastSAM.available_models:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available: {', '.join(YoloFastSAM.available_models.keys())}")
        sys.exit(1)

    fsam = YoloFastSAM(args.model)
    logger.info("Segmenting image: %s", args.image)
    segments = fsam.segment(args.image)
    logger.info("Found %d segments", len(segments))

    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.save_masks:
        save_masks(segments, output_dir=args.masks_dir)
    overlay_masks(segments, args.image, output_path=args.output, alpha=args.transparency)
    print("Done!")


def cmd_yoloe(args: argparse.Namespace) -> None:
    if _handle_version(args):
        return
    from yoloe_segment import YoloESegment

    if args.model not in YoloESegment.available_models:
        print(f"Error: Invalid model '{args.model}'")
        print(f"Available: {', '.join(YoloESegment.available_models.keys())}")
        sys.exit(1)

    yoloe = YoloESegment(model_name=args.model, classes=args.prompt)

    if yoloe.is_prompt_free:
        logger.info("Using prompt-free model (detects all COCO classes)")
    else:
        logger.info("Using text prompts: %s", yoloe.classes)

    logger.info("Running segmentation on: %s", args.image)
    yoloe.segment(args.image)

    if args.output:
        yoloe.save(args.output)
        logger.info("Results saved to: %s", args.output)
    if args.show:
        yoloe.show()
    print("Done!")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YoloLLM — YOLO/SAM detection, segmentation & pose estimation",
    )
    parser.add_argument("--version", action="store_true", help="Show version and exit")

    sub = parser.add_subparsers(dest="command")

    # detect
    p = sub.add_parser("detect", help="Object detection")
    p.add_argument("--version", action="store_true", help="Show version and exit")
    p.add_argument("-i", "--input", required=True, help="Path to image or video")
    p.add_argument("-m", "--model", default=None, help="Model name (default: v8-large)")
    p.set_defaults(func=cmd_detect)

    # segment
    p = sub.add_parser("segment", help="YOLO segmentation")
    p.add_argument("--version", action="store_true", help="Show version and exit")
    p.add_argument("-i", "--image", default="https://ultralytics.com/images/bus.jpg", help="Image path or URL")
    p.add_argument("-m", "--model", default="v11-nano", help="Model name")
    p.add_argument("-o", "--output-dir", default=str(SEG_RESULTS_DIR), help="Output directory")
    p.add_argument("-a", "--alpha", type=float, default=0.6, help="Mask transparency (0-1)")
    p.add_argument("--overlay-filename", default="overlay.png", help="Overlay filename")
    p.set_defaults(func=cmd_segment)

    # pose
    p = sub.add_parser("pose", help="Pose estimation")
    p.add_argument("--version", action="store_true", help="Show version and exit")
    p.add_argument("--model", default="yolo11n", help="Model name")
    p.add_argument("--source", default="https://ultralytics.com/images/bus.jpg", help="Image source")
    p.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    p.add_argument("--show-all", action="store_true", help="Show bboxes + keypoints")
    p.add_argument("--list-models", action="store_true", help="List available models")
    p.set_defaults(func=cmd_pose)

    # sam
    p = sub.add_parser("sam", help="SAM segmentation")
    p.add_argument("--version", action="store_true", help="Show version and exit")
    p.add_argument("-i", "--image", required=True, help="Path to input image")
    p.add_argument("-m", "--model", default="sam2.1-base", help="SAM model name")
    p.add_argument("-o", "--output", default=str(SEG_RESULTS_DIR / "sam_segment.png"), help="Output path")
    p.add_argument("-t", "--transparency", type=float, default=0.25, help="Mask transparency")
    p.add_argument("-s", "--save-masks", action="store_true", help="Save individual masks")
    p.add_argument("-d", "--masks-dir", default=str(SEG_RESULTS_DIR / "masks"), help="Masks directory")
    p.set_defaults(func=cmd_sam)

    # fastsam
    p = sub.add_parser("fastsam", help="FastSAM segmentation")
    p.add_argument("--version", action="store_true", help="Show version and exit")
    p.add_argument("-i", "--image", required=True, help="Path to input image")
    p.add_argument("-m", "--model", default="fastsam-small", help="FastSAM model name")
    p.add_argument("-o", "--output", default=str(SEG_RESULTS_DIR / "fastsam_segment.png"), help="Output path")
    p.add_argument("-t", "--transparency", type=float, default=0.25, help="Mask transparency")
    p.add_argument("-s", "--save-masks", action="store_true", help="Save individual masks")
    p.add_argument("-d", "--masks-dir", default=str(SEG_RESULTS_DIR / "masks"), help="Masks directory")
    p.set_defaults(func=cmd_fastsam)

    # yoloe
    p = sub.add_parser("yoloe", help="YOLOE instance segmentation")
    p.add_argument("--version", action="store_true", help="Show version and exit")
    p.add_argument("image", type=str, help="Path to input image")
    p.add_argument("-m", "--model", default="yoloe11l", help="YOLOE model name")
    p.add_argument("-p", "--prompt", type=str, nargs="+", help="Text prompts for classes")
    p.add_argument("-o", "--output", default="image_segment.png", help="Output path")
    p.add_argument("--show", action="store_true", help="Display result")
    p.set_defaults(func=cmd_yoloe)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.version and not args.command:
        print(f"YoloLLM {__version__}")
        return

    if args.command:
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

