import colorsys
import logging
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Segment:
    """Represents a single segmentation mask with its label."""

    def __init__(self, mask: np.ndarray, label: str = "object"):
        self.mask = mask
        self.label = label


def save_masks(segments: List[Segment], output_dir: str = ".") -> None:
    """Save segmentation masks as PNG images.

    Args:
        segments: List of Segment objects.
        output_dir: Directory to save mask images.
    """
    if not segments:
        logger.warning("No segments to save.")
        return

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    label_counts: dict[str, int] = {}

    for segment in segments:
        label_counts[segment.label] = label_counts.get(segment.label, 0) + 1
        number = label_counts[segment.label]

        mask_image = (segment.mask * 255).astype(np.uint8)
        pil_image = Image.fromarray(mask_image)

        filename = f"mask_{segment.label}_{number}.png"
        filepath = Path(output_dir) / filename
        pil_image.save(filepath)
        logger.info("Saved %s", filepath)


def overlay_masks(
    segments: List[Segment],
    image_path: str,
    output_path: str = "overlay.png",
    alpha: float = 0.6,
) -> None:
    """Overlay segmentation masks on the original image.

    Args:
        segments: List of Segment objects.
        image_path: Path to the original image.
        output_path: Path to save the overlay image.
        alpha: Mask transparency (0 = opaque, 1 = invisible).
    """
    if not segments:
        logger.warning("No segments to overlay.")
        return

    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)
    image_height, image_width = original_array.shape[:2]

    overlay_array = original_array.copy().astype(np.float32)

    unique_labels = list({seg.label for seg in segments})
    label_colors: dict[str, tuple[int, int, int]] = {}
    n_labels = len(unique_labels)
    for i, label in enumerate(unique_labels):
        hue = i / n_labels if n_labels > 1 else 0.0
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        label_colors[label] = tuple(int(c * 255) for c in rgb)

    for segment in segments:
        mask = segment.mask
        color = label_colors[segment.label]

        if mask.shape != (image_height, image_width):
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_image = mask_image.resize((image_width, image_height), Image.NEAREST)
            mask = np.array(mask_image) / 255.0

        mask_bool = mask > 0.5
        for c in range(3):
            overlay_array[mask_bool, c] = (
                overlay_array[mask_bool, c] * alpha + color[c] * (1 - alpha)
            )

    overlay_array = np.clip(overlay_array, 0, 255).astype(np.uint8)
    overlay_image = Image.fromarray(overlay_array)
    overlay_image.save(output_path)
    logger.info("Saved overlay image: %s", output_path)

