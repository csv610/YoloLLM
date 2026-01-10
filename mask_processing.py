import os
from PIL import Image
import numpy as np
import colorsys


def save_masks(segments, output_dir="."):
    """
    Save segmentation masks as PNG images.

    Args:
        segments: List of Segment objects
        output_dir: Directory to save mask images (default: current directory)
    """
    if not segments:
        print("No segments to save")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Track count for each label
    label_counts = {}

    for segment in segments:
        # Increment count for this label
        label_counts[segment.label] = label_counts.get(segment.label, 0) + 1
        number = label_counts[segment.label]

        # Convert binary mask to 8-bit image (0 or 255)
        mask_image = (segment.mask * 255).astype(np.uint8)

        # Convert numpy array to PIL Image
        pil_image = Image.fromarray(mask_image)

        # Save as PNG
        filename = f"mask_{segment.label}_{number}.png"
        filepath = os.path.join(output_dir, filename)
        pil_image.save(filepath)
        print(f"Saved {filepath}")


def overlay_masks(segments, image_path, output_path="overlay.png", alpha=0.6):
    """
    Overlay segmentation masks on the original image.

    Args:
        segments: List of Segment objects
        image_path: Path or URL to the original image
        output_path: Path to save the overlay image (default: overlay.png)
        alpha: Transparency of masks (0-1, default: 0.6)
    """
    if not segments:
        print("No segments to overlay")
        return

    # Load original image
    original_image = Image.open(image_path).convert("RGB")
    original_array = np.array(original_image)
    image_height, image_width = original_array.shape[:2]

    # Create overlay image as copy of original
    overlay_array = original_array.copy().astype(np.float32)

    # Generate distinct colors for each unique label
    unique_labels = list(set(seg.label for seg in segments))
    label_colors = {}
    for i, label in enumerate(unique_labels):
        hue = i / len(unique_labels)
        rgb = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
        label_colors[label] = tuple(int(c * 255) for c in rgb)

    # Overlay each mask
    for segment in segments:
        mask = segment.mask
        color = label_colors[segment.label]

        # Resize mask to match image dimensions if needed
        if mask.shape != (image_height, image_width):
            mask_image = Image.fromarray((mask * 255).astype(np.uint8))
            mask_image = mask_image.resize((image_width, image_height), Image.NEAREST)
            mask = np.array(mask_image) / 255.0

        # Apply mask with color
        mask_bool = mask > 0.5
        for c in range(3):
            overlay_array[mask_bool, c] = (
                overlay_array[mask_bool, c] * (1 - alpha) + color[c] * alpha
            )

    # Convert back to uint8 and save
    overlay_array = np.clip(overlay_array, 0, 255).astype(np.uint8)
    overlay_image = Image.fromarray(overlay_array)
    overlay_image.save(output_path)
    print(f"Saved overlay image: {output_path}")
