"""Build prosthesis option pool from InclusiveVLM-LEP dataset.

This script extracts prosthesis crops from the dataset and builds a reusable
option pool for Prosthesis Matching.

Usage:
    conda activate inclusivevlm-lep
    python -m scripts.inclusive_vlm_lep.prosthesis_match.build_options_pool \
        --anno-path data/inclusive_vlm_lep/annotations/full_dataset.json \
        --image-root data/inclusive_vlm_lep/images \
        --output-dir outputs/inclusive_vlm_lep/prosthesis_matching/options
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image, ImageDraw

from .config import (
    DEFAULT_ANNO_PATH,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_OPTIONS_DIR,
    DEFAULT_SEED,
    OPTION_POOL_MIN_AREA,
    OPTION_POOL_OUT_SIZE,
    OPTION_POOL_PAD_PX,
    PART8_TO_COARSE,
)
from .dataset_adapter import InclusiveVLMSourceAdapter, ProsthesisInstance

__all__ = ["build_options_pool", "OptionEntry", "OptionsPoolConfig"]

logger = logging.getLogger(__name__)


@dataclass
class OptionsPoolConfig:
    """Configuration for option pool construction."""

    min_area: int = OPTION_POOL_MIN_AREA
    pad_px: int = OPTION_POOL_PAD_PX
    out_size: int = OPTION_POOL_OUT_SIZE
    max_aspect_ratio: float = 4.0
    min_aspect_ratio: float = 0.25
    seed: int = DEFAULT_SEED


@dataclass
class OptionEntry:
    """Metadata for a single prosthesis option image."""

    option_id: str
    source_image_id: int
    annotation_id: int
    bbox_xyxy: List[float]
    part8: str
    coarse_part: str
    category_name: str
    area: float
    crop_width: int
    crop_height: int
    file_name: str
    hash: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def polygon_to_mask(
    segmentation: List[List[float]],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert COCO polygon segmentation to binary mask.

    Args:
        segmentation: List of polygon coordinates.
        width: Image width.
        height: Image height.

    Returns:
        Binary mask as numpy array (H, W).
    """
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)

    for polygon in segmentation:
        if len(polygon) < 6:
            continue
        # Convert flat list to (x, y) tuples
        coords = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
        draw.polygon(coords, fill=255)

    return np.array(mask)


def crop_prosthesis(
    image: Image.Image,
    instance: ProsthesisInstance,
    config: OptionsPoolConfig,
) -> Optional[Image.Image]:
    """Crop prosthesis region from image with padding.

    Args:
        image: Source PIL Image.
        instance: ProsthesisInstance with bbox and segmentation.
        config: Option pool configuration.

    Returns:
        Cropped and resized PIL Image, or None if quality check fails.
    """
    # Check area
    if instance.area < config.min_area:
        logger.debug(
            f"Skipping annotation {instance.annotation_id}: area {instance.area} <"
            f" {config.min_area}"
        )
        return None

    # Get bbox and add padding
    x1, y1, x2, y2 = instance.bbox_xyxy
    w, h = x2 - x1, y2 - y1

    # Check aspect ratio
    aspect = w / h if h > 0 else float("inf")
    if aspect > config.max_aspect_ratio or aspect < config.min_aspect_ratio:
        logger.debug(
            f"Skipping annotation {instance.annotation_id}: aspect ratio {aspect:.2f}"
        )
        return None

    # Add padding
    pad = config.pad_px
    x1_pad = max(0, x1 - pad)
    y1_pad = max(0, y1 - pad)
    x2_pad = min(image.width, x2 + pad)
    y2_pad = min(image.height, y2 + pad)

    # Crop region
    crop = image.crop((int(x1_pad), int(y1_pad), int(x2_pad), int(y2_pad)))

    # Create mask for the cropped region
    mask = polygon_to_mask(
        instance.segmentation,
        image.width,
        image.height,
    )
    mask_crop = mask[int(y1_pad) : int(y2_pad), int(x1_pad) : int(x2_pad)]

    # Apply mask to create RGBA image with transparent background
    crop_rgba = crop.convert("RGBA")
    crop_array = np.array(crop_rgba)

    # Create alpha channel from mask
    alpha = np.zeros_like(mask_crop)
    alpha[mask_crop > 0] = 255

    # Apply alpha
    crop_array[:, :, 3] = alpha

    # Convert back to PIL and add white background
    crop_rgba = Image.fromarray(crop_array, "RGBA")

    # Create white background
    background = Image.new("RGB", crop_rgba.size, (255, 255, 255))
    background.paste(crop_rgba, mask=crop_rgba.split()[3])

    # Resize to target size while maintaining aspect ratio
    background.thumbnail((config.out_size, config.out_size), Image.Resampling.LANCZOS)

    # Pad to square
    result = Image.new("RGB", (config.out_size, config.out_size), (255, 255, 255))
    paste_x = (config.out_size - background.width) // 2
    paste_y = (config.out_size - background.height) // 2
    result.paste(background, (paste_x, paste_y))

    return result


def compute_image_hash(image: Image.Image) -> str:
    """Compute hash of image content.

    Args:
        image: PIL Image.

    Returns:
        SHA256 hash string (first 16 characters).
    """
    img_bytes = image.tobytes()
    return hashlib.sha256(img_bytes).hexdigest()[:16]


def build_options_pool(
    adapter: InclusiveVLMSourceAdapter,
    output_dir: Path,
    config: Optional[OptionsPoolConfig] = None,
) -> List[OptionEntry]:
    """Build prosthesis option pool from dataset.

    Args:
        adapter: InclusiveVLMSourceAdapter instance.
        output_dir: Directory to save option images.
        config: Option pool configuration.

    Returns:
        List of OptionEntry objects.
    """
    if config is None:
        config = OptionsPoolConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries: List[OptionEntry] = []
    option_counter = 0
    skipped_count = 0

    logger.info("Building prosthesis option pool...")

    for image_id, instance in adapter.iter_prostheses():
        try:
            image = adapter.load_image(image_id)
        except FileNotFoundError as e:
            logger.warning(f"Image not found: {e}")
            skipped_count += 1
            continue

        # Crop prosthesis
        crop = crop_prosthesis(image, instance, config)
        if crop is None:
            skipped_count += 1
            continue

        # Generate option ID and file name
        option_id = f"opt_{option_counter:06d}"
        file_name = f"{option_id}.png"
        file_path = output_dir / file_name

        # Save crop
        crop.save(file_path, "PNG")

        # Compute hash
        img_hash = compute_image_hash(crop)

        # Create entry
        entry = OptionEntry(
            option_id=option_id,
            source_image_id=image_id,
            annotation_id=instance.annotation_id,
            bbox_xyxy=instance.bbox_xyxy,
            part8=instance.part8,
            coarse_part=PART8_TO_COARSE[instance.part8],
            category_name=instance.category_name,
            area=instance.area,
            crop_width=crop.width,
            crop_height=crop.height,
            file_name=file_name,
            hash=img_hash,
        )
        entries.append(entry)
        option_counter += 1

        if option_counter % 100 == 0:
            logger.info(f"Processed {option_counter} prosthesis crops...")

    # Save manifest
    manifest_path = output_dir / "options_manifest.jsonl"
    with open(manifest_path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "total_options": len(entries),
        "skipped": skipped_count,
        "config": asdict(config),
        "by_part8": {},
        "by_coarse_part": {},
    }

    for entry in entries:
        summary["by_part8"][entry.part8] = summary["by_part8"].get(entry.part8, 0) + 1
        summary["by_coarse_part"][entry.coarse_part] = (
            summary["by_coarse_part"].get(entry.coarse_part, 0) + 1
        )

    summary_path = output_dir / "options_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        f"Built option pool with {len(entries)} options, skipped {skipped_count}"
    )
    logger.info(f"Manifest saved to {manifest_path}")
    logger.info(f"Summary saved to {summary_path}")

    return entries


def load_options_manifest(manifest_path: Path) -> List[OptionEntry]:
    """Load options manifest from JSONL file.

    Args:
        manifest_path: Path to options_manifest.jsonl.

    Returns:
        List of OptionEntry objects.
    """
    entries = []
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                entries.append(OptionEntry(**data))
    return entries


def configure_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Build prosthesis option pool for Prosthesis Matching."
    )
    parser.add_argument(
        "--anno-path",
        type=Path,
        default=DEFAULT_ANNO_PATH,
        help=f"Path to annotation JSON file. Default: {DEFAULT_ANNO_PATH}",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=DEFAULT_IMAGE_ROOT,
        help=f"Root directory for images. Default: {DEFAULT_IMAGE_ROOT}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OPTIONS_DIR,
        help=f"Output directory for options. Default: {DEFAULT_OPTIONS_DIR}",
    )
    parser.add_argument(
        "--min-area",
        type=int,
        default=OPTION_POOL_MIN_AREA,
        help=f"Minimum area for prosthesis crop. Default: {OPTION_POOL_MIN_AREA}",
    )
    parser.add_argument(
        "--pad-px",
        type=int,
        default=OPTION_POOL_PAD_PX,
        help=f"Padding pixels around bbox. Default: {OPTION_POOL_PAD_PX}",
    )
    parser.add_argument(
        "--out-size",
        type=int,
        default=OPTION_POOL_OUT_SIZE,
        help=f"Output image size. Default: {OPTION_POOL_OUT_SIZE}",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Prosthesis Matching: Build Options Pool")
    logger.info("=" * 60)

    # Initialize adapter
    adapter = InclusiveVLMSourceAdapter(
        anno_path=args.anno_path,
        image_root=args.image_root,
    )

    # Create config
    config = OptionsPoolConfig(
        min_area=args.min_area,
        pad_px=args.pad_px,
        out_size=args.out_size,
        seed=args.seed,
    )

    # Build option pool
    entries = build_options_pool(adapter, args.output_dir, config)

    logger.info("=" * 60)
    logger.info(f"Done! Created {len(entries)} prosthesis option images.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
