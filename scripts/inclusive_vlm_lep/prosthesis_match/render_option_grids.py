"""Render option grid images for Prosthesis Matching.

This script renders K option crops into a single grid image for each benchmark item.
This is useful for VLM runtimes that don't support multiple image inputs.

Usage:
    conda activate inclusivevlm-lep
    python -m scripts.inclusive_vlm_lep.prosthesis_match.render_option_grids \
        --items-path data/inclusive_vlm_lep/prosthesis_match/items/prosthesis_match.jsonl \
        --output-dir data/inclusive_vlm_lep/prosthesis_match/grids
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from .config import (
    ALLOWED_CHOICES,
    DEFAULT_DATA_GRIDS_DIR,
    DEFAULT_DATA_ITEMS_PATH,
    GRID_COLS,
    OPTION_POOL_OUT_SIZE,
    PROJECT_ROOT,
)

__all__ = ["render_option_grids", "GridConfig"]

logger = logging.getLogger(__name__)


def _relative_to_project(path: Path) -> str:
    """Return path relative to project root when possible."""

    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class GridConfig:
    """Configuration for grid rendering."""

    grid_cols: int = GRID_COLS
    cell_size: int = OPTION_POOL_OUT_SIZE
    padding: int = 10
    label_height: int = 30
    background_color: Tuple[int, int, int] = (240, 240, 240)
    label_color: Tuple[int, int, int] = (0, 0, 0)
    border_color: Tuple[int, int, int] = (200, 200, 200)
    font_size: int = 20


def render_option_grid(
    option_paths: List[Path],
    option_ids: List[str],
    output_path: Path,
    config: Optional[GridConfig] = None,
) -> Image.Image:
    """Render options into a single grid image.

    Args:
        option_paths: List of paths to option images.
        option_ids: List of option IDs (A, B, C, ...).
        output_path: Path to save grid image.
        config: Grid configuration.

    Returns:
        Grid PIL Image.
    """
    if config is None:
        config = GridConfig()

    n_options = len(option_paths)
    cols = min(config.grid_cols, n_options)
    rows = (n_options + cols - 1) // cols

    # Calculate grid dimensions
    cell_w = config.cell_size + config.padding * 2
    cell_h = config.cell_size + config.label_height + config.padding * 2

    grid_w = cols * cell_w + config.padding
    grid_h = rows * cell_h + config.padding

    # Create grid image
    grid = Image.new("RGB", (grid_w, grid_h), config.background_color)
    draw = ImageDraw.Draw(grid)

    # Try to load a font
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", config.font_size
        )
    except (OSError, IOError):
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", config.font_size
            )
        except (OSError, IOError):
            font = ImageFont.load_default()

    for i, (opt_path, opt_id) in enumerate(zip(option_paths, option_ids)):
        row = i // cols
        col = i % cols

        # Calculate cell position
        x = config.padding + col * cell_w
        y = config.padding + row * cell_h

        # Draw border
        draw.rectangle(
            [x, y, x + cell_w - config.padding, y + cell_h - config.padding],
            outline=config.border_color,
            width=2,
        )

        # Draw option label
        label_x = x + config.padding
        label_y = y + config.padding

        draw.text(
            (label_x + config.cell_size // 2, label_y + config.label_height // 2),
            opt_id,
            fill=config.label_color,
            font=font,
            anchor="mm",
        )

        # Load and paste option image
        try:
            opt_img = Image.open(opt_path).convert("RGB")
            # Resize to fit cell
            opt_img.thumbnail(
                (config.cell_size, config.cell_size), Image.Resampling.LANCZOS
            )

            # Center in cell
            img_x = x + config.padding + (config.cell_size - opt_img.width) // 2
            img_y = (
                y
                + config.label_height
                + config.padding
                + (config.cell_size - opt_img.height) // 2
            )

            grid.paste(opt_img, (img_x, img_y))

        except FileNotFoundError:
            logger.warning(f"Option image not found: {opt_path}")
            # Draw placeholder
            placeholder_x = x + config.padding
            placeholder_y = y + config.label_height + config.padding
            draw.rectangle(
                [
                    placeholder_x,
                    placeholder_y,
                    placeholder_x + config.cell_size,
                    placeholder_y + config.cell_size,
                ],
                fill=(200, 200, 200),
                outline=config.border_color,
            )
            draw.text(
                (
                    placeholder_x + config.cell_size // 2,
                    placeholder_y + config.cell_size // 2,
                ),
                "?",
                fill=config.label_color,
                font=font,
                anchor="mm",
            )

    # Save grid
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path, "PNG")

    return grid


def render_option_grids(
    items_path: Path,
    output_dir: Path,
    config: Optional[GridConfig] = None,
) -> Dict[str, Path]:
    """Render option grids for all benchmark items.

    Args:
        items_path: Path to items JSONL file.
        output_dir: Directory to save grid images.
        config: Grid configuration.

    Returns:
        Dictionary mapping item_id to grid image path.
    """
    if config is None:
        config = GridConfig()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grid_paths: Dict[str, Path] = {}

    # Load items
    items = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    logger.info(f"Rendering grids for {len(items)} items...")

    for i, item in enumerate(items):
        item_id = item["item_id"]

        # Extract option paths and IDs
        option_paths = []
        option_ids = []
        for opt in item["options"]:
            option_paths.append(Path(opt["path"]))
            option_ids.append(opt["option_id"])

        # Generate output path
        # Use item_id hash for unique filename
        grid_filename = f"grid_{item_id.replace(':', '_')}.png"
        grid_path = output_dir / grid_filename

        # Render grid
        render_option_grid(option_paths, option_ids, grid_path, config)
        grid_paths[item_id] = grid_path

        if (i + 1) % 100 == 0:
            logger.info(f"Rendered {i + 1}/{len(items)} grids...")

    # Save grid manifest
    manifest_path = output_dir / "grids_manifest.json"
    manifest = {
        item_id: str(path.relative_to(output_dir))
        for item_id, path in grid_paths.items()
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Rendered {len(grid_paths)} grids")
    logger.info(f"Manifest saved to {manifest_path}")

    return grid_paths


def update_items_with_grids(
    items_path: Path,
    grids_dir: Path,
    output_path: Optional[Path] = None,
) -> None:
    """Update items JSONL with grid image paths.

    Args:
        items_path: Path to original items JSONL.
        grids_dir: Directory containing grid images.
        output_path: Path to save updated items. If None, overwrite original.
    """
    if output_path is None:
        output_path = items_path

    grids_dir = Path(grids_dir)
    manifest_path = grids_dir / "grids_manifest.json"

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Load and update items
    updated_items = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                item_id = item["item_id"]

                if item_id in manifest:
                    grid_path = grids_dir / manifest[item_id]
                    item["options_grid"] = {
                        "enabled": True,
                        "path": _relative_to_project(grid_path),
                    }
                else:
                    item["options_grid"] = {
                        "enabled": False,
                        "path": None,
                    }

                updated_items.append(item)

    # Save updated items
    with open(output_path, "w", encoding="utf-8") as f:
        for item in updated_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info(f"Updated {len(updated_items)} items with grid paths")
    logger.info(f"Saved to {output_path}")


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
        description="Render option grids for Prosthesis Matching."
    )
    parser.add_argument(
        "--items-path",
        type=Path,
        default=DEFAULT_DATA_ITEMS_PATH,
        help=f"Path to items JSONL file. Default: {DEFAULT_DATA_ITEMS_PATH}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_DATA_GRIDS_DIR,
        help=f"Output directory for grids. Default: {DEFAULT_DATA_GRIDS_DIR}",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=GRID_COLS,
        help=f"Number of columns in grid. Default: {GRID_COLS}",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=OPTION_POOL_OUT_SIZE,
        help=f"Size of each cell. Default: {OPTION_POOL_OUT_SIZE}",
    )
    parser.add_argument(
        "--update-items",
        action="store_true",
        help="Update items JSONL with grid paths.",
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
    logger.info("Prosthesis Matching: Render Option Grids")
    logger.info("=" * 60)

    # Create config
    config = GridConfig(
        grid_cols=args.grid_cols,
        cell_size=args.cell_size,
    )

    # Render grids
    grid_paths = render_option_grids(args.items_path, args.output_dir, config)

    # Optionally update items with grid paths
    if args.update_items:
        update_items_with_grids(args.items_path, args.output_dir)

    logger.info("=" * 60)
    logger.info(f"Done! Rendered {len(grid_paths)} option grids.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
