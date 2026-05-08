"""Dataset adapter for InclusiveVLM-LEP dataset.

This module provides a unified interface to access the InclusiveVLM-LEP
dataset for Prosthesis Matching.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
from PIL import Image

from .config import (
    DEFAULT_ANNO_PATH,
    DEFAULT_IMAGE_ROOT,
    LIMB_CATEGORY_TO_PART8,
    PROSTHESIS_CATEGORY_IDS,
    PROSTHESIS_CATEGORY_TO_PART8,
    RESIDUAL_CATEGORY_IDS,
    RESIDUAL_CATEGORY_TO_PART8,
)

__all__ = [
    "InclusiveVLMSourceAdapter",
    "ImageInfo",
    "AnnotationInfo",
    "ProsthesisInstance",
    "ResidualLimbInstance",
]

logger = logging.getLogger(__name__)


@dataclass
class ImageInfo:
    """Image metadata."""

    image_id: int
    file_name: str
    width: int
    height: int

    def __hash__(self) -> int:
        return hash(self.image_id)


@dataclass
class AnnotationInfo:
    """Single annotation entry."""

    annotation_id: int
    image_id: int
    category_id: int
    category_name: str
    bbox: List[float]  # [x, y, w, h]
    segmentation: List[List[float]]
    area: float


@dataclass
class ProsthesisInstance:
    """Prosthesis instance with metadata."""

    annotation_id: int
    image_id: int
    category_id: int
    category_name: str
    part8: str
    bbox_xywh: List[float]
    bbox_xyxy: List[float]
    segmentation: List[List[float]]
    area: float

    def __post_init__(self) -> None:
        """Compute bbox_xyxy from bbox_xywh if not set."""
        if not self.bbox_xyxy:
            x, y, w, h = self.bbox_xywh
            self.bbox_xyxy = [x, y, x + w, y + h]


@dataclass
class ResidualLimbInstance:
    """Residual limb instance with metadata."""

    annotation_id: int
    image_id: int
    category_id: int
    category_name: str
    part8: str
    bbox_xywh: List[float]
    bbox_xyxy: List[float]
    segmentation: List[List[float]]
    area: float

    def __post_init__(self) -> None:
        """Compute bbox_xyxy from bbox_xywh if not set."""
        if not self.bbox_xyxy:
            x, y, w, h = self.bbox_xywh
            self.bbox_xyxy = [x, y, x + w, y + h]


class InclusiveVLMSourceAdapter:
    """Adapter for accessing InclusiveVLM-LEP dataset.

    This class provides a unified interface for loading images, annotations,
    and extracting prosthesis/residual limb instances.

    Attributes:
        anno_path: Path to the annotation JSON file.
        image_root: Root directory containing images.
        images: Mapping from image_id to ImageInfo.
        annotations: List of all annotations.
        categories: Mapping from category_id to category_name.
    """

    def __init__(
        self,
        anno_path: Path | str = DEFAULT_ANNO_PATH,
        image_root: Path | str = DEFAULT_IMAGE_ROOT,
    ) -> None:
        """Initialize the adapter.

        Args:
            anno_path: Path to the COCO-format annotation JSON file.
            image_root: Root directory containing images.
        """
        self.anno_path = Path(anno_path)
        self.image_root = Path(image_root)

        self._raw_data: Dict[str, Any] = {}
        self.images: Dict[int, ImageInfo] = {}
        self.annotations: List[AnnotationInfo] = []
        self.categories: Dict[int, str] = {}

        # Indexed by image_id
        self._annotations_by_image: Dict[int, List[AnnotationInfo]] = {}
        self._prostheses_by_image: Dict[int, List[ProsthesisInstance]] = {}
        self._residuals_by_image: Dict[int, List[ResidualLimbInstance]] = {}

        self._loaded = False

    def load(self) -> None:
        """Load and parse the annotation file."""
        if self._loaded:
            return

        logger.info(f"Loading annotations from {self.anno_path}")

        with open(self.anno_path, "r", encoding="utf-8") as f:
            self._raw_data = json.load(f)

        # Parse categories
        for cat in self._raw_data.get("categories", []):
            self.categories[cat["id"]] = cat["name"]

        # Parse images
        for img in self._raw_data.get("images", []):
            self.images[img["id"]] = ImageInfo(
                image_id=img["id"],
                file_name=img["file_name"],
                width=img["width"],
                height=img["height"],
            )

        # Parse annotations
        for ann in self._raw_data.get("annotations", []):
            cat_id = ann["category_id"]
            cat_name = self.categories.get(cat_id, "unknown")

            anno_info = AnnotationInfo(
                annotation_id=ann["id"],
                image_id=ann["image_id"],
                category_id=cat_id,
                category_name=cat_name,
                bbox=ann.get("bbox", [0, 0, 0, 0]),
                segmentation=ann.get("segmentation", []),
                area=ann.get("area", 0),
            )
            self.annotations.append(anno_info)

            # Index by image
            if anno_info.image_id not in self._annotations_by_image:
                self._annotations_by_image[anno_info.image_id] = []
            self._annotations_by_image[anno_info.image_id].append(anno_info)

        # Build prosthesis and residual indexes
        self._build_instance_indexes()

        logger.info(
            f"Loaded {len(self.images)} images, {len(self.annotations)} annotations"
        )
        self._loaded = True

    def _build_instance_indexes(self) -> None:
        """Build indexes for prosthesis and residual limb instances."""
        prosthesis_cat_ids = set(PROSTHESIS_CATEGORY_IDS.keys())
        residual_cat_ids = set(RESIDUAL_CATEGORY_IDS.keys())

        for anno in self.annotations:
            cat_id = anno.category_id
            x, y, w, h = anno.bbox
            bbox_xyxy = [x, y, x + w, y + h]

            if cat_id in prosthesis_cat_ids:
                instance = ProsthesisInstance(
                    annotation_id=anno.annotation_id,
                    image_id=anno.image_id,
                    category_id=cat_id,
                    category_name=anno.category_name,
                    part8=PROSTHESIS_CATEGORY_TO_PART8[cat_id],
                    bbox_xywh=anno.bbox,
                    bbox_xyxy=bbox_xyxy,
                    segmentation=anno.segmentation,
                    area=anno.area,
                )
                if anno.image_id not in self._prostheses_by_image:
                    self._prostheses_by_image[anno.image_id] = []
                self._prostheses_by_image[anno.image_id].append(instance)

            elif cat_id in residual_cat_ids:
                instance = ResidualLimbInstance(
                    annotation_id=anno.annotation_id,
                    image_id=anno.image_id,
                    category_id=cat_id,
                    category_name=anno.category_name,
                    part8=RESIDUAL_CATEGORY_TO_PART8[cat_id],
                    bbox_xywh=anno.bbox,
                    bbox_xyxy=bbox_xyxy,
                    segmentation=anno.segmentation,
                    area=anno.area,
                )
                if anno.image_id not in self._residuals_by_image:
                    self._residuals_by_image[anno.image_id] = []
                self._residuals_by_image[anno.image_id].append(instance)

    def list_images(self) -> List[int]:
        """List all image IDs.

        Returns:
            List of image IDs.
        """
        self.load()
        return list(self.images.keys())

    def load_image(self, image_id: int) -> Image.Image:
        """Load an image by ID.

        Args:
            image_id: Image ID.

        Returns:
            PIL Image object.

        Raises:
            KeyError: If image_id not found.
            FileNotFoundError: If image file not found.
        """
        self.load()
        if image_id not in self.images:
            raise KeyError(f"Image ID {image_id} not found")

        img_info = self.images[image_id]
        img_path = self.image_root / img_info.file_name

        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")

        return Image.open(img_path).convert("RGB")

    def get_image_info(self, image_id: int) -> ImageInfo:
        """Get image metadata.

        Args:
            image_id: Image ID.

        Returns:
            ImageInfo object.
        """
        self.load()
        return self.images[image_id]

    def get_image_path(self, image_id: int) -> Path:
        """Get image file path.

        Args:
            image_id: Image ID.

        Returns:
            Path to image file.
        """
        self.load()
        img_info = self.images[image_id]
        return self.image_root / img_info.file_name

    def get_annotations_for_image(self, image_id: int) -> List[AnnotationInfo]:
        """Get all annotations for an image.

        Args:
            image_id: Image ID.

        Returns:
            List of annotations.
        """
        self.load()
        return self._annotations_by_image.get(image_id, [])

    def get_prostheses_for_image(self, image_id: int) -> List[ProsthesisInstance]:
        """Get all prosthesis instances for an image.

        Args:
            image_id: Image ID.

        Returns:
            List of prosthesis instances.
        """
        self.load()
        return self._prostheses_by_image.get(image_id, [])

    def get_residuals_for_image(self, image_id: int) -> List[ResidualLimbInstance]:
        """Get all residual limb instances for an image.

        Args:
            image_id: Image ID.

        Returns:
            List of residual limb instances.
        """
        self.load()
        return self._residuals_by_image.get(image_id, [])

    def iter_prostheses(self) -> Iterator[Tuple[int, ProsthesisInstance]]:
        """Iterate over all prosthesis instances.

        Yields:
            Tuple of (image_id, ProsthesisInstance).
        """
        self.load()
        for image_id, instances in self._prostheses_by_image.items():
            for inst in instances:
                yield image_id, inst

    def iter_residuals(self) -> Iterator[Tuple[int, ResidualLimbInstance]]:
        """Iterate over all residual limb instances.

        Yields:
            Tuple of (image_id, ResidualLimbInstance).
        """
        self.load()
        for image_id, instances in self._residuals_by_image.items():
            for inst in instances:
                yield image_id, inst

    def get_images_with_prostheses(self) -> List[int]:
        """Get list of image IDs that contain prosthesis annotations.

        Returns:
            List of image IDs.
        """
        self.load()
        return list(self._prostheses_by_image.keys())

    def get_images_with_residuals(self) -> List[int]:
        """Get list of image IDs that contain residual limb annotations.

        Returns:
            List of image IDs.
        """
        self.load()
        return list(self._residuals_by_image.keys())

    def get_images_with_limb_deficiency(self) -> List[int]:
        """Get list of image IDs that contain either prosthesis or residual limb.

        Returns:
            List of image IDs.
        """
        self.load()
        return list(
            set(self._prostheses_by_image.keys()) | set(self._residuals_by_image.keys())
        )

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics.

        Returns:
            Dictionary with statistics.
        """
        self.load()

        total_prostheses = sum(len(v) for v in self._prostheses_by_image.values())
        total_residuals = sum(len(v) for v in self._residuals_by_image.values())

        # Count by part8
        prostheses_by_part: Dict[str, int] = {}
        for instances in self._prostheses_by_image.values():
            for inst in instances:
                prostheses_by_part[inst.part8] = (
                    prostheses_by_part.get(inst.part8, 0) + 1
                )

        residuals_by_part: Dict[str, int] = {}
        for instances in self._residuals_by_image.values():
            for inst in instances:
                residuals_by_part[inst.part8] = residuals_by_part.get(inst.part8, 0) + 1

        return {
            "total_images": len(self.images),
            "total_annotations": len(self.annotations),
            "images_with_prostheses": len(self._prostheses_by_image),
            "images_with_residuals": len(self._residuals_by_image),
            "total_prostheses": total_prostheses,
            "total_residuals": total_residuals,
            "prostheses_by_part": prostheses_by_part,
            "residuals_by_part": residuals_by_part,
        }
