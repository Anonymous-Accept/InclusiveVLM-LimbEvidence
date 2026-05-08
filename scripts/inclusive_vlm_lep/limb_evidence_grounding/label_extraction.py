"""Extract residual/prosthesis presence labels from COCO annotations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .category_mapping import CategoryParseResult, parse_category_name
from .config import SEGMENTS


@dataclass
class ImageLabels:
    """Aggregated labels for a single image."""

    image_id: int
    file_name: str
    residual_present: Dict[str, bool]
    prosthesis_present: Dict[str, bool]

    @property
    def has_residual(self) -> bool:
        """Return True if any residual limb is present."""

        return any(self.residual_present.values())

    @property
    def has_prosthesis(self) -> bool:
        """Return True if any prosthesis is present."""

        return any(self.prosthesis_present.values())

    @property
    def presence_label(self) -> str:
        """Compute image-level presence label."""

        if self.has_residual and not self.has_prosthesis:
            return "residual_only"
        if not self.has_residual and self.has_prosthesis:
            return "prosthesis_only"
        if self.has_residual and self.has_prosthesis:
            return "residual_and_prosthesis"
        return "none"

    def build_target_labels(self, keep_empty_segments: bool) -> List[str]:
        """Return target label list for Task 2."""

        target_labels: List[str] = []
        for segment in SEGMENTS:
            if self.residual_present.get(segment):
                target_labels.append(f"{segment}_residual")
            if self.prosthesis_present.get(segment):
                target_labels.append(f"{segment}_prosthesis")

        if not target_labels and keep_empty_segments:
            target_labels.append("none")
        return target_labels


def load_annotations(annotations_path: Path) -> Dict[str, List[Dict]]:
    """Load COCO annotations JSON."""

    with annotations_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def build_category_name_map(categories: Iterable[Dict]) -> Dict[int, str]:
    """Create a map from category_id to name."""

    return {cat["id"]: cat["name"] for cat in categories}


def bucket_annotations_by_image(annotations: Iterable[Dict]) -> Dict[int, List[Dict]]:
    """Group annotations by image_id."""

    bucket: Dict[int, List[Dict]] = {}
    for ann in annotations:
        bucket.setdefault(ann["image_id"], []).append(ann)
    return bucket


def _update_presence_flags(
    result: CategoryParseResult,
    residual_present: Dict[str, bool],
    prosthesis_present: Dict[str, bool],
) -> None:
    """Mark residual/prosthesis presence for a parsed category."""

    if result.kind == "residual":
        residual_present[result.segment] = True
    elif result.kind == "prosthesis":
        prosthesis_present[result.segment] = True


def extract_image_labels(
    image: Dict,
    annotations: List[Dict],
    category_name_map: Dict[int, str],
) -> ImageLabels:
    """Aggregate labels for a single image entry."""

    residual_present = {segment: False for segment in SEGMENTS}
    prosthesis_present = {segment: False for segment in SEGMENTS}

    for ann in annotations:
        category_name = category_name_map.get(ann["category_id"])
        if category_name is None:
            continue
        parsed = parse_category_name(category_name)
        if parsed is None:
            continue
        _update_presence_flags(parsed, residual_present, prosthesis_present)

    return ImageLabels(
        image_id=image["id"],
        file_name=image["file_name"],
        residual_present=residual_present,
        prosthesis_present=prosthesis_present,
    )


def extract_all_labels(data: Dict[str, List[Dict]]) -> List[ImageLabels]:
    """Extract labels for every image entry in a COCO dataset."""

    category_name_map = build_category_name_map(data.get("categories", []))
    annotations_by_image = bucket_annotations_by_image(data.get("annotations", []))

    labels: List[ImageLabels] = []
    for image in data.get("images", []):
        image_annotations = annotations_by_image.get(image["id"], [])
        labels.append(extract_image_labels(image, image_annotations, category_name_map))
    return labels
