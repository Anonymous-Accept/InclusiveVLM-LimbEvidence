"""Build Prosthesis Matching benchmark items."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .build_options_pool import OptionEntry, load_options_manifest
from .config import (
    ALLOWED_CHOICES,
    COARSE_PARTS,
    DEFAULT_ANNO_PATH,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_ITEMS_PATH,
    DEFAULT_K_OPTIONS,
    DEFAULT_OPTIONS_DIR,
    DEFAULT_SEED,
    HARD_NEGATIVE_RATIO,
    PART8_TO_COARSE,
    PROJECT_ROOT,
    get_cross_limb_parts,
    get_hard_negative_parts,
)
from .dataset_adapter import (
    InclusiveVLMSourceAdapter,
    ProsthesisInstance,
    ResidualLimbInstance,
)

__all__ = ["build_prosthesis_match_items", "BenchmarkItem", "ItemsConfig"]

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
class TargetInstance:
    """Target instance (residual limb or prosthesis) in person image."""

    instance_id: str
    annotation_id: int
    coarse_part: str
    part8: str
    bbox_xyxy: List[float]
    source_type: str  # "residual" or "prosthesis"


@dataclass
class OptionItem:
    """Single option in benchmark item."""

    option_id: str
    type: str  # "image"
    path: str
    part8: str
    coarse_part: str


@dataclass
class BenchmarkItem:
    """Single Prosthesis Matching benchmark item."""

    item_id: str
    person_image: Dict[str, Any]
    query: Dict[str, str]
    options: List[Dict[str, str]]
    answer: Dict[str, Any]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ItemsConfig:
    """Configuration for benchmark item construction."""

    k_options: int = DEFAULT_K_OPTIONS
    mode: str = (
        "instance_split"  # "instance_split" or "multi_answer" or "variable_answer"
    )
    hard_negative_ratio: float = HARD_NEGATIVE_RATIO
    seed: int = DEFAULT_SEED
    include_prosthesis_targets: bool = True  # Include images with prosthesis as targets
    include_residual_targets: bool = (
        True  # Include images with residual limbs as targets
    )
    # Variable answer mode settings
    min_correct: int = 0  # Minimum number of correct answers (0-8)
    max_correct: int = 8  # Maximum number of correct answers (0-8, 8 = all correct)
    zero_correct_ratio: float = 0.15  # Ratio of items with zero correct answers


# Query templates (v2: support variable number of correct answers)
QUERY_TEMPLATES = {
    "pos_select_v1": (
        "Select all prosthesis options that match the limb deficiency "
        "visible in the image. Use only visible evidence. "
        "There may be zero, one, or multiple matching options. "
        'Reply with JSON: {"choices": ["A", "B"]} or {"choices": []} if none match.'
    ),
    "pos_select_v2": (
        "Look at the person in the image. Which prosthesis options are "
        "consistent with the benchmark-defined visible evidence? "
        "Select all that match, or return empty list if none. "
        'Output only JSON: {"choices": ["A"]}'
    ),
    "pos_select_v3": (
        "Based on the visible residual limb or prosthesis in the image, "
        "select all matching prosthesis types from the options. "
        "Return all matches, or empty list if none. "
        'Respond with JSON: {"choices": ["A", "C"]}'
    ),
}

DEFAULT_TEMPLATE_ID = "pos_select_v1"


class OptionPool:
    """Manages the prosthesis option pool for sampling."""

    def __init__(self, entries: List[OptionEntry], options_dir: Path) -> None:
        """Initialize option pool.

        Args:
            entries: List of OptionEntry from manifest.
            options_dir: Directory containing option images.
        """
        self.entries = entries
        self.options_dir = Path(options_dir)

        # Index by coarse part
        self._by_coarse_part: Dict[str, List[OptionEntry]] = {}
        for entry in entries:
            coarse = entry.coarse_part
            if coarse not in self._by_coarse_part:
                self._by_coarse_part[coarse] = []
            self._by_coarse_part[coarse].append(entry)

        # Index by part8
        self._by_part8: Dict[str, List[OptionEntry]] = {}
        for entry in entries:
            part8 = entry.part8
            if part8 not in self._by_part8:
                self._by_part8[part8] = []
            self._by_part8[part8].append(entry)

    def get_option_path(self, entry: OptionEntry) -> Path:
        """Get full path to option image."""
        return self.options_dir / entry.file_name

    def sample_positives(
        self,
        target_coarse: str,
        count: int,
        exclude_ids: Optional[Set[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[OptionEntry]:
        """Sample positive options matching target coarse part.

        Args:
            target_coarse: Target coarse part.
            count: Number of options to sample.
            exclude_ids: Option IDs to exclude.
            rng: Random number generator.

        Returns:
            List of sampled OptionEntry objects.
        """
        if rng is None:
            rng = random.Random()

        candidates = self._by_coarse_part.get(target_coarse, [])
        if exclude_ids:
            candidates = [e for e in candidates if e.option_id not in exclude_ids]

        if len(candidates) <= count:
            return candidates.copy()

        return rng.sample(candidates, count)

    def sample_hard_negatives(
        self,
        target_coarse: str,
        count: int,
        exclude_ids: Optional[Set[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[OptionEntry]:
        """Sample hard negative options (same limb, different segment).

        Args:
            target_coarse: Target coarse part.
            count: Number of options to sample.
            exclude_ids: Option IDs to exclude.
            rng: Random number generator.

        Returns:
            List of sampled OptionEntry objects.
        """
        if rng is None:
            rng = random.Random()

        hard_neg_parts = get_hard_negative_parts(target_coarse)
        candidates = []
        for part in hard_neg_parts:
            candidates.extend(self._by_coarse_part.get(part, []))

        if exclude_ids:
            candidates = [e for e in candidates if e.option_id not in exclude_ids]

        if len(candidates) <= count:
            return candidates.copy()

        return rng.sample(candidates, count)

    def sample_cross_limb_negatives(
        self,
        target_coarse: str,
        count: int,
        exclude_ids: Optional[Set[str]] = None,
        rng: Optional[random.Random] = None,
    ) -> List[OptionEntry]:
        """Sample cross-limb negative options (arm vs leg).

        Args:
            target_coarse: Target coarse part.
            count: Number of options to sample.
            exclude_ids: Option IDs to exclude.
            rng: Random number generator.

        Returns:
            List of sampled OptionEntry objects.
        """
        if rng is None:
            rng = random.Random()

        cross_parts = get_cross_limb_parts(target_coarse)
        candidates = []
        for part in cross_parts:
            candidates.extend(self._by_coarse_part.get(part, []))

        if exclude_ids:
            candidates = [e for e in candidates if e.option_id not in exclude_ids]

        if len(candidates) <= count:
            return candidates.copy()

        return rng.sample(candidates, count)


def extract_targets_from_image(
    adapter: InclusiveVLMSourceAdapter,
    image_id: int,
    config: ItemsConfig,
) -> List[TargetInstance]:
    """Extract target instances (residual limbs and/or prostheses) from image.

    Args:
        adapter: Dataset adapter.
        image_id: Image ID.
        config: Items configuration.

    Returns:
        List of TargetInstance objects.
    """
    targets = []
    target_counter = 0

    # Extract residual limb targets
    if config.include_residual_targets:
        residuals = adapter.get_residuals_for_image(image_id)
        for res in residuals:
            target = TargetInstance(
                instance_id=f"t{target_counter}",
                annotation_id=res.annotation_id,
                coarse_part=PART8_TO_COARSE[res.part8],
                part8=res.part8,
                bbox_xyxy=res.bbox_xyxy,
                source_type="residual",
            )
            targets.append(target)
            target_counter += 1

    # Extract prosthesis targets (the prosthesis itself indicates the matching part)
    if config.include_prosthesis_targets:
        prostheses = adapter.get_prostheses_for_image(image_id)
        for pros in prostheses:
            target = TargetInstance(
                instance_id=f"t{target_counter}",
                annotation_id=pros.annotation_id,
                coarse_part=PART8_TO_COARSE[pros.part8],
                part8=pros.part8,
                bbox_xyxy=pros.bbox_xyxy,
                source_type="prosthesis",
            )
            targets.append(target)
            target_counter += 1

    return targets


def sample_options_for_target(
    option_pool: OptionPool,
    target: TargetInstance,
    config: ItemsConfig,
    exclude_source_image: int,
    rng: random.Random,
) -> Tuple[List[OptionEntry], List[str]]:
    """Sample K options for a target, including positives and negatives.

    Args:
        option_pool: Option pool for sampling.
        target: Target instance.
        config: Items configuration.
        exclude_source_image: Source image ID to exclude from options.
        rng: Random number generator.

    Returns:
        Tuple of (list of OptionEntry, list of correct option IDs).
    """
    k = config.k_options
    exclude_ids: Set[str] = set()

    # Exclude options from the same source image
    for entry in option_pool.entries:
        if entry.source_image_id == exclude_source_image:
            exclude_ids.add(entry.option_id)

    # Sample at least 1 positive
    num_positives = max(1, k // 4)
    positives = option_pool.sample_positives(
        target.coarse_part, num_positives, exclude_ids, rng
    )
    exclude_ids.update(e.option_id for e in positives)

    # Sample hard negatives
    num_hard_neg = int((k - len(positives)) * config.hard_negative_ratio)
    hard_negatives = option_pool.sample_hard_negatives(
        target.coarse_part, num_hard_neg, exclude_ids, rng
    )
    exclude_ids.update(e.option_id for e in hard_negatives)

    # Sample cross-limb negatives for remaining slots
    remaining = k - len(positives) - len(hard_negatives)
    cross_negatives = option_pool.sample_cross_limb_negatives(
        target.coarse_part, remaining, exclude_ids, rng
    )

    # Combine and shuffle
    all_options = positives + hard_negatives + cross_negatives
    rng.shuffle(all_options)

    # Assign option IDs (A, B, C, ...)
    correct_ids = []
    for i, entry in enumerate(all_options):
        if entry.coarse_part == target.coarse_part:
            correct_ids.append(ALLOWED_CHOICES[i])

    return all_options, correct_ids


def sample_options_variable_correct(
    option_pool: OptionPool,
    target: TargetInstance,
    config: ItemsConfig,
    exclude_source_image: int,
    rng: random.Random,
    num_correct: int,
) -> Tuple[List[OptionEntry], List[str]]:
    """Sample K options with a specific number of correct answers (0-6).

    Args:
        option_pool: Option pool for sampling.
        target: Target instance.
        config: Items configuration.
        exclude_source_image: Source image ID to exclude from options.
        rng: Random number generator.
        num_correct: Exact number of correct options to include (0-6).

    Returns:
        Tuple of (list of OptionEntry, list of correct option IDs).
    """
    k = config.k_options
    exclude_ids: Set[str] = set()

    # Exclude options from the same source image
    for entry in option_pool.entries:
        if entry.source_image_id == exclude_source_image:
            exclude_ids.add(entry.option_id)

    # Sample exact number of positives (can be 0)
    positives = []
    if num_correct > 0:
        positives = option_pool.sample_positives(
            target.coarse_part, num_correct, exclude_ids, rng
        )
        exclude_ids.update(e.option_id for e in positives)

    # Fill remaining slots with negatives
    remaining = k - len(positives)

    # Split between hard and cross-limb negatives
    num_hard_neg = int(remaining * config.hard_negative_ratio)
    hard_negatives = option_pool.sample_hard_negatives(
        target.coarse_part, num_hard_neg, exclude_ids, rng
    )
    exclude_ids.update(e.option_id for e in hard_negatives)

    # Cross-limb negatives for remaining
    num_cross = remaining - len(hard_negatives)
    cross_negatives = option_pool.sample_cross_limb_negatives(
        target.coarse_part, num_cross, exclude_ids, rng
    )

    # Combine and shuffle
    all_options = positives + hard_negatives + cross_negatives
    all_options = all_options[:k]  # Ensure we don't exceed k
    rng.shuffle(all_options)

    # Determine correct option IDs
    correct_ids = []
    for i, entry in enumerate(all_options):
        if entry.coarse_part == target.coarse_part:
            correct_ids.append(ALLOWED_CHOICES[i])

    return all_options, correct_ids


def build_item_instance_split(
    adapter: InclusiveVLMSourceAdapter,
    option_pool: OptionPool,
    image_id: int,
    target: TargetInstance,
    item_counter: int,
    config: ItemsConfig,
    rng: random.Random,
) -> Optional[BenchmarkItem]:
    """Build a single benchmark item for instance-split mode.

    Args:
        adapter: Dataset adapter.
        option_pool: Option pool.
        image_id: Person image ID.
        target: Target instance.
        item_counter: Item counter for ID generation.
        config: Items configuration.
        rng: Random number generator.

    Returns:
        BenchmarkItem or None if cannot build.
    """
    # Sample options
    options, correct_ids = sample_options_for_target(
        option_pool, target, config, image_id, rng
    )

    if len(options) < 2:
        logger.debug(
            f"Not enough options for image {image_id}, target {target.instance_id}"
        )
        return None

    # Note: correct_ids can be empty (zero correct answers allowed)
    # This is a valid case for variable_answer mode

    # Get image info
    img_info = adapter.get_image_info(image_id)
    img_path = adapter.get_image_path(image_id)

    # Build item
    item_id = f"prosthesis_match:{item_counter:08d}"

    person_image = {
        "image_id": str(image_id),
        "path": _relative_to_project(img_path),
        "focus": {
            "mode": "crop",
            "bbox_xyxy": target.bbox_xyxy,
        },
    }

    query = {
        "template_id": DEFAULT_TEMPLATE_ID,
        "text": QUERY_TEMPLATES[DEFAULT_TEMPLATE_ID],
    }

    options_list = []
    for i, entry in enumerate(options):
        opt_dict = {
            "option_id": ALLOWED_CHOICES[i],
            "type": "image",
            "path": _relative_to_project(option_pool.get_option_path(entry)),
            "part8": entry.part8,
            "coarse_part": entry.coarse_part,
        }
        options_list.append(opt_dict)

    answer = {
        "mode": "single",
        "correct_option_ids": correct_ids,
        "target_coarse_parts": [target.coarse_part],
    }

    meta = {
        "source": "inclusive_vlm_lep",
        "has_multiple_targets": False,
        "target_instances": [
            {
                "instance_id": target.instance_id,
                "coarse_part": target.coarse_part,
                "part8": target.part8,
                "bbox_xyxy": target.bbox_xyxy,
                "source_type": target.source_type,
            }
        ],
        "seed": rng.randint(0, 2**31),
    }

    return BenchmarkItem(
        item_id=item_id,
        person_image=person_image,
        query=query,
        options=options_list,
        answer=answer,
        meta=meta,
    )


def build_item_multi_answer(
    adapter: InclusiveVLMSourceAdapter,
    option_pool: OptionPool,
    image_id: int,
    targets: List[TargetInstance],
    item_counter: int,
    config: ItemsConfig,
    rng: random.Random,
) -> Optional[BenchmarkItem]:
    """Build a single benchmark item for multi-answer mode.

    Args:
        adapter: Dataset adapter.
        option_pool: Option pool.
        image_id: Person image ID.
        targets: List of target instances.
        item_counter: Item counter for ID generation.
        config: Items configuration.
        rng: Random number generator.

    Returns:
        BenchmarkItem or None if cannot build.
    """
    if not targets:
        return None

    # Get unique coarse parts
    target_coarse_parts = list(set(t.coarse_part for t in targets))

    # Sample options ensuring we have positives for each target part
    k = config.k_options
    exclude_ids: Set[str] = set()

    # Exclude options from same source image
    for entry in option_pool.entries:
        if entry.source_image_id == image_id:
            exclude_ids.add(entry.option_id)

    # Sample positives for each target part
    all_positives = []
    for coarse_part in target_coarse_parts:
        positives = option_pool.sample_positives(coarse_part, 1, exclude_ids, rng)
        all_positives.extend(positives)
        exclude_ids.update(e.option_id for e in positives)

    # Fill remaining with negatives
    remaining = k - len(all_positives)

    # Split between hard and cross-limb negatives
    num_hard = int(remaining * config.hard_negative_ratio)
    num_cross = remaining - num_hard

    hard_negatives = []
    for coarse_part in target_coarse_parts:
        negs = option_pool.sample_hard_negatives(
            coarse_part, num_hard // len(target_coarse_parts), exclude_ids, rng
        )
        hard_negatives.extend(negs)
        exclude_ids.update(e.option_id for e in negs)

    cross_negatives = option_pool.sample_cross_limb_negatives(
        target_coarse_parts[0], num_cross, exclude_ids, rng
    )

    # Combine and shuffle
    all_options = all_positives + hard_negatives + cross_negatives
    all_options = all_options[:k]  # Ensure we don't exceed k
    rng.shuffle(all_options)

    if len(all_options) < 2:
        return None

    # Determine correct option IDs
    correct_ids = []
    for i, entry in enumerate(all_options):
        if entry.coarse_part in target_coarse_parts:
            correct_ids.append(ALLOWED_CHOICES[i])

    if not correct_ids:
        # Zero correct answers is valid in variable_answer mode
        pass

    # Get image info
    img_info = adapter.get_image_info(image_id)
    img_path = adapter.get_image_path(image_id)

    # Build item
    item_id = f"prosthesis_match:{item_counter:08d}"

    person_image = {
        "image_id": str(image_id),
        "path": _relative_to_project(img_path),
        "focus": {
            "mode": "full",
            "bbox_xyxy": None,
        },
    }

    query = {
        "template_id": DEFAULT_TEMPLATE_ID,
        "text": QUERY_TEMPLATES[DEFAULT_TEMPLATE_ID],
    }

    options_list = []
    for i, entry in enumerate(all_options):
        opt_dict = {
            "option_id": ALLOWED_CHOICES[i],
            "type": "image",
            "path": _relative_to_project(option_pool.get_option_path(entry)),
            "part8": entry.part8,
            "coarse_part": entry.coarse_part,
        }
        options_list.append(opt_dict)

    answer = {
        "mode": "set",
        "correct_option_ids": correct_ids,
        "target_coarse_parts": target_coarse_parts,
    }

    meta = {
        "source": "inclusive_vlm_lep",
        "has_multiple_targets": len(targets) > 1,
        "target_instances": [
            {
                "instance_id": t.instance_id,
                "coarse_part": t.coarse_part,
                "part8": t.part8,
                "bbox_xyxy": t.bbox_xyxy,
                "source_type": t.source_type,
            }
            for t in targets
        ],
        "seed": rng.randint(0, 2**31),
    }

    return BenchmarkItem(
        item_id=item_id,
        person_image=person_image,
        query=query,
        options=options_list,
        answer=answer,
        meta=meta,
    )


def build_item_variable_answer(
    adapter: InclusiveVLMSourceAdapter,
    option_pool: OptionPool,
    image_id: int,
    target: TargetInstance,
    item_counter: int,
    config: ItemsConfig,
    rng: random.Random,
    num_correct: int,
) -> Optional[BenchmarkItem]:
    """Build a single benchmark item with variable number of correct answers.

    Args:
        adapter: Dataset adapter.
        option_pool: Option pool.
        image_id: Person image ID.
        target: Target instance.
        item_counter: Item counter for ID generation.
        config: Items configuration.
        rng: Random number generator.
        num_correct: Number of correct options (0-6).

    Returns:
        BenchmarkItem or None if cannot build.
    """
    # Sample options with specified number of correct answers
    options, correct_ids = sample_options_variable_correct(
        option_pool, target, config, image_id, rng, num_correct
    )

    if len(options) < 2:
        logger.debug(
            f"Not enough options for image {image_id}, target {target.instance_id}"
        )
        return None

    # Get image info
    img_info = adapter.get_image_info(image_id)
    img_path = adapter.get_image_path(image_id)

    # Build item
    item_id = f"prosthesis_match:{item_counter:08d}"

    person_image = {
        "image_id": str(image_id),
        "path": _relative_to_project(img_path),
        "focus": {
            "mode": "crop",
            "bbox_xyxy": target.bbox_xyxy,
        },
    }

    query = {
        "template_id": DEFAULT_TEMPLATE_ID,
        "text": QUERY_TEMPLATES[DEFAULT_TEMPLATE_ID],
    }

    options_list = []
    for i, entry in enumerate(options):
        opt_dict = {
            "option_id": ALLOWED_CHOICES[i],
            "type": "image",
            "path": _relative_to_project(option_pool.get_option_path(entry)),
            "part8": entry.part8,
            "coarse_part": entry.coarse_part,
        }
        options_list.append(opt_dict)

    # Determine answer mode based on number of correct answers
    if num_correct == 0:
        answer_mode = "none"
    elif num_correct == 1:
        answer_mode = "single"
    else:
        answer_mode = "multiple"

    answer = {
        "mode": answer_mode,
        "correct_option_ids": correct_ids,
        "target_coarse_parts": [target.coarse_part] if num_correct > 0 else [],
        "num_correct": num_correct,
    }

    meta = {
        "source": "inclusive_vlm_lep",
        "has_multiple_targets": False,
        "target_instances": [
            {
                "instance_id": target.instance_id,
                "coarse_part": target.coarse_part,
                "part8": target.part8,
                "bbox_xyxy": target.bbox_xyxy,
                "source_type": target.source_type,
            }
        ],
        "seed": rng.randint(0, 2**31),
    }

    return BenchmarkItem(
        item_id=item_id,
        person_image=person_image,
        query=query,
        options=options_list,
        answer=answer,
        meta=meta,
    )


def build_prosthesis_match_items(
    adapter: InclusiveVLMSourceAdapter,
    option_pool: OptionPool,
    output_path: Path,
    config: Optional[ItemsConfig] = None,
) -> List[BenchmarkItem]:
    """Build Prosthesis Matching benchmark items.

    Args:
        adapter: Dataset adapter.
        option_pool: Option pool.
        output_path: Path to save items JSONL.
        config: Items configuration.

    Returns:
        List of BenchmarkItem objects.
    """
    if config is None:
        config = ItemsConfig()

    rng = random.Random(config.seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items: List[BenchmarkItem] = []
    item_counter = 0

    # Get images with limb deficiency
    image_ids = adapter.get_images_with_limb_deficiency()
    logger.info(f"Found {len(image_ids)} images with limb deficiency")

    for image_id in image_ids:
        # Extract targets from image
        targets = extract_targets_from_image(adapter, image_id, config)

        if not targets:
            continue

        if config.mode == "instance_split":
            # Create one item per target instance
            for target in targets:
                item = build_item_instance_split(
                    adapter, option_pool, image_id, target, item_counter, config, rng
                )
                if item:
                    items.append(item)
                    item_counter += 1

        elif config.mode == "multi_answer":
            # Create one item per image
            item = build_item_multi_answer(
                adapter, option_pool, image_id, targets, item_counter, config, rng
            )
            if item:
                items.append(item)
                item_counter += 1

        elif config.mode == "variable_answer":
            # Create items with variable number of correct answers (0-6)
            for target in targets:
                # Decide number of correct answers
                if rng.random() < config.zero_correct_ratio:
                    num_correct = 0
                else:
                    num_correct = rng.randint(
                        max(1, config.min_correct), config.max_correct
                    )

                item = build_item_variable_answer(
                    adapter,
                    option_pool,
                    image_id,
                    target,
                    item_counter,
                    config,
                    rng,
                    num_correct,
                )
                if item:
                    items.append(item)
                    item_counter += 1

        if item_counter % 100 == 0 and item_counter > 0:
            logger.info(f"Built {item_counter} items...")

    # Save items
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    # Save summary
    summary = {
        "total_items": len(items),
        "mode": config.mode,
        "k_options": config.k_options,
        "seed": config.seed,
        "by_coarse_part": {},
        "by_num_correct": {},
    }

    for item in items:
        for part in item.answer["target_coarse_parts"]:
            summary["by_coarse_part"][part] = summary["by_coarse_part"].get(part, 0) + 1

        # Track distribution of correct answer counts
        num_correct = len(item.answer["correct_option_ids"])
        summary["by_num_correct"][str(num_correct)] = (
            summary["by_num_correct"].get(str(num_correct), 0) + 1
        )

    summary_path = output_path.parent / "items_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"Built {len(items)} benchmark items")
    logger.info(f"Items saved to {output_path}")
    logger.info(f"Summary saved to {summary_path}")

    return items


def load_benchmark_items(items_path: Path) -> List[BenchmarkItem]:
    """Load benchmark items from JSONL file.

    Args:
        items_path: Path to items JSONL file.

    Returns:
        List of BenchmarkItem objects.
    """
    items = []
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                items.append(BenchmarkItem(**data))
    return items


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
        description="Build Prosthesis Matching benchmark items."
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
        "--options-dir",
        type=Path,
        default=DEFAULT_OPTIONS_DIR,
        help=f"Directory containing option images. Default: {DEFAULT_OPTIONS_DIR}",
    )
    parser.add_argument(
        "--options-manifest",
        type=Path,
        default=None,
        help="Path to options manifest. Default: <options-dir>/options_manifest.jsonl",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_ITEMS_PATH,
        help=f"Output path for items JSONL. Default: {DEFAULT_ITEMS_PATH}",
    )
    parser.add_argument(
        "--k-options",
        type=int,
        default=DEFAULT_K_OPTIONS,
        help=f"Number of options per item. Default: {DEFAULT_K_OPTIONS}",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="instance_split",
        choices=["instance_split", "multi_answer", "variable_answer"],
        help="Benchmark mode. Default: instance_split",
    )
    parser.add_argument(
        "--hard-negative-ratio",
        type=float,
        default=HARD_NEGATIVE_RATIO,
        help=f"Ratio of hard negatives. Default: {HARD_NEGATIVE_RATIO}",
    )
    parser.add_argument(
        "--min-correct",
        type=int,
        default=0,
        help="Minimum correct answers (variable_answer mode). Default: 0",
    )
    parser.add_argument(
        "--max-correct",
        type=int,
        default=8,
        help="Maximum correct answers (variable_answer mode, 8=all). Default: 8",
    )
    parser.add_argument(
        "--zero-correct-ratio",
        type=float,
        default=0.15,
        help="Ratio of items with zero correct answers. Default: 0.15",
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
    logger.info("Prosthesis Matching: Build Benchmark Items")
    logger.info("=" * 60)

    # Initialize adapter
    adapter = InclusiveVLMSourceAdapter(
        anno_path=args.anno_path,
        image_root=args.image_root,
    )
    adapter.load()

    # Load option pool
    options_manifest = args.options_manifest
    if options_manifest is None:
        options_manifest = args.options_dir / "options_manifest.jsonl"

    logger.info(f"Loading options manifest from {options_manifest}")
    entries = load_options_manifest(options_manifest)
    option_pool = OptionPool(entries, args.options_dir)
    logger.info(f"Loaded {len(entries)} option entries")

    # Create config
    config = ItemsConfig(
        k_options=args.k_options,
        mode=args.mode,
        hard_negative_ratio=args.hard_negative_ratio,
        seed=args.seed,
        min_correct=args.min_correct,
        max_correct=args.max_correct,
        zero_correct_ratio=args.zero_correct_ratio,
    )

    # Build items
    items = build_prosthesis_match_items(adapter, option_pool, args.output_path, config)

    logger.info("=" * 60)
    logger.info(f"Done! Created {len(items)} benchmark items.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
