"""Build Compatibility-Category subset items for Prosthesis Matching.

Design (2026-04-21): 4-option hit@1 MCQ, one option per ``coarse_part``
category (``arm_upper``, ``arm_lower``, ``leg_upper``, ``leg_lower``).
Gold is the set of coarse-part categories present on the target person
(e.g. a bilateral leg_lower amputee -> ``{leg_lower}``; a leg+arm
amputee -> ``{leg_lower, arm_upper}``). Scoring picks the option whose
category is in the gold set; multi-pick is allowed and counts as a hit
if any pick matches (see ``_shared.py::score_compatibility_category``).

This is deliberately distinct from Compatibility-Diversity, which is an
8-option set-selection task. Do not merge the builders.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .build_options_pool import OptionEntry, load_options_manifest
from .build_prosthesis_match_items import BenchmarkItem, OptionPool
from .config import (
    ALLOWED_CHOICES,
    COARSE_PARTS,
    DEFAULT_ANNO_PATH,
    DEFAULT_IMAGE_ROOT,
    DEFAULT_OPTIONS_DIR,
    DEFAULT_SEED,
    PART8_LABELS,
    PART8_TO_COARSE,
    PROJECT_ROOT,
)
from .dataset_adapter import InclusiveVLMSourceAdapter

__all__ = ["build_category_items", "CategoryItemsConfig"]

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = (
    DEFAULT_OPTIONS_DIR.parent / "items" / "compatibility_category.jsonl"
)
DEFAULT_TEMPLATE_ID = "category_select_v2_4opt"
DEFAULT_NUM_OPTIONS = 4
# Category golds intentionally match on coarse body-part only, so left/right
# laterality variants of the same anatomical segment are jointly correct.
CATEGORY_GOLD_MATCH_FIELD = "coarse_part"
QUERY_TEMPLATES = {
    "category_select_v2_4opt": (
        "Look at the person in the image. Each option shows a prosthesis from "
        "a different body-part category. Pick the single option whose body-part "
        "category best matches the person's visible limb deficiency. If more "
        "than one category applies (e.g. the person has deficiencies in "
        "multiple limb segments), pick any one that applies. Output only JSON: "
        '{"choices": ["A"]}'
    ),
    "category_select_v1": (
        "Look at the person in the image. Each option shows a prosthesis from a "
        "different limb category. Select all options whose body-part category "
        "matches the person's visible limb deficiency. There may be one or more "
        'correct answers. Output only JSON: {"choices": ["A", "C"]}'
    ),
}


def _relative_to_project(path: Path) -> str:
    """Return path relative to project root when possible."""

    if not path.is_absolute():
        return str(path)
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


@dataclass
class CategoryTarget:
    """Single target limb-deficiency instance in a person image."""

    annotation_id: int
    coarse_part: str
    part8: str
    bbox_xyxy: List[float]
    source_type: str


@dataclass
class CategoryItemsConfig:
    """Configuration for Compatibility-Category item construction."""

    seed: int = DEFAULT_SEED
    num_options: int = DEFAULT_NUM_OPTIONS
    template_id: str = DEFAULT_TEMPLATE_ID


def _collect_targets_for_image(
    adapter: InclusiveVLMSourceAdapter,
    image_id: int,
) -> Tuple[List[str], List[CategoryTarget]]:
    """Collect target part8 categories and source instances for an image.

    Args:
        adapter: Dataset adapter.
        image_id: Person image ID.

    Returns:
        Tuple of ordered unique target part8 labels and source instances.
    """
    targets: List[CategoryTarget] = []

    for residual in adapter.get_residuals_for_image(image_id):
        targets.append(
            CategoryTarget(
                annotation_id=residual.annotation_id,
                coarse_part=PART8_TO_COARSE[residual.part8],
                part8=residual.part8,
                bbox_xyxy=residual.bbox_xyxy,
                source_type="residual",
            )
        )

    for prosthesis in adapter.get_prostheses_for_image(image_id):
        targets.append(
            CategoryTarget(
                annotation_id=prosthesis.annotation_id,
                coarse_part=PART8_TO_COARSE[prosthesis.part8],
                part8=prosthesis.part8,
                bbox_xyxy=prosthesis.bbox_xyxy,
                source_type="prosthesis",
            )
        )

    target_part8_set = {target.part8 for target in targets}
    target_part8s = [part8 for part8 in PART8_LABELS if part8 in target_part8_set]
    return target_part8s, targets


def _get_excluded_option_ids(
    option_pool: OptionPool,
    source_image_id: int,
) -> Set[str]:
    """Return option IDs that must be excluded for an image."""

    return {
        entry.option_id
        for entry in option_pool.entries
        if entry.source_image_id == source_image_id
    }


def sample_coarse_part_options(
    option_pool: OptionPool,
    source_image_id: int,
    coarse_parts: List[str],
    rng: random.Random,
) -> Tuple[List[OptionEntry], List[str]]:
    """Sample one option per coarse_part category, excluding source image.

    The returned list has at most ``len(coarse_parts)`` entries; each sampled
    option's ``coarse_part`` is distinct. When a category has no usable
    option after exclusion it is recorded in ``missing`` and skipped.

    Args:
        option_pool: Option pool for sampling.
        source_image_id: Source image ID to exclude.
        coarse_parts: Coarse-part categories to sample (length = num_options).
        rng: Random number generator.

    Returns:
        Tuple of sampled options and missing coarse_part categories.
    """
    exclude_ids = _get_excluded_option_ids(option_pool, source_image_id)
    sampled_options: List[OptionEntry] = []
    missing_coarse_parts: List[str] = []

    for coarse in coarse_parts:
        candidates = [
            entry
            for entry in option_pool._by_coarse_part.get(coarse, [])
            if entry.option_id not in exclude_ids
        ]
        if not candidates:
            missing_coarse_parts.append(coarse)
            continue

        selected = rng.choice(candidates)
        sampled_options.append(selected)
        exclude_ids.add(selected.option_id)

    rng.shuffle(sampled_options)
    return sampled_options, missing_coarse_parts


def build_category_item(
    adapter: InclusiveVLMSourceAdapter,
    option_pool: OptionPool,
    image_id: int,
    item_counter: int,
    rng: random.Random,
    config: CategoryItemsConfig,
) -> Optional[BenchmarkItem]:
    """Build a single Compatibility-Category item for one person image.

    Args:
        adapter: Dataset adapter.
        option_pool: Option pool.
        image_id: Person image ID.
        item_counter: Item counter for ID generation.
        rng: Random number generator.
        config: Item-construction configuration (num_options / template_id).

    Returns:
        BenchmarkItem or None if the item cannot be built.
    """
    target_part8s, targets = _collect_targets_for_image(adapter, image_id)
    if not target_part8s:
        return None

    # Under the 4-option design we sample one option per coarse_part.
    # For other values of num_options (legacy 8-option mode), fall back to
    # PART8_LABELS sampling (one option per part8) to stay backwards-
    # compatible with callers that want laterality-specific slots.
    if config.num_options == len(COARSE_PARTS):
        categories = list(COARSE_PARTS)
        sampled_options, missing = sample_coarse_part_options(
            option_pool, image_id, categories, rng
        )
        missing_coarse_parts = missing
        missing_part8s: List[str] = []
    elif config.num_options == len(PART8_LABELS):
        # Legacy 8-option path: one option per part8 label.
        exclude_ids = _get_excluded_option_ids(option_pool, image_id)
        sampled_options = []
        missing_part8s = []
        for part8 in PART8_LABELS:
            candidates = [
                entry
                for entry in option_pool._by_part8.get(part8, [])
                if entry.option_id not in exclude_ids
            ]
            if not candidates:
                missing_part8s.append(part8)
                continue
            selected = rng.choice(candidates)
            sampled_options.append(selected)
            exclude_ids.add(selected.option_id)
        rng.shuffle(sampled_options)
        missing_coarse_parts = []
    else:
        raise ValueError(
            "Unsupported num_options="
            f"{config.num_options}; expected {len(COARSE_PARTS)} (4, coarse) "
            f"or {len(PART8_LABELS)} (8, part8)"
        )

    target_coarse_parts = sorted({PART8_TO_COARSE[part8] for part8 in target_part8s})
    sampled_coarse_parts = {entry.coarse_part for entry in sampled_options}

    # Skip items whose true target category cannot appear in the options.
    if not any(coarse in sampled_coarse_parts for coarse in target_coarse_parts):
        logger.debug(
            "Skipping image %s: no target coarse_part available in options "
            "(targets=%s, sampled=%s)",
            image_id,
            ",".join(target_coarse_parts),
            ",".join(sorted(sampled_coarse_parts)),
        )
        return None

    if len(sampled_options) < 2:
        logger.debug("Skipping image %s: not enough sampled options", image_id)
        return None

    img_path = adapter.get_image_path(image_id)
    item_id = f"compatibility_category:{item_counter:08d}"

    person_image = {
        "image_id": str(image_id),
        "path": _relative_to_project(img_path),
        "focus": {
            "mode": "full",
            "bbox_xyxy": None,
        },
    }

    query = {
        "template_id": config.template_id,
        "text": QUERY_TEMPLATES[config.template_id],
    }

    options: List[Dict[str, str]] = []
    correct_option_ids: List[str] = []
    for index, entry in enumerate(sampled_options):
        option_id = ALLOWED_CHOICES[index]
        options.append(
            {
                "option_id": option_id,
                "type": "image",
                "path": _relative_to_project(option_pool.get_option_path(entry)),
                "part8": entry.part8,
                "coarse_part": entry.coarse_part,
            }
        )
        if entry.coarse_part in target_coarse_parts:
            correct_option_ids.append(option_id)

    answer = {
        "mode": "category",
        "correct_option_ids": correct_option_ids,
        "target_part8s": target_part8s,
        "target_coarse_parts": target_coarse_parts,
    }

    meta = {
        "source": "inclusive_vlm_lep",
        "subset": "category",
        "num_options": config.num_options,
        "has_multiple_targets": len(targets) > 1,
        "target_instances": [
            {
                "annotation_id": target.annotation_id,
                "coarse_part": target.coarse_part,
                "part8": target.part8,
                "bbox_xyxy": target.bbox_xyxy,
                "source_type": target.source_type,
            }
            for target in targets
        ],
        "missing_option_coarse_parts": missing_coarse_parts,
        "missing_option_part8s": missing_part8s,
        "seed": rng.randint(0, 2**31),
    }

    return BenchmarkItem(
        item_id=item_id,
        person_image=person_image,
        query=query,
        options=options,
        answer=answer,
        meta=meta,
    )


def build_category_items(
    adapter: InclusiveVLMSourceAdapter,
    option_pool: OptionPool,
    output_path: Path,
    config: Optional[CategoryItemsConfig] = None,
) -> List[BenchmarkItem]:
    """Build Compatibility-Category subset items.

    Args:
        adapter: Dataset adapter.
        option_pool: Option pool.
        output_path: Path to save items JSONL.
        config: Item construction configuration.

    Returns:
        List of built benchmark items.
    """
    if config is None:
        config = CategoryItemsConfig()

    if config.num_options > len(ALLOWED_CHOICES):
        raise ValueError(
            f"num_options={config.num_options} exceeds available option "
            f"letters ({len(ALLOWED_CHOICES)})"
        )
    if config.template_id not in QUERY_TEMPLATES:
        raise ValueError(f"Unknown template_id={config.template_id}")

    rng = random.Random(config.seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    items: List[BenchmarkItem] = []
    skipped_images = 0

    image_ids = adapter.get_images_with_limb_deficiency()
    logger.info("Found %s images with limb deficiency", len(image_ids))

    for image_id in image_ids:
        item = build_category_item(
            adapter, option_pool, image_id, len(items), rng, config
        )
        if item is None:
            skipped_images += 1
            continue
        items.append(item)

        if len(items) % 100 == 0:
            logger.info("Built %s category items...", len(items))

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    summary = {
        "total_items": len(items),
        "skipped_images": skipped_images,
        "seed": config.seed,
        "subset": "category",
        "num_options": config.num_options,
        "template_id": config.template_id,
        "target_part8_order": PART8_LABELS,
        "coarse_part_order": COARSE_PARTS,
        "by_num_correct": {},
        "by_num_targets": {},
        "by_option_count": {},
        "by_target_part8": {},
        "by_target_coarse_part": {},
    }

    for item in items:
        num_correct = len(item.answer["correct_option_ids"])
        num_targets = len(item.answer["target_part8s"])
        option_count = len(item.options)

        summary["by_num_correct"][str(num_correct)] = (
            summary["by_num_correct"].get(str(num_correct), 0) + 1
        )
        summary["by_num_targets"][str(num_targets)] = (
            summary["by_num_targets"].get(str(num_targets), 0) + 1
        )
        summary["by_option_count"][str(option_count)] = (
            summary["by_option_count"].get(str(option_count), 0) + 1
        )

        for part8 in item.answer["target_part8s"]:
            summary["by_target_part8"][part8] = (
                summary["by_target_part8"].get(part8, 0) + 1
            )
        for coarse in item.answer["target_coarse_parts"]:
            summary["by_target_coarse_part"][coarse] = (
                summary["by_target_coarse_part"].get(coarse, 0) + 1
            )

    summary_path = output_path.parent / "items_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Built %s Compatibility-Category items", len(items))
    logger.info("Items saved to %s", output_path)
    logger.info("Summary saved to %s", summary_path)

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
        description="Build Compatibility-Category subset items."
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
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output path for items JSONL. Default: {DEFAULT_OUTPUT_PATH}",
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=DEFAULT_NUM_OPTIONS,
        choices=[len(COARSE_PARTS), len(PART8_LABELS)],
        help=(
            f"Number of MCQ options per item. {len(COARSE_PARTS)} = one per "
            f"coarse_part (hit@1 design, default). {len(PART8_LABELS)} = legacy "
            "one-per-part8 design (do not use for current Category variant)."
        ),
    )
    parser.add_argument(
        "--template-id",
        type=str,
        default=DEFAULT_TEMPLATE_ID,
        choices=sorted(QUERY_TEMPLATES.keys()),
        help=f"Query template ID. Default: {DEFAULT_TEMPLATE_ID}",
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
    logger.info("Prosthesis Matching: Build Compatibility-Category Items")
    logger.info("num_options=%s, template_id=%s", args.num_options, args.template_id)
    logger.info("=" * 60)

    adapter = InclusiveVLMSourceAdapter(
        anno_path=args.anno_path,
        image_root=args.image_root,
    )
    adapter.load()

    options_manifest = args.options_manifest
    if options_manifest is None:
        options_manifest = args.options_dir / "options_manifest.jsonl"

    logger.info("Loading options manifest from %s", options_manifest)
    entries = load_options_manifest(options_manifest)
    option_pool = OptionPool(entries, args.options_dir)
    logger.info("Loaded %s option entries", len(entries))

    config = CategoryItemsConfig(
        seed=args.seed,
        num_options=args.num_options,
        template_id=args.template_id,
    )
    items = build_category_items(adapter, option_pool, args.output_path, config)
    mean_gold_count = (
        sum(len(item.answer["correct_option_ids"]) for item in items) / len(items)
        if items
        else 0.0
    )
    sys.stdout.write(
        "Rebuilt compatibility_category.jsonl: "
        f"{len(items)} items, num_options={args.num_options}, "
        f"mean gold_count = {mean_gold_count:.2f}\n"
    )

    logger.info("=" * 60)
    logger.info("Done! Created %s Compatibility-Category items.", len(items))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
