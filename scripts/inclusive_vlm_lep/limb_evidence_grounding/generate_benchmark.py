"""Build Limb-Evidence Grounding query JSONL (spec_008)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import config
from .label_extraction import ImageLabels, extract_all_labels, load_annotations
from .prompts import load_presence_prompts, load_segment_prompts
from .qa_generation import build_queries


def configure_logging(verbose: bool = False) -> None:
    """Set up console logging."""

    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Generate Limb-Evidence Grounding queries (spec_008)."
    )
    parser.add_argument(
        "--anno-path",
        type=Path,
        default=config.DEFAULT_ANNO_PATH,
        help="Path to COCO annotations JSON.",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=config.DEFAULT_IMAGE_ROOT,
        help="Root directory for images.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=config.DEFAULT_WORK_DIR,
        help="Working directory for benchmark outputs.",
    )
    parser.add_argument(
        "--presence-prompts-path",
        type=Path,
        default=config.DEFAULT_PRESENCE_PROMPTS_PATH,
        help="Path to presence_paraphrases.json.",
    )
    parser.add_argument(
        "--segment-prompts-path",
        type=Path,
        default=config.DEFAULT_SEGMENT_PROMPTS_PATH,
        help="Path to segment_paraphrases.json.",
    )
    parser.add_argument(
        "--num-presence-prompts",
        type=int,
        default=config.DEFAULT_NUM_PRESENCE_PROMPTS,
        help="Number of presence prompts to sample per image.",
    )
    parser.add_argument(
        "--num-segment-prompts",
        type=int,
        default=config.DEFAULT_NUM_SEGMENT_PROMPTS,
        help="Number of segment prompts to sample per image.",
    )
    parser.add_argument(
        "--include-none-class",
        action="store_true",
        default=config.DEFAULT_INCLUDE_NONE_CLASS,
        help="Include images with presence_label == none in Task 1.",
    )
    parser.add_argument(
        "--include-empty-segment-images",
        action="store_true",
        default=config.DEFAULT_INCLUDE_EMPTY_SEGMENT_IMAGES,
        help="Include Task 2 entries for images without any residual/prosthesis labels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.DEFAULT_SEED,
        help="Random seed for prompt sampling.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging output."
    )
    return parser.parse_args()


def _set_random_seed(seed: int) -> None:
    """Seed Python and NumPy RNGs."""

    random.seed(seed)
    np.random.seed(seed)


def _log_summary(
    images: List[ImageLabels],
    queries: List[Dict[str, Any]],
) -> None:
    """Log summary statistics about generated queries."""

    presence_counter = Counter(img.presence_label for img in images)
    presence_queries = sum(1 for q in queries if q["task_name"] == "recognition")
    segment_queries = sum(1 for q in queries if q["task_name"] == "attribution")

    logging.info("Processed %d images", len(images))
    for label, count in sorted(presence_counter.items()):
        logging.info("presence_label=%s: %d", label, count)
    logging.info("Generated %d presence queries", presence_queries)
    logging.info("Generated %d segment queries", segment_queries)


def write_jsonl(records: List[Dict[str, Any]], output_path: Path) -> None:
    """Write records to JSONL file."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    logging.info("Saved %d records to %s", len(records), output_path)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.verbose)
    _set_random_seed(args.seed)

    logging.info("Loading annotations from %s", args.anno_path)
    annotations = load_annotations(args.anno_path)
    images = extract_all_labels(annotations)

    logging.info("Loading prompts")
    presence_prompts = load_presence_prompts(args.presence_prompts_path)
    segment_prompts = load_segment_prompts(args.segment_prompts_path)

    output_path = args.work_dir / "queries.jsonl"
    logging.info("Generating queries to %s", output_path)
    queries = build_queries(
        images=images,
        presence_prompts=presence_prompts,
        segment_prompts=segment_prompts,
        num_presence_prompts=args.num_presence_prompts,
        num_segment_prompts=args.num_segment_prompts,
        image_root=args.image_root,
        include_none_class=args.include_none_class,
        keep_empty_segments=args.include_empty_segment_images,
    )

    write_jsonl(queries, output_path)
    _log_summary(images, queries)


if __name__ == "__main__":
    main()
