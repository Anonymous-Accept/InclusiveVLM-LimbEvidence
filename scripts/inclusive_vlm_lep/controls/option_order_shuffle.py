"""Generate shuffled Compatibility items for option-order sensitivity checks."""

from __future__ import annotations

import argparse
import copy
import logging
import math
import random
from pathlib import Path
from typing import Any

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    append_jsonl_line,
    get_options,
    get_record_id,
    get_variant,
    is_compatibility_variant,
    load_subset,
    save_report,
    open_jsonl_writer,
)

logger = logging.getLogger(__name__)


def _sample_distinct_permutations(
    option_ids: list[str],
    num_permutations: int,
    rng: random.Random,
) -> list[list[str]]:
    """Sample distinct non-identity permutations."""

    identity = tuple(option_ids)
    seen: set[tuple[str, ...]] = {identity}
    permutations: list[list[str]] = []
    max_unique = math.factorial(len(option_ids)) - 1
    if num_permutations > max_unique:
        raise ValueError(
            f"Requested {num_permutations} permutations for {len(option_ids)} options, "
            f"but only {max_unique} non-identity permutations exist."
        )

    attempts = 0
    max_attempts = max(32, num_permutations * 20)
    while len(permutations) < num_permutations and attempts < max_attempts:
        attempts += 1
        candidate = list(option_ids)
        rng.shuffle(candidate)
        candidate_key = tuple(candidate)
        if candidate_key in seen:
            continue
        seen.add(candidate_key)
        permutations.append(candidate)

    if len(permutations) != num_permutations:
        raise RuntimeError("Failed to sample enough distinct permutations.")
    return permutations


def run_option_order_shuffle(
    subset_jsonl: Path,
    output_subset_jsonl: Path,
    *,
    num_permutations: int = 3,
    seed: int = 0,
) -> tuple[Path, Path]:
    """Generate shuffled Compatibility items and audit metadata."""

    items = load_subset(subset_jsonl)
    rng = random.Random(seed)
    audit_entries: list[dict[str, Any]] = []
    written = 0
    skipped = 0
    audit_path = output_subset_jsonl.parent / "shuffle_audit.json"

    with open_jsonl_writer(output_subset_jsonl) as handle:
        for item in items:
            variant = get_variant(item)
            option_count = item.get("option_count")
            if option_count is None or not is_compatibility_variant(variant):
                skipped += 1
                continue

            options = get_options(item)
            if len(options) != int(option_count):
                logger.warning(
                    "Skipping %s because option_count=%s but %d options were found",
                    get_record_id(item),
                    option_count,
                    len(options),
                )
                skipped += 1
                continue

            original_order = [option["option_id"] for option in options]
            permutations = _sample_distinct_permutations(
                original_order,
                num_permutations,
                rng,
            )
            option_by_id = {option["option_id"]: option for option in options}

            for index, shuffled_ids in enumerate(permutations):
                shuffled_item = copy.deepcopy(item)
                shuffled_item["record_id"] = f"{get_record_id(item)}::shuffle{index}"
                if shuffled_item.get("item_id") is not None:
                    shuffled_item["item_id"] = shuffled_item["record_id"]
                shuffled_item["source_record_id"] = get_record_id(item)
                shuffled_item["shuffle_index"] = index
                shuffled_item["options"] = [
                    copy.deepcopy(option_by_id[option_id]) for option_id in shuffled_ids
                ]
                append_jsonl_line(handle, shuffled_item)
                written += 1

                audit_entries.append(
                    {
                        "source_record_id": get_record_id(item),
                        "shuffled_record_id": shuffled_item["record_id"],
                        "variant": variant,
                        "shuffle_index": index,
                        "seed": seed,
                        "original_option_order": original_order,
                        "shuffled_option_order": shuffled_ids,
                    }
                )

    save_report(
        {
            "analysis_type": "option_order_shuffle_audit",
            "subset_path": str(subset_jsonl),
            "output_subset_path": str(output_subset_jsonl),
            "seed": seed,
            "num_permutations": num_permutations,
            "num_written": written,
            "num_skipped": skipped,
            "entries": audit_entries,
        },
        audit_path,
    )
    logger.info("Wrote %d shuffled items to %s", written, output_subset_jsonl)
    logger.info("Wrote shuffle audit to %s", audit_path)
    return output_subset_jsonl, audit_path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-subset-jsonl", type=Path, required=True)
    parser.add_argument("--num-permutations", type=int, default=3)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    configure_logging()
    run_option_order_shuffle(
        args.subset_jsonl,
        args.output_subset_jsonl,
        num_permutations=args.num_permutations,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
