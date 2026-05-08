"""Compute parser coverage diagnostics for free-form control outputs."""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    extract_prediction_labels,
    get_gold_count,
    get_record_id,
    get_variant,
    infer_answer_space_type,
    load_predictions,
    load_subset,
    save_report,
)

logger = logging.getLogger(__name__)


def run_parse_coverage(
    predictions: Path,
    subset_jsonl: Path,
    output_json: Path,
) -> Path:
    """Compute parse coverage metrics for free-form variants."""

    items = load_subset(subset_jsonl)
    answer_space_types = {infer_answer_space_type(item) for item in items}
    if answer_space_types != {"freeform"}:
        raise ValueError(
            "parse_coverage only supports free-form subsets. "
            f"Found answer-space types: {sorted(answer_space_types)}"
        )

    item_by_id = {get_record_id(item): item for item in items}
    variant = get_variant(items[0]) if items else ""
    per_model_counts: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "n": 0,
            "parse_success": 0,
            "unresolved_tokens_total": 0,
            "hallucinated": 0,
            "omissions": 0,
        }
    )

    for record in load_predictions(predictions):
        model_name = str(record.get("model_name") or "unknown_model")
        record_id = str(record.get("record_id") or record.get("item_id") or "")
        item = item_by_id.get(record_id)
        if item is None:
            continue

        counts = per_model_counts[model_name]
        counts["n"] += 1

        predictions_list = extract_prediction_labels(record)
        unresolved_tokens = list(record.get("unresolved_tokens") or [])
        if predictions_list:
            counts["parse_success"] += 1
        if unresolved_tokens:
            counts["hallucinated"] += 1
        counts["unresolved_tokens_total"] += len(unresolved_tokens)
        if len(predictions_list) < get_gold_count(item):
            counts["omissions"] += 1

    per_model: dict[str, Any] = {}
    for model_name, counts in per_model_counts.items():
        total = counts["n"]
        per_model[model_name] = {
            "n": total,
            "parse_success_rate": counts["parse_success"] / total if total else 0.0,
            "unresolved_token_rate": counts["unresolved_tokens_total"] / total if total else 0.0,
            "hallucination_rate": counts["hallucinated"] / total if total else 0.0,
            "omission_rate": counts["omissions"] / total if total else 0.0,
        }

    save_report(
        {
            "report_type": "parse_coverage",
            "variant": variant,
            "predictions_path": str(predictions),
            "subset_path": str(subset_jsonl),
            "per_model": per_model,
        },
        output_json,
    )
    logger.info("Wrote parse coverage report to %s", output_json)
    return output_json


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    configure_logging()
    run_parse_coverage(args.predictions, args.subset_jsonl, args.output_json)


if __name__ == "__main__":
    main()
