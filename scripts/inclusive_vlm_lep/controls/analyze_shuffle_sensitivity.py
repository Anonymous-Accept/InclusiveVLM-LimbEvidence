"""Analyze stability of predictions across shuffled option orders."""

from __future__ import annotations

import argparse
import logging
import statistics
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    compute_set_metrics,
    extract_prediction_labels,
    get_correct_option_ids,
    get_record_id,
    get_variant,
    load_predictions,
    load_subset,
    save_report,
)

logger = logging.getLogger(__name__)


def _base_record_id(record_id: str) -> str:
    return record_id.split("::shuffle", 1)[0]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def run_shuffle_sensitivity_analysis(
    predictions: Path,
    subset_jsonl: Path,
    output_json: Path,
) -> Path:
    """Compute permutation-level stability summaries."""

    items = {
        get_record_id(item): item
        for item in load_subset(subset_jsonl)
    }
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )

    for record in load_predictions(predictions):
        record_id = str(record.get("record_id") or record.get("item_id") or "")
        model_name = str(record.get("model_name") or "unknown_model")
        grouped[model_name][_base_record_id(record_id)].append(record)

    per_model: dict[str, Any] = {}
    variant = ""
    for model_name, by_item in grouped.items():
        item_rows: list[dict[str, Any]] = []
        jaccards: list[float] = []
        exact_stds: list[float] = []
        f1_stds: list[float] = []

        for base_record_id, records in sorted(by_item.items()):
            metrics_per_perm: list[dict[str, float]] = []
            selected_sets: list[set[str]] = []
            for record in sorted(records, key=lambda value: str(value.get("record_id", ""))):
                item = items.get(str(record.get("record_id") or record.get("item_id") or ""))
                if item is None:
                    continue
                variant = variant or get_variant(item)
                gold = set(get_correct_option_ids(item))
                pred = set(extract_prediction_labels(record))
                metrics_per_perm.append(compute_set_metrics(gold, pred))
                selected_sets.append(pred)

            if not metrics_per_perm:
                continue

            item_jaccards = [
                _jaccard(left, right)
                for left, right in combinations(selected_sets, 2)
            ]
            item_jaccard_mean = (
                statistics.mean(item_jaccards) if item_jaccards else 1.0
            )
            exact_values = [metric["exact_accuracy"] for metric in metrics_per_perm]
            f1_values = [metric["avg_f1"] for metric in metrics_per_perm]
            exact_std = statistics.pstdev(exact_values) if len(exact_values) > 1 else 0.0
            f1_std = statistics.pstdev(f1_values) if len(f1_values) > 1 else 0.0

            item_rows.append(
                {
                    "base_record_id": base_record_id,
                    "num_permutations": len(metrics_per_perm),
                    "exact_accuracy_mean": statistics.mean(exact_values),
                    "exact_accuracy_std": exact_std,
                    "partial_accuracy_mean": statistics.mean(
                        metric["partial_accuracy"] for metric in metrics_per_perm
                    ),
                    "avg_f1_mean": statistics.mean(f1_values),
                    "avg_f1_std": f1_std,
                    "selected_set_jaccard_mean": item_jaccard_mean,
                }
            )
            jaccards.append(item_jaccard_mean)
            exact_stds.append(exact_std)
            f1_stds.append(f1_std)

        per_model[model_name] = {
            "summary": {
                "num_items": len(item_rows),
                "selected_set_jaccard_mean": statistics.mean(jaccards) if jaccards else 0.0,
                "exact_accuracy_std_mean": statistics.mean(exact_stds) if exact_stds else 0.0,
                "avg_f1_std_mean": statistics.mean(f1_stds) if f1_stds else 0.0,
            },
            "per_item": item_rows,
        }

    save_report(
        {
            "analysis_type": "option_order_shuffle",
            "variant": variant,
            "predictions_path": str(predictions),
            "subset_path": str(subset_jsonl),
            "per_model": per_model,
        },
        output_json,
    )
    logger.info("Wrote shuffle sensitivity report to %s", output_json)
    return output_json


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    configure_logging()
    run_shuffle_sensitivity_analysis(
        args.predictions,
        args.subset_jsonl,
        args.output_json,
    )


if __name__ == "__main__":
    main()
