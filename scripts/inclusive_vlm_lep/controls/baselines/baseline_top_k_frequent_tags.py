"""Fixed top-k frequent canonical tags for attribution variants."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    append_jsonl_line,
    classify_subset,
    compute_label_frequency,
    get_item_canonical_label_space,
    get_variant,
    infer_global_canonical_vocab,
    is_attribution_variant,
    load_subset,
    make_prediction_record,
    open_jsonl_writer,
    top_labels_from_counts,
)

logger = logging.getLogger(__name__)

BASELINE_NAME = "baseline_top_k_frequent_tags"


def run_baseline_top_k_frequent_tags(
    subset_jsonl: Path,
    output_predictions: Path,
    *,
    seed: int = 0,
    k: int = 2,
) -> Path:
    """Run fixed top-k canonical-label prior on attribution subsets."""

    items = load_subset(subset_jsonl)
    answer_space_type = classify_subset(items)
    if answer_space_type != "freeform":
        raise ValueError("baseline_top_k_frequent_tags only supports free-form attribution subsets.")
    variants = {get_variant(item) for item in items}
    if not all(is_attribution_variant(variant) for variant in variants):
        raise ValueError("baseline_top_k_frequent_tags only supports Attribution-family variants.")

    counts = compute_label_frequency(items, answer_space_type="freeform")
    ranked_labels = top_labels_from_counts(counts, len(counts))
    global_vocab = infer_global_canonical_vocab(items)

    with open_jsonl_writer(output_predictions) as handle:
        for item in items:
            allowed = set(get_item_canonical_label_space(item, global_vocab))
            selected = [label for label in ranked_labels if label in allowed][:k]
            append_jsonl_line(
                handle,
                make_prediction_record(
                    item=item,
                    model_name=BASELINE_NAME,
                    seed=seed,
                    selected_labels=selected,
                ),
            )

    logger.info(
        "Wrote %d predictions to %s using fixed k=%d",
        len(items),
        output_predictions,
        k,
    )
    return output_predictions


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--k", type=int, default=2)
    args = parser.parse_args()

    configure_logging()
    run_baseline_top_k_frequent_tags(
        args.subset_jsonl,
        args.output_predictions,
        seed=args.seed,
        k=args.k,
    )


if __name__ == "__main__":
    main()
