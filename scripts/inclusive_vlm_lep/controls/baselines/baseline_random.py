"""Uniform random baseline over the valid answer space."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    append_jsonl_line,
    classify_subset,
    get_item_canonical_label_space,
    get_option_ids,
    infer_global_canonical_vocab,
    load_subset,
    make_prediction_record,
    open_jsonl_writer,
    sample_uniform_subset,
)

logger = logging.getLogger(__name__)

BASELINE_NAME = "baseline_random"


def run_baseline_random(
    subset_jsonl: Path,
    output_predictions: Path,
    *,
    seed: int = 0,
) -> Path:
    """Run the random baseline on one subset."""

    items = load_subset(subset_jsonl)
    answer_space_type = classify_subset(items)
    rng = random.Random(seed)
    global_vocab = infer_global_canonical_vocab(items)

    with open_jsonl_writer(output_predictions) as handle:
        for item in items:
            if answer_space_type == "constrained":
                selected = sample_uniform_subset(
                    get_option_ids(item),
                    rng,
                    allow_empty=True,
                )
            else:
                selected = sample_uniform_subset(
                    get_item_canonical_label_space(item, global_vocab),
                    rng,
                    allow_empty=False,
                )
            append_jsonl_line(
                handle,
                make_prediction_record(
                    item=item,
                    model_name=BASELINE_NAME,
                    seed=seed,
                    selected_labels=selected,
                ),
            )

    logger.info("Wrote %d predictions to %s", len(items), output_predictions)
    return output_predictions


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-predictions", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    configure_logging()
    run_baseline_random(
        args.subset_jsonl,
        args.output_predictions,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
