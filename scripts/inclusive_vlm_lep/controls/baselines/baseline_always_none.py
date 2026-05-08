"""Always abstain / none baseline."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    append_jsonl_line,
    classify_subset,
    find_none_option_id,
    load_subset,
    make_prediction_record,
    open_jsonl_writer,
)

logger = logging.getLogger(__name__)

BASELINE_NAME = "baseline_always_none"


def run_baseline_always_none(
    subset_jsonl: Path,
    output_predictions: Path,
) -> Path:
    """Run always-none baseline."""

    items = load_subset(subset_jsonl)
    answer_space_type = classify_subset(items)

    with open_jsonl_writer(output_predictions) as handle:
        for item in items:
            if answer_space_type == "constrained":
                none_option = find_none_option_id(item)
                selected = [none_option] if none_option else []
            else:
                selected = []
            append_jsonl_line(
                handle,
                make_prediction_record(
                    item=item,
                    model_name=BASELINE_NAME,
                    seed=None,
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
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    configure_logging()
    run_baseline_always_none(args.subset_jsonl, args.output_predictions)


if __name__ == "__main__":
    main()
