"""Recompute InclusiveVLM-LEP metrics from existing prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.core.logging_utils import configure_logging, get_logger

from ._shared import (
    FREEFORM_POLICY_VERSION,
    get_variant_definition,
    parse_bool_arg,
    recompute_variant_metrics,
    write_metrics_bundle,
)

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--subset-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument(
        "--consistency-gate",
        type=str,
        default=None,
        help="Optional true/false override. Defaults are variant-specific.",
    )
    parser.add_argument(
        "--policy-version",
        type=str,
        default=FREEFORM_POLICY_VERSION,
        help="Attribution-FreeForm parser policy version.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    variant = get_variant_definition(args.variant).name
    gate = (
        None
        if args.consistency_gate is None
        else parse_bool_arg(args.consistency_gate)
    )
    payload, per_item_rows, per_group_rows = recompute_variant_metrics(
        variant,
        args.predictions,
        args.subset_jsonl,
        consistency_gate=gate,
        policy_version=args.policy_version,
    )
    write_metrics_bundle(args.output_json, payload, per_item_rows, per_group_rows)
    LOGGER.info(
        "Recomputed %s for %s samples (%s groups) -> %s",
        variant,
        payload["num_samples"],
        payload["num_paraphrase_groups"],
        args.output_json,
    )


if __name__ == "__main__":
    main()
