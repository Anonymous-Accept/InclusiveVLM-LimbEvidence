"""Run the InclusiveVLM-LEP recompute/baseline/findings/report pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.core.logging_utils import configure_logging, get_logger

from ._shared import (
    DEFAULT_BASELINES_ROOT,
    DEFAULT_CONFIG_PATH,
    DEFAULT_FINDINGS_ROOT,
    DEFAULT_METRICS_ROOT,
    DEFAULT_PREDICTIONS_ROOT_BY_VARIANT,
    VARIANT_ORDER,
    discover_models,
    load_config,
    recompute_variant_metrics,
    resolve_predictions_path,
    resolve_subset_path,
    timestamp_slug,
    write_metrics_bundle,
)
from .aggregate_report import build_report
from .run_all_baselines import main as run_baselines_main
from .run_findings_analysis import main as run_findings_main

LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def _collect_models(config: dict[str, object]) -> list[str]:
    configured = list(config.get("models") or [])
    if configured:
        return [str(model) for model in configured]
    discovered = set()
    for variant in VARIANT_ORDER:
        variant_cfg = (config.get("variants") or {}).get(variant, {})
        root = Path(
            variant_cfg.get("predictions_root")
            or DEFAULT_PREDICTIONS_ROOT_BY_VARIANT[variant]
        )
        if root and str(root) != ".":
            discovered.update(discover_models(root))
    return sorted(discovered)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    config = load_config(args.config)
    metrics_root = Path(config.get("metrics_root") or DEFAULT_METRICS_ROOT)
    baselines_root = Path(config.get("baselines_root") or DEFAULT_BASELINES_ROOT)
    findings_root = Path(config.get("findings_root") or DEFAULT_FINDINGS_ROOT)
    report_output = Path(
        config.get("output_report")
        or metrics_root.parent / f"main_results_{timestamp_slug()}.md"
    )
    metrics_root.mkdir(parents=True, exist_ok=True)
    models = _collect_models(config)
    LOGGER.info("Running recompute for %d discovered/configured models", len(models))
    for model_name in models:
        for variant in VARIANT_ORDER:
            predictions_path = resolve_predictions_path(variant, model_name, config)
            if not predictions_path.exists():
                LOGGER.warning("Missing predictions for %s x %s: %s", model_name, variant, predictions_path)
                continue
            output_json = metrics_root / f"{model_name}__{variant}.metrics.json"
            if args.resume and output_json.exists():
                LOGGER.info("Skipping existing metrics: %s", output_json)
                continue
            subset_path = resolve_subset_path(variant, config)
            payload, per_item_rows, per_group_rows = recompute_variant_metrics(
                variant,
                predictions_path,
                subset_path,
                consistency_gate=None,
                policy_version=str(config.get("policy_version") or "1.1.0"),
            )
            write_metrics_bundle(output_json, payload, per_item_rows, per_group_rows)
    # Run baselines/findings/report as subprocess-free local calls to keep the wrapper thin.
    import sys

    argv_backup = list(sys.argv)
    try:
        sys.argv = [
            "run_all_baselines",
            "--output-dir",
            str(baselines_root),
            "--seed",
            str(int(config.get("seed") or 0)),
        ]
        run_baselines_main()
        sys.argv = [
            "run_findings_analysis",
            "--metrics-root",
            str(metrics_root),
            "--output-dir",
            str(findings_root),
        ]
        run_findings_main()
    finally:
        sys.argv = argv_backup
    build_report(metrics_root, baselines_root, findings_root, report_output)
    LOGGER.info("Pipeline complete. Report: %s", report_output)


if __name__ == "__main__":
    main()
