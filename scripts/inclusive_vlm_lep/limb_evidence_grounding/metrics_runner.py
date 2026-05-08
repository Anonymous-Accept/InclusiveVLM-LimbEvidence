"""Compute Limb-Evidence Grounding metrics from prediction results (spec_009)."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

from .eval_config import DEFAULT_METRICS_DIR, DEFAULT_QUERIES_PATH, DEFAULT_RESULTS_DIR
from .metrics import build_metrics
from ..prosthesis_match.config import (
    DEFAULT_DATA_ITEMS_PATH as DEFAULT_PROSTHESIS_ITEMS_PATH,
    DEFAULT_WORK_DIR as DEFAULT_PROSTHESIS_WORK_DIR,
)


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Compute metrics for Limb-Evidence Grounding predictions."
    )
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to queries.jsonl (optional, used for sanity checks).",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to model prediction JSONL.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for metrics JSON.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--prosthesis-results-path",
        type=Path,
        default=None,
        help="Optional Prosthesis Matching predictions JSONL for cross-benchmark analysis.",
    )
    parser.add_argument(
        "--prosthesis-items-path",
        type=Path,
        default=DEFAULT_PROSTHESIS_ITEMS_PATH,
        help="Path to Prosthesis Matching items JSONL (default: dataset items).",
    )
    parser.add_argument(
        "--correctness-strategy",
        type=str,
        default="majority",
        choices=["majority", "all", "any"],
        help="Image-level correctness aggregation strategy.",
    )
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CI (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42).",
    )
    return parser.parse_args()


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Load JSONL records into a list."""

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _infer_metrics_path(results_path: Path) -> Path:
    """Infer metrics path based on results filename."""

    if (
        results_path.name == "predictions.jsonl"
        and results_path.parent.parent.name == "runs"
    ):
        model_name = results_path.parent.name
        task_root = results_path.parent.parent.parent
        return task_root / "metrics" / f"{model_name}.metrics.json"
    return DEFAULT_METRICS_DIR / f"{results_path.stem}.metrics.json"


def _load_items_by_id(path: Path) -> Dict[str, Dict[str, Any]]:
    """Load Prosthesis Matching items into a dict keyed by item_id."""

    items: Dict[str, Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            items[item.get("item_id")] = item
    return items


def _infer_prosthesis_results_path(results_path: Path) -> Path | None:
    """Infer canonical Prosthesis Matching diversity results for the same model."""

    model_name = results_path.parent.name
    candidate = (
        DEFAULT_PROSTHESIS_WORK_DIR
        / "runs"
        / model_name
        / "compatibility_diversity"
        / "predictions.jsonl"
    )
    return candidate if candidate.exists() else None


def run_metrics(
    results_path: Path,
    output_path: Path,
    queries_path: Path | None = None,
    log_level: str = "INFO",
    compute_bootstrap: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    prosthesis_results_path: Path | None = None,
    prosthesis_items_path: Path | None = None,
    correctness_strategy: str = "majority",
) -> Dict[str, Any]:
    """Compute and persist metrics for a results file."""

    configure_logging(log_level)
    logging.info("Loading results from %s", results_path)
    results = _load_jsonl(results_path)

    queries: List[Dict[str, Any]] | None = None
    if queries_path and queries_path.exists():
        queries = _load_jsonl(queries_path)
        if queries and len(queries) != len(results):
            logging.warning(
                "Query/result count mismatch: queries=%d results=%d",
                len(queries),
                len(results),
            )

    prosthesis_predictions: List[Dict[str, Any]] | None = None
    prosthesis_items: Dict[str, Dict[str, Any]] | None = None
    resolved_prosthesis_path = prosthesis_results_path or _infer_prosthesis_results_path(
        results_path
    )
    if resolved_prosthesis_path and resolved_prosthesis_path.exists():
        logging.info(
            "Loading Prosthesis Matching results from %s", resolved_prosthesis_path
        )
        prosthesis_predictions = _load_jsonl(resolved_prosthesis_path)
        if prosthesis_items_path and prosthesis_items_path.exists():
            prosthesis_items = _load_items_by_id(prosthesis_items_path)
        else:
            logging.warning(
                "Prosthesis Matching items not found at %s; skipping cross-benchmark analysis.",
                prosthesis_items_path,
            )

    metrics = build_metrics(
        results,
        compute_bootstrap=compute_bootstrap,
        bootstrap_samples=bootstrap_samples,
        confidence_level=confidence_level,
        bootstrap_seed=bootstrap_seed,
        queries=queries,
        prosthesis_predictions=prosthesis_predictions,
        prosthesis_items=prosthesis_items,
        correctness_strategy=correctness_strategy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logging.info("Saved metrics to %s", output_path)
    return metrics


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    output_path = (
        args.output_path
        if args.output_path is not None
        else _infer_metrics_path(args.results_path)
    )
    run_metrics(
        results_path=args.results_path,
        output_path=output_path,
        queries_path=args.queries_path,
        log_level=args.log_level,
        compute_bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        prosthesis_results_path=args.prosthesis_results_path,
        prosthesis_items_path=args.prosthesis_items_path,
        correctness_strategy=args.correctness_strategy,
    )


if __name__ == "__main__":
    main()
