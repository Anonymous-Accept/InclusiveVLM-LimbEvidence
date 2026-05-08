"""Evaluate Prosthesis Matching predictions."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from .config import (
    ALLOWED_CHOICES,
    COARSE_PARTS,
    DEFAULT_DATA_ITEMS_PATH,
    DEFAULT_WORK_DIR,
)
from scripts.core.evaluation import bootstrap_confidence_interval

__all__ = ["evaluate_prosthesis_match", "ProsthesisMetrics"]

logger = logging.getLogger(__name__)


def _infer_output_path(predictions_path: Path) -> Path:
    """Infer the canonical metrics path from a predictions file."""

    if (
        predictions_path.name == "predictions.jsonl"
        and predictions_path.parent.parent.name == "runs"
    ):
        task_root = predictions_path.parent.parent.parent
        model_name = predictions_path.parent.name
        return task_root / "metrics" / f"{model_name}.metrics.json"
    return predictions_path.parent / "metrics.json"


@dataclass
class ClassMetrics:
    """Metrics for a single class."""

    n: int = 0
    correct: int = 0
    accuracy: float = 0.0


@dataclass
class ProsthesisMetrics:
    """Prosthesis Matching evaluation metrics."""

    overall: Dict[str, Any] = field(default_factory=dict)
    by_coarse_part: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    confusion: Dict[str, Dict[str, int]] = field(default_factory=dict)
    parse_errors: Dict[str, Any] = field(default_factory=dict)
    consistency: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def _consistency_score(values: List[frozenset]) -> float:
    """Return smoothed consistency score in [0,1].

    Score = 1 - (distinct-1)/(n-1); single-item groups are treated as fully consistent.

    Args:
        values: List of prediction sets for the same image.

    Returns:
        Consistency score in [0, 1].
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return 1.0
    distinct = len(set(values))
    return max(0.0, 1.0 - (distinct - 1) / (len(values) - 1))


def _collect_consistency_scores(
    predictions: List[Dict[str, Any]],
    items: Dict[str, Dict[str, Any]],
    *,
    require_correct: bool = True,
) -> Tuple[List[float], int]:
    """Collect consistency scores for query image groups."""

    groups: Dict[str, List[Tuple[frozenset[str], bool]]] = defaultdict(list)
    for pred in predictions:
        item_id = pred.get("item_id")
        if item_id not in items:
            continue
        item = items[item_id]
        image_path = item.get("query", {}).get("image_path", "")
        if not image_path:
            continue
        pred_choices = pred.get("choices", [])
        if not pred_choices and pred.get("choice"):
            pred_choices = [pred["choice"]]
        pred_set = frozenset(pred_choices) if pred_choices else frozenset()
        correct_ids = frozenset(item.get("answer", {}).get("correct_option_ids") or [])
        is_correct = not bool(pred.get("parse_error", 0)) and pred_set == correct_ids
        groups[image_path].append((pred_set, is_correct))

    scores: List[float] = []
    num_groups_excluded = 0
    for entries in groups.values():
        if len(entries) <= 1:
            continue
        if require_correct and not any(is_correct for _, is_correct in entries):
            num_groups_excluded += 1
            continue
        scores.append(_consistency_score([pred_set for pred_set, _ in entries]))
    return scores, num_groups_excluded


def _build_bootstrap_ci(
    values: List[float],
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> Dict[str, float]:
    """Compute bootstrap CI summary for scalar values."""

    if not values or n_bootstrap <= 0:
        return {}
    mean, std, ci_lower, ci_upper = bootstrap_confidence_interval(
        values,
        n_bootstrap=n_bootstrap,
        confidence=confidence_level,
        random_state=seed,
    )
    return {
        "mean": round(mean, 4),
        "std": round(std, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
    }


def compute_consistency(
    predictions: List[Dict[str, Any]],
    items: Dict[str, Dict[str, Any]],
    *,
    require_correct: bool = True,
) -> Dict[str, Any]:
    """Compute consistency across predictions for items with same query image.

    Args:
        predictions: List of prediction dictionaries.
        items: Dictionary mapping item_id to item data.

    Returns:
        Consistency metrics dictionary.
    """
    scores, num_groups_excluded = _collect_consistency_scores(
        predictions,
        items,
        require_correct=require_correct,
    )
    num_groups = len(scores)
    mean_consistency = sum(scores) / len(scores) if scores else 1.0

    return {
        "num_groups": num_groups,
        "num_groups_excluded": num_groups_excluded,
        "mean_consistency": round(mean_consistency, 4),
    }


def load_predictions(predictions_path: Path) -> List[Dict[str, Any]]:
    """Load predictions from JSONL file.

    Args:
        predictions_path: Path to predictions JSONL.

    Returns:
        List of prediction dictionaries.
    """
    predictions = []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    return predictions


def load_items(items_path: Path) -> Dict[str, Dict[str, Any]]:
    """Load items and index by item_id.

    Args:
        items_path: Path to items JSONL.

    Returns:
        Dictionary mapping item_id to item data.
    """
    items = {}
    with open(items_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                items[item["item_id"]] = item
    return items


def get_option_coarse_part(
    item: Dict[str, Any],
    choice: str,
) -> Optional[str]:
    """Get coarse part for a selected option.

    Args:
        item: Benchmark item.
        choice: Selected option ID (A, B, C, ...).

    Returns:
        Coarse part or None if not found.
    """
    for opt in item["options"]:
        if opt["option_id"] == choice:
            return opt.get("coarse_part")
    return None


def evaluate_prosthesis_match(
    predictions_path: Path,
    items_path: Path,
    output_path: Optional[Path] = None,
    compute_bootstrap: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
) -> ProsthesisMetrics:
    """Evaluate Prosthesis Matching predictions.

    Args:
        predictions_path: Path to predictions JSONL.
        items_path: Path to items JSONL.
        output_path: Path to save metrics JSON.
        compute_bootstrap: Whether to compute bootstrap confidence intervals.
        bootstrap_samples: Number of bootstrap samples.
        confidence_level: Confidence level for bootstrap CI.
        bootstrap_seed: Random seed for bootstrap.

    Returns:
        ProsthesisMetrics object.
    """
    # Load data
    predictions = load_predictions(predictions_path)
    items = load_items(items_path)

    logger.info(f"Loaded {len(predictions)} predictions, {len(items)} items")

    # Initialize metrics
    total = 0
    exact_match = 0  # All choices match exactly
    partial_match = 0  # At least one correct choice
    precision_sum = 0.0
    recall_sum = 0.0
    f1_sum = 0.0
    parse_errors = 0
    per_item_values: Dict[str, List[float]] = {
        "exact_accuracy": [],
        "partial_accuracy": [],
        "avg_precision": [],
        "avg_recall": [],
        "avg_f1": [],
        "parse_error_rate": [],
    }

    by_part: Dict[str, Dict[str, int]] = {
        part: {"n": 0, "exact_match": 0, "partial_match": 0} for part in COARSE_PARTS
    }

    # By number of correct answers
    by_num_correct: Dict[str, Dict[str, int]] = {}

    # Confusion matrix: target -> pred -> count
    confusion: Dict[str, Dict[str, int]] = {
        f"target={part}": {f"pred={p}": 0 for p in COARSE_PARTS + ["unknown"]}
        for part in COARSE_PARTS
    }

    # Parse error tracking
    parse_error_samples: List[Dict[str, str]] = []

    # Evaluate each prediction
    for pred in predictions:
        item_id = pred["item_id"]
        # Support both old "choice" and new "choices" format
        pred_choices = pred.get("choices", [])
        if not pred_choices and pred.get("choice"):
            pred_choices = [pred["choice"]]
        pred_choices = set(pred_choices) if pred_choices else set()

        parse_error = pred.get("parse_error", 0)

        if item_id not in items:
            logger.warning(f"Item not found: {item_id}")
            continue

        item = items[item_id]
        correct_ids = set(item["answer"]["correct_option_ids"])
        target_parts = item["answer"]["target_coarse_parts"]
        num_correct = len(correct_ids)

        total += 1

        # Check parse error
        if parse_error:
            parse_errors += 1
            per_item_values["exact_accuracy"].append(0.0)
            per_item_values["partial_accuracy"].append(0.0)
            per_item_values["avg_precision"].append(0.0)
            per_item_values["avg_recall"].append(0.0)
            per_item_values["avg_f1"].append(0.0)
            per_item_values["parse_error_rate"].append(1.0)
            if len(parse_error_samples) < 10:
                parse_error_samples.append({
                    "item_id": item_id,
                    "raw_text": pred.get("raw_text", ""),
                })
            continue

        # Compute set-based metrics
        true_positives = len(pred_choices & correct_ids)
        false_positives = len(pred_choices - correct_ids)
        false_negatives = len(correct_ids - pred_choices)

        # Precision: TP / (TP + FP)
        precision = (
            true_positives / len(pred_choices)
            if pred_choices
            else (1.0 if not correct_ids else 0.0)
        )
        # Recall: TP / (TP + FN)
        recall = (
            true_positives / len(correct_ids)
            if correct_ids
            else (1.0 if not pred_choices else 0.0)
        )
        # F1
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        precision_sum += precision
        recall_sum += recall
        f1_sum += f1

        # Check exact match
        is_exact = pred_choices == correct_ids
        if is_exact:
            exact_match += 1

        # Check partial match (at least one correct)
        is_partial = true_positives > 0 or (
            len(correct_ids) == 0 and len(pred_choices) == 0
        )
        if is_partial:
            partial_match += 1

        per_item_values["exact_accuracy"].append(1.0 if is_exact else 0.0)
        per_item_values["partial_accuracy"].append(1.0 if is_partial else 0.0)
        per_item_values["avg_precision"].append(precision)
        per_item_values["avg_recall"].append(recall)
        per_item_values["avg_f1"].append(f1)
        per_item_values["parse_error_rate"].append(0.0)

        # Update per-part metrics
        for target_part in target_parts:
            by_part[target_part]["n"] += 1
            if is_exact:
                by_part[target_part]["exact_match"] += 1
            if is_partial:
                by_part[target_part]["partial_match"] += 1

        # Update by_num_correct metrics
        num_key = str(num_correct)
        if num_key not in by_num_correct:
            by_num_correct[num_key] = {"n": 0, "exact_match": 0, "partial_match": 0}
        by_num_correct[num_key]["n"] += 1
        if is_exact:
            by_num_correct[num_key]["exact_match"] += 1
        if is_partial:
            by_num_correct[num_key]["partial_match"] += 1

        # Update confusion matrix (use first predicted choice for visualization)
        if pred_choices:
            for choice in pred_choices:
                pred_part = get_option_coarse_part(item, choice)
                if pred_part is None:
                    pred_part = "unknown"
                for target_part in target_parts:
                    conf_key = f"target={target_part}"
                    pred_key = f"pred={pred_part}"
                    if conf_key in confusion and pred_key in confusion[conf_key]:
                        confusion[conf_key][pred_key] += 1

    # Compute metrics
    exact_accuracy = exact_match / total if total > 0 else 0.0
    partial_accuracy = partial_match / total if total > 0 else 0.0
    avg_precision = precision_sum / total if total > 0 else 0.0
    avg_recall = recall_sum / total if total > 0 else 0.0
    avg_f1 = f1_sum / total if total > 0 else 0.0
    parse_error_rate = parse_errors / total if total > 0 else 0.0

    overall = {
        "n": total,
        "exact_match": exact_match,
        "exact_accuracy": round(exact_accuracy, 4),
        "partial_match": partial_match,
        "partial_accuracy": round(partial_accuracy, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "avg_f1": round(avg_f1, 4),
        "parse_error_count": parse_errors,
        "parse_error_rate": round(parse_error_rate, 4),
    }

    by_coarse_part = {}
    for part, stats in by_part.items():
        n = stats["n"]
        exact = stats["exact_match"]
        partial = stats["partial_match"]
        by_coarse_part[part] = {
            "n": n,
            "exact_match": exact,
            "exact_accuracy": round(exact / n, 4) if n > 0 else 0.0,
            "partial_match": partial,
            "partial_accuracy": round(partial / n, 4) if n > 0 else 0.0,
        }

    # By number of correct answers
    by_num_correct_metrics = {}
    for num_key, stats in sorted(by_num_correct.items(), key=lambda x: int(x[0])):
        n = stats["n"]
        exact = stats["exact_match"]
        partial = stats["partial_match"]
        by_num_correct_metrics[num_key] = {
            "n": n,
            "exact_match": exact,
            "exact_accuracy": round(exact / n, 4) if n > 0 else 0.0,
            "partial_match": partial,
            "partial_accuracy": round(partial / n, 4) if n > 0 else 0.0,
        }

    # Compute consistency metrics
    consistency = compute_consistency(predictions, items)

    # Create metrics object
    metrics = ProsthesisMetrics(
        overall=overall,
        by_coarse_part=by_coarse_part,
        confusion=confusion,
        parse_errors={
            "count": parse_errors,
            "rate": round(parse_error_rate, 4),
            "samples": parse_error_samples,
        },
        consistency=consistency,
    )
    # Add by_num_correct to metrics dict
    metrics_dict = metrics.to_dict()
    metrics_dict["by_num_correct"] = by_num_correct_metrics

    if compute_bootstrap and bootstrap_samples > 0:
        overall_ci: Dict[str, Dict[str, float]] = {}
        for key, values in per_item_values.items():
            summary = _build_bootstrap_ci(
                values,
                n_bootstrap=bootstrap_samples,
                confidence_level=confidence_level,
                seed=bootstrap_seed,
            )
            if summary:
                overall_ci[key] = summary
        if overall_ci:
            metrics_dict["overall"]["ci"] = overall_ci

        consistency_scores, _ = _collect_consistency_scores(predictions, items)
        consistency_ci = _build_bootstrap_ci(
            consistency_scores,
            n_bootstrap=bootstrap_samples,
            confidence_level=confidence_level,
            seed=bootstrap_seed,
        )
        if not consistency_ci:
            mean_consistency = float(consistency.get("mean_consistency", 0.0))
            consistency_ci = {
                "mean": round(mean_consistency, 4),
                "std": 0.0,
                "ci_lower": round(mean_consistency, 4),
                "ci_upper": round(mean_consistency, 4),
            }
        metrics_dict.setdefault("consistency", {})["ci"] = {
            "mean_consistency": consistency_ci
        }
        metrics_dict["bootstrap"] = {
            "samples": bootstrap_samples,
            "confidence": confidence_level,
            "seed": bootstrap_seed,
        }

    # Save metrics
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=2, ensure_ascii=False)
        logger.info(f"Metrics saved to {output_path}")

    # Print summary
    logger.info("=" * 60)
    logger.info("Prosthesis Matching Evaluation Results (Variable Answer Mode)")
    logger.info("=" * 60)
    logger.info(f"Total items: {total}")
    logger.info(f"Exact match: {exact_match} ({exact_accuracy:.2%})")
    logger.info(f"Partial match: {partial_match} ({partial_accuracy:.2%})")
    logger.info(f"Avg Precision: {avg_precision:.4f}")
    logger.info(f"Avg Recall: {avg_recall:.4f}")
    logger.info(f"Avg F1: {avg_f1:.4f}")
    logger.info(f"Parse errors: {parse_errors} ({parse_error_rate:.2%})")
    logger.info(
        f"Consistency: {consistency['mean_consistency']:.4f} "
        f"({consistency['num_groups']} groups)"
    )
    logger.info("")
    logger.info("By coarse part:")
    for part, stats in by_coarse_part.items():
        logger.info(
            f"  {part}:"
            f" {stats['exact_accuracy']:.2%} ({stats['exact_match']}/{stats['n']})"
        )

    return metrics


def evaluate_multiple_runs(
    runs_dir: Path,
    items_path: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, ProsthesisMetrics]:
    """Evaluate multiple inference runs.

    Args:
        runs_dir: Directory containing model run directories.
        items_path: Path to items JSONL.
        output_path: Path to save aggregated metrics.

    Returns:
        Dictionary mapping model_name to metrics.
    """
    all_metrics = {}

    runs_dir = Path(runs_dir)
    for model_dir in runs_dir.iterdir():
        if not model_dir.is_dir():
            continue

        # Find latest run
        run_dirs = sorted(model_dir.iterdir(), reverse=True)
        if not run_dirs:
            continue

        latest_run = run_dirs[0]
        predictions_path = latest_run / "predictions.jsonl"

        if not predictions_path.exists():
            logger.warning(f"No predictions found in {latest_run}")
            continue

        model_name = model_dir.name
        logger.info(f"Evaluating {model_name}...")

        metrics = evaluate_prosthesis_match(
            predictions_path,
            items_path,
            latest_run / "metrics.json",
        )
        all_metrics[model_name] = metrics

    # Save aggregated metrics
    if output_path and all_metrics:
        summary = {
            model: {
                "accuracy": m.overall["accuracy"],
                "parse_error_rate": m.overall["parse_error_rate"],
                "n": m.overall["n"],
            }
            for model, m in all_metrics.items()
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        logger.info(f"Summary saved to {output_path}")

    return all_metrics


def configure_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Prosthesis Matching predictions."
    )
    parser.add_argument(
        "--predictions-path",
        type=Path,
        required=True,
        help="Path to predictions JSONL file.",
    )
    parser.add_argument(
        "--items-path",
        type=Path,
        default=DEFAULT_DATA_ITEMS_PATH,
        help=f"Path to items JSONL file. Default: {DEFAULT_DATA_ITEMS_PATH}",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path to save metrics JSON. Default: <predictions_dir>/metrics.json",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
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


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Prosthesis Matching: Evaluate Predictions")
    logger.info("=" * 60)

    # Set default output path
    output_path = args.output_path
    if output_path is None:
        output_path = _infer_output_path(args.predictions_path)

    # Run evaluation
    metrics = evaluate_prosthesis_match(
        args.predictions_path,
        args.items_path,
        output_path,
        compute_bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )

    logger.info("=" * 60)
    logger.info("Evaluation complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
