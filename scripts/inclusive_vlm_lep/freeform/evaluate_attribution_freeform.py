"""Evaluate Attribution-FreeForm predictions and paired constrained deltas."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from scripts.core.evaluation import bootstrap_confidence_interval
from .normalize_attribution_freeform import (
    DEFAULT_ALIAS_CONFIG,
    load_alias_table,
    parse_freeform_output,
)

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_METRICS_ROOT = (
    PROJECT_ROOT / "outputs" / "inclusive_vlm_lep" / "freeform" / "metrics"
)
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_BOOTSTRAP_SEED = 42
PAIR_AUDIT_FILENAME = "paired_audit.json"
DATE_SUFFIX_PATTERN = re.compile(r"_(\d{8})$")
LABEL_PATTERN = re.compile(
    r"^(upper_arm|forearm|thigh|calf)_([lr])_(residual|prosthesis)$"
)


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Iterate over JSONL records."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _load_subset(path: Path) -> dict[str, dict[str, Any]]:
    """Load subset records keyed by record_id."""

    records: dict[str, dict[str, Any]] = {}
    for record in _iter_jsonl(path):
        record_id = str(record.get("record_id") or "").strip()
        if not record_id:
            raise ValueError(f"Subset record is missing record_id in {path}")
        records[record_id] = record
    return records


def _verify_paired_audit(
    subset_path: Path,
    subset_records: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Verify the mandatory paired audit sidecar before scoring."""

    paired_audit_path = subset_path.parent / PAIR_AUDIT_FILENAME
    if not paired_audit_path.exists():
        raise FileNotFoundError(
            f"Required paired audit sidecar missing: {paired_audit_path}"
        )
    with paired_audit_path.open("r", encoding="utf-8") as handle:
        audit = json.load(handle)

    pairs = audit.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError(f"paired_audit.json has no 'pairs' list: {paired_audit_path}")

    ineligible = [pair for pair in pairs if not pair.get("eligible_for_scoring", False)]
    if ineligible:
        raise ValueError(
            f"paired_audit.json contains {len(ineligible)} ineligible pair(s); fail closed."
        )

    missing = []
    for pair in pairs:
        right_item_id = str(pair.get("right_item_id") or "")
        if right_item_id not in subset_records:
            missing.append(right_item_id)
    if missing:
        raise ValueError(
            f"paired_audit.json references {len(missing)} missing subset record(s)."
        )
    return audit


def _discover_prediction_files(path_spec: str) -> list[Path]:
    """Resolve a path, directory, or glob into prediction files."""

    path = Path(path_spec)
    if any(token in path_spec for token in "*?[]"):
        matches = sorted(Path().glob(path_spec))
        return [match for match in matches if match.is_file()]
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted(path.rglob("predictions.jsonl"))
        if files:
            return files
        return sorted(item for item in path.rglob("*.jsonl") if item.is_file())
    raise FileNotFoundError(f"No predictions found for {path_spec}")


def _default_output_dir_for_predictions(path_spec: str) -> Path:
    """Infer the default metrics directory for one prediction input."""

    prediction_files = _discover_prediction_files(path_spec)
    if len(prediction_files) != 1:
        raise ValueError(
            "When --output-dir is omitted, --predictions must resolve to exactly one predictions.jsonl file."
        )
    parent_name = prediction_files[0].parent.name
    match = DATE_SUFFIX_PATTERN.search(parent_name)
    model_name = parent_name[: match.start()] if match else parent_name
    return DEFAULT_METRICS_ROOT / model_name


def _infer_model_name(prediction_path: Path, records: list[dict[str, Any]]) -> str:
    """Infer model name from record payloads or the parent directory."""

    record_names = {
        str(record.get("model_name")).strip()
        for record in records
        if str(record.get("model_name") or "").strip()
    }
    if len(record_names) == 1:
        return next(iter(record_names))

    parent_name = prediction_path.parent.name
    match = DATE_SUFFIX_PATTERN.search(parent_name)
    if match:
        return parent_name[: match.start()]
    return parent_name or prediction_path.stem


def _record_prefix_from_item(item: dict[str, Any]) -> str:
    """Resolve a stable constrained record prefix."""

    record_id = str(item.get("record_id") or "").strip()
    if record_id:
        return record_id
    group_id = str(item.get("group_id") or "")
    prompt_id = item.get("prompt_id")
    if group_id and prompt_id is not None:
        return f"{group_id}::prompt{prompt_id}"
    if group_id:
        return group_id
    raise ValueError(f"Unable to build record prefix from item: {item}")


def _canonicalize_composite_labels(labels: list[str] | None) -> set[str]:
    """Convert constrained composite labels into the canonical cue space."""

    normalized = [str(label).strip().lower() for label in (labels or []) if str(label).strip()]
    if not normalized or normalized == ["none"]:
        return {"none_visible"}

    canonical: set[str] = set()
    for label in normalized:
        match = LABEL_PATTERN.match(label)
        if not match:
            continue
        segment, side, kind = match.groups()
        canonical.add(segment)
        canonical.add("left" if side == "l" else "right")
        canonical.add("residual_limb" if kind == "residual" else "prosthesis")
    return canonical


def _score_set_prediction(gold: set[str], pred: set[str]) -> dict[str, Any]:
    """Compute per-item set metrics."""

    union = gold | pred
    inter = gold & pred
    precision = len(inter) / len(pred) if pred else 0.0
    recall = len(inter) / len(gold) if gold else 0.0
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return {
        "exact_set_match": pred == gold,
        "set_iou": 1.0 if not union else len(inter) / len(union),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": len(inter),
        "fp": len(pred - gold),
        "fn": len(gold - pred),
        "missing_labels": sorted(gold - pred),
        "extra_labels": sorted(pred - gold),
        "overlap_labels": sorted(inter),
    }


def _counter_stats(values: list[int]) -> dict[str, Any]:
    """Summarize an integer count distribution."""

    if not values:
        return {
            "histogram": {},
            "mean": 0.0,
            "median": 0.0,
            "p10": 0.0,
            "p90": 0.0,
            "n": 0,
        }
    array = np.array(values, dtype=float)
    hist = Counter(values)
    return {
        "histogram": {str(key): int(hist[key]) for key in sorted(hist)},
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p10": float(np.percentile(array, 10)),
        "p90": float(np.percentile(array, 90)),
        "n": int(len(values)),
    }


def _summarize_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate summary metrics for one model."""

    num_samples = len(rows)
    exact_values = [1.0 if row["exact_set_match"] else 0.0 for row in rows]
    iou_values = [float(row["set_iou"]) for row in rows]
    precision_values = [float(row["precision"]) for row in rows]
    recall_values = [float(row["recall"]) for row in rows]
    f1_values = [float(row["f1"]) for row in rows]
    invalid_values = [1.0 if row["invalid"] else 0.0 for row in rows]
    hallucinated_values = [1.0 if row["hallucinated"] else 0.0 for row in rows]
    omitted_values = [1.0 if row["omitted"] else 0.0 for row in rows]

    tp = sum(int(row["tp"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    micro_precision = tp / (tp + fp) if tp + fp else 0.0
    micro_recall = tp / (tp + fn) if tp + fn else 0.0
    micro_f1 = (
        0.0
        if micro_precision + micro_recall == 0.0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )

    return {
        "num_samples": num_samples,
        "exact_set_match": float(np.mean(exact_values)) if exact_values else 0.0,
        "set_miou": float(np.mean(iou_values)) if iou_values else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
        "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
        "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
        "invalid_rate": float(np.mean(invalid_values)) if invalid_values else 0.0,
        "hallucinated_rate": float(np.mean(hallucinated_values)) if hallucinated_values else 0.0,
        "omitted_rate": float(np.mean(omitted_values)) if omitted_values else 0.0,
        "count_diagnostics": {
            "gold_count": _counter_stats([int(row["gold_count"]) for row in rows]),
            "answer_count_predicted": _counter_stats(
                [int(row["answer_count_predicted"]) for row in rows]
            ),
        },
        "metric_notes": {
            "macro_definition": "item-average",
            "canonical_space": "parser_normalization_policy.md v1.1.0",
        },
    }


def _rank(values: list[float]) -> list[float]:
    """Compute average ranks with ties."""

    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        j = idx
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[idx][1]:
            j += 1
        avg_rank = 1.0 + (idx + j) / 2.0
        for k in range(idx, j + 1):
            ranks[indexed[k][0]] = avg_rank
        idx = j + 1
    return ranks


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation."""

    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x_vals = x[:n]
    y_vals = y[:n]
    mx = sum(x_vals) / n
    my = sum(y_vals) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x_vals, y_vals))
    den_x = math.sqrt(sum((a - mx) ** 2 for a in x_vals))
    den_y = math.sqrt(sum((b - my) ** 2 for b in y_vals))
    if den_x == 0.0 or den_y == 0.0:
        return 0.0
    return num / (den_x * den_y)


def _spearman(x: list[float], y: list[float]) -> float:
    """Compute Spearman rank correlation."""

    return _pearson(_rank(x), _rank(y))


def _kendall_tau_b(x: list[float], y: list[float]) -> float:
    """Compute Kendall tau-b."""

    n = min(len(x), len(y))
    if n < 2:
        return 0.0
    x_vals = x[:n]
    y_vals = y[:n]
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = 0 if x_vals[i] == x_vals[j] else (1 if x_vals[i] > x_vals[j] else -1)
            dy = 0 if y_vals[i] == y_vals[j] else (1 if y_vals[i] > y_vals[j] else -1)
            if dx == 0 or dy == 0:
                continue
            if dx == dy:
                concordant += 1
            else:
                discordant += 1

    tie_x = Counter(x_vals)
    tie_y = Counter(y_vals)
    n0 = n * (n - 1) / 2
    n1 = sum(count * (count - 1) / 2 for count in tie_x.values())
    n2 = sum(count * (count - 1) / 2 for count in tie_y.values())
    denom = math.sqrt((n0 - n1) * (n0 - n2))
    if denom == 0.0:
        return 0.0
    return (concordant - discordant) / denom


def _bootstrap_delta(
    values: list[float],
    *,
    n_bootstrap: int,
    confidence_level: float,
    seed: int,
) -> dict[str, float]:
    """Bootstrap the mean delta for paired item scores."""

    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}
    mean, std, ci_lower, ci_upper = bootstrap_confidence_interval(
        values,
        n_bootstrap=n_bootstrap,
        confidence=confidence_level,
        random_state=seed,
    )
    return {
        "mean": float(mean),
        "std": float(std),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON to disk."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _load_predictions_by_model(path_spec: str) -> dict[str, list[dict[str, Any]]]:
    """Load one or more prediction files grouped by model name."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for prediction_path in _discover_prediction_files(path_spec):
        records = list(_iter_jsonl(prediction_path))
        if not records:
            continue
        model_name = _infer_model_name(prediction_path, records)
        grouped.setdefault(model_name, []).extend(records)
    return grouped


def _score_freeform_predictions(
    predictions_by_model: dict[str, list[dict[str, Any]]],
    subset_records: dict[str, dict[str, Any]],
    alias_table: dict[str, list[str]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Score freeform predictions against the paired subset."""

    model_summaries: dict[str, dict[str, Any]] = {}
    per_item_rows: list[dict[str, Any]] = []

    for model_name, predictions in sorted(predictions_by_model.items()):
        rows: list[dict[str, Any]] = []
        for prediction in predictions:
            record_id = str(prediction.get("record_id") or "").strip()
            if not record_id or record_id not in subset_records:
                continue
            subset_item = subset_records[record_id]
            gold = set(subset_item.get("canonical_target_set") or [])
            parsed = parse_freeform_output(
                raw_text=str(prediction.get("raw_output") or ""),
                canonical_target_set=gold,
                alias_table=alias_table,
            )
            predicted_set = set(parsed.predicted_labels)
            score = _score_set_prediction(gold=gold, pred=predicted_set)
            row = {
                "model_name": model_name,
                "record_id": record_id,
                "source_record_id": subset_item.get("source_record_id"),
                "pair_id": subset_item.get("pair_id"),
                "image_id": subset_item.get("image_id"),
                "group_id": subset_item.get("group_id"),
                "variant": subset_item.get("variant"),
                "gold_labels": sorted(gold),
                "predicted_labels": list(parsed.predicted_labels),
                "raw_output": str(prediction.get("raw_output") or ""),
                "normalized_text": parsed.normalized_text,
                "unresolved_tokens": list(parsed.unresolved_tokens),
                "flags": list(parsed.flags),
                "parser_status": parsed.parser_status,
                "invalid": parsed.parser_status in {"hallucinated", "unparsable"},
                "hallucinated": "hallucinated" in parsed.flags,
                "omitted": bool(gold - predicted_set),
                "gold_count": len(gold),
                "answer_count_predicted": len(parsed.predicted_labels),
                "latency_ms": prediction.get("latency_ms"),
                **score,
            }
            rows.append(row)
            per_item_rows.append(row)
        model_summaries[model_name] = _summarize_model(rows)
    return model_summaries, per_item_rows


def _score_constrained_predictions(
    predictions_by_model: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, dict[str, Any]]]:
    """Score constrained predictions in the same canonical space."""

    by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model_name, predictions in sorted(predictions_by_model.items()):
        model_rows: dict[str, dict[str, Any]] = {}
        for prediction in predictions:
            record_prefix = _record_prefix_from_item(prediction)
            gold = _canonicalize_composite_labels(prediction.get("target_labels") or [])
            pred = _canonicalize_composite_labels(prediction.get("prediction_labels") or [])
            score = _score_set_prediction(gold=gold, pred=pred)
            model_rows[record_prefix] = {
                "record_id": record_prefix,
                "gold_labels": sorted(gold),
                "predicted_labels": sorted(pred),
                "invalid": not bool(prediction.get("success", True)),
                "source_raw_output": prediction.get("prediction_raw"),
                **score,
            }
        by_model[model_name] = model_rows
    return by_model


def _write_per_item_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write per-item JSONL scoring rows."""

    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = {
                key: value
                for key, value in row.items()
                if key not in {"tp", "fp", "fn"}
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_failure_examples(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a short markdown file with representative failures."""

    categories = {
        "parse_error": lambda row: row["parser_status"] == "unparsable",
        "hallucination": lambda row: row["hallucinated"],
        "omission": lambda row: row["omitted"] and row["parser_status"] != "unparsable",
        "partial_match": lambda row: (not row["exact_set_match"]) and bool(row["overlap_labels"]),
    }

    lines = ["# Failure Examples", ""]
    for category, predicate in categories.items():
        example = next((row for row in rows if predicate(row)), None)
        lines.append(f"## {category}")
        if example is None:
            lines.append("No example found.")
            lines.append("")
            continue
        lines.extend(
            [
                f"- model: `{example['model_name']}`",
                f"- record_id: `{example['record_id']}`",
                f"- gold: `{example['gold_labels']}`",
                f"- predicted: `{example['predicted_labels']}`",
                f"- unresolved_tokens: `{example['unresolved_tokens']}`",
                f"- flags: `{example['flags']}`",
                f"- raw_output: `{example['raw_output']}`",
                "",
            ]
        )

    path.write_text("\n".join(lines), encoding="utf-8")


def _write_paired_delta_csv(
    path: Path,
    freeform_rows: list[dict[str, Any]],
    constrained_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Write per-item constrained/freeform deltas and summarize bootstrap stats."""

    fieldnames = [
        "model_name",
        "pair_id",
        "record_id_constrained",
        "record_id_freeform",
        "image_id",
        "group_id",
        "constrained_exact_set_match",
        "freeform_exact_set_match",
        "delta_exact_set_match",
        "constrained_set_iou",
        "freeform_set_iou",
        "delta_set_iou",
        "constrained_f1",
        "freeform_f1",
        "delta_f1",
        "constrained_invalid",
        "freeform_invalid",
    ]

    bootstrap_summary: dict[str, Any] = {}
    deltas_by_model: dict[str, dict[str, list[float]]] = {}
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for freeform_row in freeform_rows:
            model_name = str(freeform_row["model_name"])
            source_record_id = str(freeform_row["source_record_id"] or "")
            constrained = constrained_rows.get(model_name, {}).get(source_record_id)
            if constrained is None:
                continue
            row = {
                "model_name": model_name,
                "pair_id": freeform_row["pair_id"],
                "record_id_constrained": constrained["record_id"],
                "record_id_freeform": freeform_row["record_id"],
                "image_id": freeform_row["image_id"],
                "group_id": freeform_row["group_id"],
                "constrained_exact_set_match": int(bool(constrained["exact_set_match"])),
                "freeform_exact_set_match": int(bool(freeform_row["exact_set_match"])),
                "delta_exact_set_match": int(bool(freeform_row["exact_set_match"]))
                - int(bool(constrained["exact_set_match"])),
                "constrained_set_iou": constrained["set_iou"],
                "freeform_set_iou": freeform_row["set_iou"],
                "delta_set_iou": freeform_row["set_iou"] - constrained["set_iou"],
                "constrained_f1": constrained["f1"],
                "freeform_f1": freeform_row["f1"],
                "delta_f1": freeform_row["f1"] - constrained["f1"],
                "constrained_invalid": int(bool(constrained["invalid"])),
                "freeform_invalid": int(bool(freeform_row["invalid"])),
            }
            writer.writerow(row)

            model_deltas = deltas_by_model.setdefault(
                model_name,
                {"exact_set_match_delta": [], "set_iou_delta": [], "f1_delta": []},
            )
            model_deltas["exact_set_match_delta"].append(float(row["delta_exact_set_match"]))
            model_deltas["set_iou_delta"].append(float(row["delta_set_iou"]))
            model_deltas["f1_delta"].append(float(row["delta_f1"]))

    for model_name, metrics in deltas_by_model.items():
        bootstrap_summary[model_name] = metrics
    return bootstrap_summary


def _build_ranking_correlation(
    freeform_summary: dict[str, dict[str, Any]],
    constrained_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    """Compute ranking correlation between constrained and freeform models."""

    overlapping_models = sorted(
        model for model in freeform_summary if model in constrained_rows and constrained_rows[model]
    )
    if len(overlapping_models) < 3:
        return {
            "status": "not_available",
            "reason": "Need at least 3 overlapping models with constrained predictions.",
            "num_models": len(overlapping_models),
        }

    freeform_scores = [freeform_summary[model]["set_miou"] for model in overlapping_models]
    constrained_scores = []
    for model in overlapping_models:
        iou_values = [float(row["set_iou"]) for row in constrained_rows[model].values()]
        constrained_scores.append(float(np.mean(iou_values)) if iou_values else 0.0)

    return {
        "status": "ok",
        "primary_metric": "set_miou",
        "models": overlapping_models,
        "freeform_scores": freeform_scores,
        "constrained_scores": constrained_scores,
        "spearman": _spearman(constrained_scores, freeform_scores),
        "kendall_tau": _kendall_tau_b(constrained_scores, freeform_scores),
    }


def evaluate_attribution_freeform(
    *,
    predictions: str,
    subset_jsonl: Path,
    output_dir: Path | None = None,
    constrained_predictions: str | None = None,
    alias_config: Path = DEFAULT_ALIAS_CONFIG,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Evaluate Attribution-FreeForm predictions.

    Args:
        predictions: Prediction file path, directory, or glob.
        subset_jsonl: Paired free-form subset JSONL.
        output_dir: Destination metrics directory.
        constrained_predictions: Optional constrained prediction file path, directory, or glob.
        alias_config: Alias-table YAML path.
        bootstrap_samples: Number of paired bootstrap samples.
        confidence_level: Bootstrap confidence level.
        bootstrap_seed: Random seed for paired bootstrap.

    Returns:
        Summary payload written to `summary.json`.
    """

    output_dir = output_dir or _default_output_dir_for_predictions(predictions)
    output_dir.mkdir(parents=True, exist_ok=True)
    subset_records = _load_subset(subset_jsonl)
    audit = _verify_paired_audit(subset_jsonl, subset_records)
    alias_table, _ = load_alias_table(alias_config)

    freeform_predictions = _load_predictions_by_model(predictions)
    freeform_summary, per_item_rows = _score_freeform_predictions(
        freeform_predictions,
        subset_records,
        alias_table,
    )

    per_item_path = output_dir / "per_item.jsonl"
    _write_per_item_jsonl(per_item_path, per_item_rows)
    _write_failure_examples(output_dir / "failure_examples.md", per_item_rows)

    constrained_rows: dict[str, dict[str, dict[str, Any]]] = {}
    if constrained_predictions:
        constrained_rows = _score_constrained_predictions(
            _load_predictions_by_model(constrained_predictions)
        )

    paired_delta_path = output_dir / "paired_delta.csv"
    paired_delta_values = _write_paired_delta_csv(
        paired_delta_path,
        per_item_rows,
        constrained_rows,
    )

    paired_bootstrap: dict[str, Any] = {
        "status": "not_available",
        "reason": "No constrained predictions provided.",
    }
    if constrained_rows:
        paired_bootstrap = {
            "status": "ok",
            "samples": bootstrap_samples,
            "confidence_level": confidence_level,
            "per_model": {},
        }
        for model_name, metrics in paired_delta_values.items():
            paired_bootstrap["per_model"][model_name] = {
                metric_name: _bootstrap_delta(
                    metric_values,
                    n_bootstrap=bootstrap_samples,
                    confidence_level=confidence_level,
                    seed=bootstrap_seed,
                )
                for metric_name, metric_values in metrics.items()
            }

    summary = {
        "subset_jsonl": str(subset_jsonl),
        "paired_audit": {
            "path": str(subset_jsonl.parent / PAIR_AUDIT_FILENAME),
            "total_pairs": audit.get("total_pairs"),
            "passed_pairs": audit.get("passed_pairs"),
            "failed_pairs": audit.get("failed_pairs"),
        },
        "models": freeform_summary,
        "ranking_correlation": _build_ranking_correlation(freeform_summary, constrained_rows),
        "paired_bootstrap": paired_bootstrap,
    }
    _write_json(output_dir / "summary.json", summary)
    LOGGER.info("Wrote evaluation summary to %s", output_dir / "summary.json")
    LOGGER.info("Wrote per-item report to %s", per_item_path)
    LOGGER.info("Wrote paired delta CSV to %s", paired_delta_path)
    return summary


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Evaluate Attribution-FreeForm predictions."
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Prediction file path, directory, or glob.",
    )
    parser.add_argument(
        "--subset-jsonl",
        type=Path,
        required=True,
        help="Attribution-FreeForm paired subset JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output metrics directory (default: outputs/inclusive_vlm_lep/freeform/metrics/<model>/).",
    )
    parser.add_argument(
        "--constrained-predictions",
        type=str,
        default=None,
        help="Optional constrained prediction file path, directory, or glob.",
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=DEFAULT_ALIAS_CONFIG,
        help="Alias-table YAML path.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help="Number of paired bootstrap samples.",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=DEFAULT_CONFIDENCE_LEVEL,
        help="Paired bootstrap confidence level.",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help="Paired bootstrap seed.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    evaluate_attribution_freeform(
        predictions=args.predictions,
        subset_jsonl=args.subset_jsonl,
        output_dir=args.output_dir,
        constrained_predictions=args.constrained_predictions,
        alias_config=args.alias_config,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
    )


if __name__ == "__main__":
    main()
