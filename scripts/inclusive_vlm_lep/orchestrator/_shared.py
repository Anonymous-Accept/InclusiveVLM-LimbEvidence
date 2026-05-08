"""Shared helpers for the InclusiveVLM-LEP recompute/orchestration pipeline."""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from scripts.core.io_utils import load_json, load_jsonl, load_yaml, save_json
from scripts.inclusive_vlm_lep.freeform.normalize_attribution_freeform import (
    DEFAULT_ALIAS_CONFIG,
    POLICY_VERSION as FREEFORM_POLICY_VERSION,
    load_alias_table,
    parse_freeform_output,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_REPORTS_ROOT = PROJECT_ROOT / "outputs" / "inclusive_vlm_lep"
DEFAULT_METRICS_ROOT = DEFAULT_REPORTS_ROOT / "metrics"
DEFAULT_BASELINES_ROOT = DEFAULT_REPORTS_ROOT / "baselines"
DEFAULT_FINDINGS_ROOT = DEFAULT_REPORTS_ROOT / "findings"
DEFAULT_MAIN_RESULTS_PATH = DEFAULT_REPORTS_ROOT / "main_results.md"
DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT / "configs" / "inclusive_vlm_lep" / "orchestrator_default.yaml"
)
RECOMPUTE_POLICY_VERSION = "2026-04-20"
VARIANT_ORDER = [
    "presence",
    "attribution_constrained",
    "compatibility_diversity",
    "compatibility_category",
    "attribution_freeform",
]
KEY_METRIC_BY_VARIANT = {
    "presence": "exact_accuracy",
    "attribution_constrained": "set_miou",
    "compatibility_diversity": "avg_f1",
    "compatibility_category": "hit_at_1",
    "attribution_freeform": "set_miou",
}
DISPLAY_NAME_BY_VARIANT = {
    "presence": "Presence",
    "attribution_constrained": "Attribution-Constrained",
    "compatibility_diversity": "Compatibility-Diversity",
    "compatibility_category": "Compatibility-Category",
    "attribution_freeform": "Attribution-FreeForm",
}
DEFAULT_SUBSET_BY_VARIANT = {
    "presence": PROJECT_ROOT
    / "data"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
    / "queries_v2.jsonl",
    "attribution_constrained": PROJECT_ROOT
    / "data"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
    / "queries_v2.jsonl",
    "compatibility_diversity": PROJECT_ROOT
    / "data"
    / "inclusive_vlm_lep"
    / "prosthesis_match"
    / "items"
    / "prosthesis_match.jsonl",
    "compatibility_category": PROJECT_ROOT
    / "data"
    / "inclusive_vlm_lep"
    / "prosthesis_match"
    / "items"
    / "compatibility_category.jsonl",
    "attribution_freeform": PROJECT_ROOT
    / "data"
    / "inclusive_vlm_lep"
    / "freeform"
    / "freeform_queries.jsonl",
}
DEFAULT_PREDICTIONS_ROOT_BY_VARIANT = {
    "presence": PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
    / "runs",
    "attribution_constrained": PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
    / "runs",
    "compatibility_diversity": PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "prosthesis_matching"
    / "runs"
    / "{model}"
    / "compatibility_diversity",
    "compatibility_category": PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "prosthesis_matching"
    / "runs"
    / "{model}"
    / "compatibility_category",
    "attribution_freeform": PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "freeform"
    / "runs",
}
DEFAULT_GATE_BY_VARIANT = {
    "presence": True,
    "attribution_constrained": True,
    "compatibility_diversity": False,
    "compatibility_category": False,
    "attribution_freeform": True,
}
TASK_NAME_ALIASES = {
    "recognition": {"recognition", "presence"},
    "attribution": {"attribution", "segments"},
}
PRESENCE_OPTIONS = {"a", "b", "c", "d"}


@dataclass(frozen=True)
class VariantDefinition:
    """Static configuration for one orchestrated benchmark variant."""

    name: str
    subset_path: Path
    predictions_root: Path
    default_consistency_gate: bool


def get_variant_definition(variant: str) -> VariantDefinition:
    """Return the static definition for a variant."""

    normalized = normalize_variant_name(variant)
    if normalized not in DEFAULT_SUBSET_BY_VARIANT:
        raise ValueError(
            f"Unknown variant: {variant}. Available: {', '.join(VARIANT_ORDER)}"
        )
    return VariantDefinition(
        name=normalized,
        subset_path=DEFAULT_SUBSET_BY_VARIANT[normalized],
        predictions_root=DEFAULT_PREDICTIONS_ROOT_BY_VARIANT[normalized],
        default_consistency_gate=DEFAULT_GATE_BY_VARIANT[normalized],
    )


def normalize_variant_name(variant: str) -> str:
    """Normalize CLI/config variant names."""

    return variant.strip().lower().replace("-", "_")


def parse_bool_arg(value: str | bool | None, *, default: bool | None = None) -> bool:
    """Parse a forgiving CLI/config boolean value."""

    if value is None:
        if default is None:
            raise ValueError("Boolean value is required.")
        return default
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Unable to parse boolean value: {value}")


def timestamp_slug() -> str:
    """Return a filesystem-safe timestamp."""

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_markdown(path: Path, text: str) -> None:
    """Write markdown text with parent creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write a CSV file with stable headers."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def metric_cell(value: float | int | None, *, digits: int = 3) -> str:
    """Render a scalar metric for markdown."""

    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Convert flat rows to a markdown table."""

    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append(
            "| "
            + " | ".join(str(row.get(key, "")) for key, _ in columns)
            + " |"
        )
    return "\n".join([header, sep, *body]) if body else "\n".join([header, sep])


def load_config(path: Path | str) -> dict[str, Any]:
    """Load YAML config with a default empty mapping."""

    return load_yaml(path)


def iter_metric_jsons(metrics_root: Path) -> list[Path]:
    """Return metric JSON files under a root."""

    if not metrics_root.exists():
        return []
    return sorted(
        path
        for path in metrics_root.rglob("*.metrics.json")
        if path.is_file()
    )


def load_metric_payloads(metrics_root: Path) -> list[dict[str, Any]]:
    """Load all recompute metric payloads under one root."""

    payloads: list[dict[str, Any]] = []
    for path in iter_metric_jsons(metrics_root):
        payload = load_json(path)
        payload["_path"] = str(path)
        payloads.append(payload)
    return payloads


def _format_predictions_root(predictions_root: Path, model_name: str) -> Path:
    """Expand optional {model} placeholders in a variant predictions root."""

    root_text = str(predictions_root)
    if "{model}" in root_text:
        return Path(root_text.format(model=model_name))
    return predictions_root / model_name


def _split_model_root_template(predictions_root: Path) -> tuple[Path, Path] | None:
    """Return the model parent and per-model suffix for a {model} root."""

    root_text = str(predictions_root)
    if "{model}" not in root_text:
        return None
    prefix, suffix = root_text.split("{model}", 1)
    model_parent = Path(prefix.rstrip("/"))
    suffix_text = suffix.lstrip("/")
    suffix_path = Path(suffix_text) if suffix_text else Path()
    return model_parent, suffix_path


def discover_models(predictions_root: Path) -> list[str]:
    """Discover model directories containing predictions."""

    template = _split_model_root_template(predictions_root)
    if template is not None:
        model_parent, suffix_path = template
        if not model_parent.exists():
            return []
        models: list[str] = []
        for path in sorted(model_parent.iterdir()):
            predictions_path = path / suffix_path / "predictions.jsonl"
            if path.is_dir() and predictions_path.exists():
                models.append(path.name)
        return models

    if not predictions_root.exists():
        return []
    models: list[str] = []
    for path in sorted(predictions_root.iterdir()):
        if path.is_dir() and (path / "predictions.jsonl").exists():
            models.append(path.name)
    return models


def resolve_predictions_path(
    variant: str,
    model_name: str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Resolve a model predictions path from config or default roots."""

    variant_config = (config or {}).get("variants", {}).get(variant, {})
    root = Path(
        variant_config.get("predictions_root")
        or get_variant_definition(variant).predictions_root
    )
    return _format_predictions_root(root, model_name) / "predictions.jsonl"


def resolve_subset_path(
    variant: str,
    config: dict[str, Any] | None = None,
) -> Path:
    """Resolve a subset JSONL path from config or defaults."""

    variant_config = (config or {}).get("variants", {}).get(variant, {})
    return Path(
        variant_config.get("subset_jsonl") or get_variant_definition(variant).subset_path
    )


def canonical_task_name(task_name: Any) -> str:
    """Normalize task-name aliases used across Limb-Evidence Grounding schema versions."""

    normalized = str(task_name or "").strip()
    for canonical_name, aliases in TASK_NAME_ALIASES.items():
        if normalized in aliases:
            return canonical_name
    return normalized


def evidence_record_key(record: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
    """Stable join key for Limb-Evidence Grounding records."""

    return (
        canonical_task_name(record.get("task_name")),
        record.get("image_id"),
        record.get("group_id"),
        record.get("prompt_id"),
    )


def compatibility_record_key(record: dict[str, Any]) -> str:
    """Stable join key for Prosthesis Matching records."""

    return str(record.get("item_id") or record.get("record_id") or "").strip()


def freeform_record_key(record: dict[str, Any]) -> str:
    """Stable join key for Attribution-FreeForm records."""

    return str(record.get("record_id") or record.get("item_id") or "").strip()


def canonical_group_join_key(group_id: str | None, image_id: Any) -> str:
    """Normalize family-specific group IDs to a shared image-like key."""

    if image_id is not None:
        return str(image_id)
    text = str(group_id or "").strip()
    match = re.search(r"img(\d+)", text)
    if match:
        return match.group(1)
    return text


def normalize_presence_prediction(record: dict[str, Any]) -> tuple[str | None, bool]:
    """Extract the presence choice and whether it is a parse failure."""

    parsed = record.get("prediction_parsed")
    if isinstance(parsed, str):
        lowered = parsed.strip().lower()
        if lowered in PRESENCE_OPTIONS:
            return lowered, False
    raw = str(record.get("prediction_raw") or record.get("raw_text") or "").strip().lower()
    for option in sorted(PRESENCE_OPTIONS):
        if re.search(rf"\b{re.escape(option)}\b", raw):
            return option, False
    return None, True


def normalize_label_list(value: Any) -> list[str]:
    """Normalize a list-like label container."""

    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [part.strip() for part in text.split(",") if part.strip()]
    labels: list[str] = []
    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text and text.lower() != "none":
                labels.append(text)
    return labels


def normalize_evidence_labels(
    record: dict[str, Any],
    label_vocab: list[str],
) -> tuple[list[str], bool]:
    """Extract normalized constrained attribution labels."""

    labels = normalize_label_list(record.get("prediction_labels"))
    if labels:
        deduped = sorted({label for label in labels if label in set(label_vocab)})
        return deduped, False
    raw = str(record.get("prediction_raw") or "").strip()
    if not raw:
        return [], True
    try:
        payload = json.loads(raw)
        values = normalize_label_list(payload.get("labels"))
        deduped = sorted({label for label in values if label in set(label_vocab)})
        return deduped, False
    except json.JSONDecodeError:
        pass
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    deduped = sorted({part for part in parts if part in set(label_vocab)})
    return deduped, not bool(deduped)


def normalize_choice_list(record: dict[str, Any]) -> tuple[list[str], bool]:
    """Extract constrained option IDs from a prediction record."""

    values = record.get("choices")
    if values is None:
        values = record.get("selected_option_ids")
    if values is None and record.get("choice"):
        values = [record["choice"]]
    choices = []
    if isinstance(values, list):
        for item in values:
            text = str(item).strip().upper()
            if text:
                choices.append(text)
    parse_error = bool(record.get("parse_error", 0))
    return sorted(dict.fromkeys(choices)), parse_error


def set_metrics(gold: set[str], pred: set[str]) -> dict[str, float]:
    """Compute exact/partial/PRF for one set prediction."""

    tp = len(gold & pred)
    fp = len(pred - gold)
    fn = len(gold - pred)
    precision = tp / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = tp / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    union = gold | pred
    return {
        "exact": 1.0 if gold == pred else 0.0,
        "partial": 1.0 if tp > 0 or (not gold and not pred) else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": 1.0 if not union else tp / len(union),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
    }


def consistency_score(values: list[Any]) -> float:
    """Return the smoothed consistency score used across the benchmark."""

    if not values:
        return 0.0
    if len(values) == 1:
        return 1.0
    distinct = len({json.dumps(value, sort_keys=True) for value in values})
    return max(0.0, 1.0 - (distinct - 1) / (len(values) - 1))


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Index scored item rows by paraphrase group."""

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["group_id"])].append(row)
    return grouped


def summarize_group(
    variant: str,
    group_id: str,
    rows: list[dict[str, Any]],
    *,
    gate_enabled: bool,
) -> dict[str, Any]:
    """Build one group-level summary row."""

    exact_values = [float(row["exact"]) for row in rows]
    iou_values = [float(row.get("iou", 0.0)) for row in rows]
    partial_values = [float(row.get("partial", 0.0)) for row in rows]
    parsed_values = [row.get("pred_signature") for row in rows]
    any_correct = any(value >= 1.0 for value in exact_values)
    counted = (not gate_enabled) or any_correct
    return {
        "variant": variant,
        "group_id": group_id,
        "join_group_id": rows[0].get("join_group_id") or group_id,
        "image_id": rows[0].get("image_id"),
        "num_items": len(rows),
        "group_any_correct": any_correct,
        "group_all_correct": all(value >= 1.0 for value in exact_values),
        "group_mean_exact": sum(exact_values) / len(exact_values),
        "group_mean_iou": sum(iou_values) / len(iou_values) if iou_values else 0.0,
        "group_mean_partial": sum(partial_values) / len(partial_values)
        if partial_values
        else 0.0,
        "group_consistency_score": consistency_score(parsed_values),
        "group_consistent": consistency_score(parsed_values) >= 1.0,
        "counted_by_gate": counted,
    }


def attach_group_metadata(
    variant: str,
    rows: list[dict[str, Any]],
    *,
    gate_enabled: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Attach gate/count metadata to per-item rows and return per-group rows."""

    grouped_rows = group_rows(rows)
    summaries = [
        summarize_group(variant, group_id, group_entries, gate_enabled=gate_enabled)
        for group_id, group_entries in sorted(grouped_rows.items())
    ]
    summary_by_group = {row["group_id"]: row for row in summaries}
    enriched_rows: list[dict[str, Any]] = []
    for row in rows:
        group_summary = summary_by_group[str(row["group_id"])]
        enriched = dict(row)
        enriched["counted_by_gate"] = group_summary["counted_by_gate"]
        enriched["group_any_correct"] = group_summary["group_any_correct"]
        enriched["group_consistent"] = group_summary["group_consistent"]
        enriched["group_consistency_score"] = group_summary["group_consistency_score"]
        enriched_rows.append(enriched)
    return enriched_rows, summaries


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _micro_macro_from_rows(
    rows: list[dict[str, Any]],
    *,
    gold_key: str,
    pred_key: str,
) -> dict[str, float]:
    """Compute label-wise micro/macro metrics from set rows."""

    labels: set[str] = set()
    for row in rows:
        labels.update(row.get(gold_key, []))
        labels.update(row.get(pred_key, []))
    stats: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in sorted(labels)
    }
    for row in rows:
        gold = set(row.get(gold_key, []))
        pred = set(row.get(pred_key, []))
        for label in stats:
            in_gold = label in gold
            in_pred = label in pred
            if in_gold and in_pred:
                stats[label]["tp"] += 1
            elif in_gold and not in_pred:
                stats[label]["fn"] += 1
            elif in_pred and not in_gold:
                stats[label]["fp"] += 1
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    tp_total = fp_total = fn_total = 0
    for label_stats in stats.values():
        tp = label_stats["tp"]
        fp = label_stats["fp"]
        fn = label_stats["fn"]
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        tp_total += tp
        fp_total += fp
        fn_total += fn
    micro_precision = _safe_div(tp_total, tp_total + fp_total)
    micro_recall = _safe_div(tp_total, tp_total + fn_total)
    micro_f1 = (
        0.0
        if micro_precision + micro_recall == 0
        else 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
    )
    return {
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": _safe_div(sum(precisions), len(precisions)) if precisions else 0.0,
        "macro_recall": _safe_div(sum(recalls), len(recalls)) if recalls else 0.0,
        "macro_f1": _safe_div(sum(f1s), len(f1s)) if f1s else 0.0,
    }


def _metrics_header(
    variant: str,
    model_name: str,
    gate_enabled: bool,
    kept_rows: list[dict[str, Any]],
    all_rows: list[dict[str, Any]],
    group_rows_payload: list[dict[str, Any]],
    policy_versions: dict[str, Any],
) -> dict[str, Any]:
    """Build the common metrics header."""

    kept_groups = [row for row in group_rows_payload if row["counted_by_gate"]]
    excluded = [row for row in group_rows_payload if not row["counted_by_gate"]]
    return {
        "variant": variant,
        "model_name": model_name,
        "num_samples": len(kept_rows),
        "raw_num_samples": len(all_rows),
        "num_paraphrase_groups": len(kept_groups),
        "raw_num_paraphrase_groups": len(group_rows_payload),
        "consistency_gate_enabled": gate_enabled,
        "num_groups_excluded_by_gate": len(excluded),
        "policy_versions": policy_versions,
    }


def score_presence(
    predictions_path: Path,
    subset_jsonl: Path,
    *,
    consistency_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute Presence metrics from existing Limb-Evidence Grounding predictions."""

    presence_aliases = TASK_NAME_ALIASES["recognition"]
    subset = [
        record
        for record in load_jsonl(subset_jsonl)
        if str(record.get("task_name")) in presence_aliases
    ]
    gold_by_key = {evidence_record_key(record): record for record in subset}
    predictions = [
        record
        for record in load_jsonl(predictions_path)
        if str(record.get("task_name")) in presence_aliases
    ]
    rows: list[dict[str, Any]] = []
    for record in predictions:
        key = evidence_record_key(record)
        gold = gold_by_key.get(key)
        if gold is None:
            continue
        pred_value, parse_error = normalize_presence_prediction(record)
        target = str(gold.get("target_answer") or "").strip().lower()
        exact = 1.0 if pred_value == target else 0.0
        rows.append(
            {
                "variant": "presence",
                "model_name": str(record.get("model_name") or predictions_path.parent.name),
                "item_id": f"{gold.get('group_id')}::{gold.get('prompt_id')}",
                "group_id": str(gold.get("group_id") or ""),
                "join_group_id": canonical_group_join_key(
                    str(gold.get("group_id") or ""), gold.get("image_id")
                ),
                "image_id": gold.get("image_id"),
                "prompt_id": gold.get("prompt_id"),
                "gold_answer": target,
                "pred_answer": pred_value,
                "gold_labels": [],
                "pred_labels": [],
                "parse_error": float(parse_error),
                "exact": exact,
                "partial": exact,
                "precision": exact,
                "recall": exact,
                "f1": exact,
                "iou": exact,
                "pred_signature": pred_value or "__parse_error__",
            }
        )
    enriched_rows, group_summaries = attach_group_metadata(
        "presence", rows, gate_enabled=consistency_gate
    )
    kept_rows = [row for row in enriched_rows if row["counted_by_gate"]]
    model_name = enriched_rows[0]["model_name"] if enriched_rows else predictions_path.parent.name
    payload = _metrics_header(
        "presence",
        model_name,
        consistency_gate,
        kept_rows,
        enriched_rows,
        group_summaries,
        {
            "recompute": RECOMPUTE_POLICY_VERSION,
            "consistency_gate": "require_correct=True",
        },
    )
    accuracy = _safe_div(sum(row["exact"] for row in kept_rows), len(kept_rows))
    parse_error_rate = _safe_div(sum(row["parse_error"] for row in kept_rows), len(kept_rows))
    payload["metrics"] = {
        "exact_accuracy": accuracy,
        "accuracy": accuracy,
        "parse_error_rate": parse_error_rate,
    }
    return payload, enriched_rows, group_summaries


def score_attribution_constrained(
    predictions_path: Path,
    subset_jsonl: Path,
    *,
    consistency_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute Attribution-Constrained metrics."""

    attribution_aliases = TASK_NAME_ALIASES["attribution"]
    subset = [
        record
        for record in load_jsonl(subset_jsonl)
        if str(record.get("task_name")) in attribution_aliases
    ]
    gold_by_key = {evidence_record_key(record): record for record in subset}
    predictions = [
        record
        for record in load_jsonl(predictions_path)
        if str(record.get("task_name")) in attribution_aliases
    ]
    rows: list[dict[str, Any]] = []
    for record in predictions:
        key = evidence_record_key(record)
        gold = gold_by_key.get(key)
        if gold is None:
            continue
        label_vocab = normalize_label_list(gold.get("label_vocab"))
        pred_labels, parse_error = normalize_evidence_labels(record, label_vocab)
        gold_labels = sorted({label for label in normalize_label_list(gold.get("target_labels"))})
        row_metrics = set_metrics(set(gold_labels), set(pred_labels))
        rows.append(
            {
                "variant": "attribution_constrained",
                "model_name": str(record.get("model_name") or predictions_path.parent.name),
                "item_id": f"{gold.get('group_id')}::{gold.get('prompt_id')}",
                "group_id": str(gold.get("group_id") or ""),
                "join_group_id": canonical_group_join_key(
                    str(gold.get("group_id") or ""), gold.get("image_id")
                ),
                "image_id": gold.get("image_id"),
                "prompt_id": gold.get("prompt_id"),
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "parse_error": float(parse_error),
                "pred_signature": pred_labels,
                **row_metrics,
            }
        )
    enriched_rows, group_summaries = attach_group_metadata(
        "attribution_constrained", rows, gate_enabled=consistency_gate
    )
    kept_rows = [row for row in enriched_rows if row["counted_by_gate"]]
    model_name = enriched_rows[0]["model_name"] if enriched_rows else predictions_path.parent.name
    payload = _metrics_header(
        "attribution_constrained",
        model_name,
        consistency_gate,
        kept_rows,
        enriched_rows,
        group_summaries,
        {
            "recompute": RECOMPUTE_POLICY_VERSION,
            "consistency_gate": "require_correct=True",
        },
    )
    micro_macro = _micro_macro_from_rows(
        kept_rows,
        gold_key="gold_labels",
        pred_key="pred_labels",
    )
    payload["metrics"] = {
        "set_miou": _safe_div(sum(row["iou"] for row in kept_rows), len(kept_rows)),
        "mean_iou": _safe_div(sum(row["iou"] for row in kept_rows), len(kept_rows)),
        "exact_accuracy": _safe_div(sum(row["exact"] for row in kept_rows), len(kept_rows)),
        "parse_error_rate": _safe_div(sum(row["parse_error"] for row in kept_rows), len(kept_rows)),
        **micro_macro,
    }
    return payload, enriched_rows, group_summaries


def _compatibility_group_id(item: dict[str, Any]) -> str:
    return str(item.get("person_image", {}).get("path") or item.get("item_id") or "")


def score_compatibility(
    variant: str,
    predictions_path: Path,
    subset_jsonl: Path,
    *,
    consistency_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute Compatibility-Diversity metrics (8-option set-selection).

    The Category variant must NOT use this scorer; it is a 4-option hit@1
    MCQ scored by ``score_compatibility_category`` below. We keep this
    function Diversity-only so that changes to Category scoring cannot
    silently leak into Diversity numbers.
    """

    subset = load_jsonl(subset_jsonl)
    gold_by_key = {compatibility_record_key(record): record for record in subset}
    predictions = load_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for record in predictions:
        item_id = compatibility_record_key(record)
        gold = gold_by_key.get(item_id)
        if gold is None:
            continue
        pred_choices, parse_error = normalize_choice_list(record)
        gold_choices = sorted(
            {
                str(choice).strip().upper()
                for choice in gold.get("answer", {}).get("correct_option_ids", [])
                if str(choice).strip()
            }
        )
        row_metrics = set_metrics(set(gold_choices), set(pred_choices))
        rows.append(
            {
                "variant": variant,
                "model_name": str(record.get("model_name") or predictions_path.parent.name),
                "item_id": item_id,
                "group_id": _compatibility_group_id(gold),
                "join_group_id": Path(_compatibility_group_id(gold)).name,
                "image_id": gold.get("person_image", {}).get("image_id"),
                "prompt_id": None,
                "gold_labels": gold_choices,
                "pred_labels": pred_choices,
                "parse_error": float(parse_error),
                "abstain": float(not pred_choices),
                "pred_signature": pred_choices,
                **row_metrics,
            }
        )
    enriched_rows, group_summaries = attach_group_metadata(
        variant, rows, gate_enabled=consistency_gate
    )
    kept_rows = [row for row in enriched_rows if row["counted_by_gate"]]
    model_name = enriched_rows[0]["model_name"] if enriched_rows else predictions_path.parent.name
    payload = _metrics_header(
        variant,
        model_name,
        consistency_gate,
        kept_rows,
        enriched_rows,
        group_summaries,
        {
            "recompute": RECOMPUTE_POLICY_VERSION,
            "compatibility_gold": "coarse_part",
        },
    )
    payload["metrics"] = {
        "exact_accuracy": _safe_div(sum(row["exact"] for row in kept_rows), len(kept_rows)),
        "partial_accuracy": _safe_div(sum(row["partial"] for row in kept_rows), len(kept_rows)),
        "avg_precision": _safe_div(sum(row["precision"] for row in kept_rows), len(kept_rows)),
        "avg_recall": _safe_div(sum(row["recall"] for row in kept_rows), len(kept_rows)),
        "avg_f1": _safe_div(sum(row["f1"] for row in kept_rows), len(kept_rows)),
        "abstention_rate": _safe_div(sum(row.get("abstain", 0.0) for row in kept_rows), len(kept_rows)),
        "parse_error_rate": _safe_div(sum(row["parse_error"] for row in kept_rows), len(kept_rows)),
    }
    return payload, enriched_rows, group_summaries


def score_compatibility_category(
    predictions_path: Path,
    subset_jsonl: Path,
    *,
    consistency_gate: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute Compatibility-Category metrics as 4-option hit@1 MCQ.

    Primary metric is ``hit_at_1``: the item is counted correct if any of
    the model's picked option letters falls inside the gold
    ``correct_option_ids`` set for that item. Under the 4-option design,
    each option corresponds to one coarse-part category (arm_upper /
    arm_lower / leg_upper / leg_lower), and gold is the subset of
    categories that match the target's visible limb deficiency.

    Secondary set metrics (exact_accuracy / partial_accuracy / avg_f1,
    etc.) are retained for continuity with the previous 8-option design,
    but are no longer the headline numbers.
    """

    variant = "compatibility_category"
    subset = load_jsonl(subset_jsonl)
    gold_by_key = {compatibility_record_key(record): record for record in subset}
    predictions = load_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for record in predictions:
        item_id = compatibility_record_key(record)
        gold = gold_by_key.get(item_id)
        if gold is None:
            continue
        pred_choices, parse_error = normalize_choice_list(record)
        gold_choices = sorted(
            {
                str(choice).strip().upper()
                for choice in gold.get("answer", {}).get("correct_option_ids", [])
                if str(choice).strip()
            }
        )
        # hit@1: model must COMMIT to exactly one option, and it must lie in gold.
        # Multi-select (>=2) is non-committal → not a hit (the task is pick-one).
        gold_set = set(gold_choices)
        pred_set = set(pred_choices)
        if len(pred_set) == 1 and (pred_set & gold_set):
            hit_at_1 = 1.0
        else:
            hit_at_1 = 0.0
        non_committal = 1.0 if len(pred_set) >= 2 else 0.0
        row_metrics = set_metrics(gold_set, pred_set)
        rows.append(
            {
                "variant": variant,
                "model_name": str(record.get("model_name") or predictions_path.parent.name),
                "item_id": item_id,
                "group_id": _compatibility_group_id(gold),
                "join_group_id": Path(_compatibility_group_id(gold)).name,
                "image_id": gold.get("person_image", {}).get("image_id"),
                "prompt_id": None,
                "gold_labels": gold_choices,
                "pred_labels": pred_choices,
                "parse_error": float(parse_error),
                "abstain": float(not pred_choices),
                "hit_at_1": hit_at_1,
                "non_committal": non_committal,
                "num_predictions": float(len(pred_set)),
                "pred_signature": pred_choices,
                **row_metrics,
            }
        )
    enriched_rows, group_summaries = attach_group_metadata(
        variant, rows, gate_enabled=consistency_gate
    )
    kept_rows = [row for row in enriched_rows if row["counted_by_gate"]]
    model_name = enriched_rows[0]["model_name"] if enriched_rows else predictions_path.parent.name
    payload = _metrics_header(
        variant,
        model_name,
        consistency_gate,
        kept_rows,
        enriched_rows,
        group_summaries,
        {
            "recompute": RECOMPUTE_POLICY_VERSION,
            "compatibility_gold": "coarse_part",
            "category_scoring": "hit_at_1_commit_4option_v2",
        },
    )
    payload["metrics"] = {
        "hit_at_1": _safe_div(sum(row["hit_at_1"] for row in kept_rows), len(kept_rows)),
        "non_committal_rate": _safe_div(
            sum(row["non_committal"] for row in kept_rows), len(kept_rows)
        ),
        "num_predictions_mean": _safe_div(
            sum(row["num_predictions"] for row in kept_rows), len(kept_rows)
        ),
        "abstention_rate": _safe_div(
            sum(row.get("abstain", 0.0) for row in kept_rows), len(kept_rows)
        ),
        "parse_error_rate": _safe_div(
            sum(row["parse_error"] for row in kept_rows), len(kept_rows)
        ),
        # Secondary set metrics kept for continuity with the old design.
        "exact_accuracy": _safe_div(sum(row["exact"] for row in kept_rows), len(kept_rows)),
        "partial_accuracy": _safe_div(sum(row["partial"] for row in kept_rows), len(kept_rows)),
        "avg_precision": _safe_div(sum(row["precision"] for row in kept_rows), len(kept_rows)),
        "avg_recall": _safe_div(sum(row["recall"] for row in kept_rows), len(kept_rows)),
        "avg_f1": _safe_div(sum(row["f1"] for row in kept_rows), len(kept_rows)),
        "set_miou": _safe_div(sum(row["iou"] for row in kept_rows), len(kept_rows)),
    }
    payload["primary_metric"] = "hit_at_1"
    return payload, enriched_rows, group_summaries


def score_attribution_freeform(
    predictions_path: Path,
    subset_jsonl: Path,
    *,
    consistency_gate: bool,
    policy_version: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Recompute Attribution-FreeForm metrics under parser v1.1.0."""

    if policy_version != FREEFORM_POLICY_VERSION:
        raise ValueError(
            f"Unsupported Attribution-FreeForm policy_version={policy_version}; "
            f"expected {FREEFORM_POLICY_VERSION}"
        )
    alias_table, _ = load_alias_table(DEFAULT_ALIAS_CONFIG)
    subset = load_jsonl(subset_jsonl)
    gold_by_key = {freeform_record_key(record): record for record in subset}
    predictions = load_jsonl(predictions_path)
    rows: list[dict[str, Any]] = []
    for record in predictions:
        record_id = freeform_record_key(record)
        gold = gold_by_key.get(record_id)
        if gold is None:
            continue
        raw_text = str(
            record.get("raw_output")
            or record.get("prediction_raw")
            or record.get("raw_text")
            or ""
        )
        gold_labels = sorted(
            {str(label) for label in gold.get("canonical_target_set", []) if str(label)}
        )
        parsed = parse_freeform_output(raw_text, gold_labels, alias_table)
        pred_labels = list(parsed.predicted_labels)
        row_metrics = set_metrics(set(gold_labels), set(pred_labels))
        rows.append(
            {
                "variant": "attribution_freeform",
                "model_name": str(record.get("model_name") or predictions_path.parent.name),
                "item_id": record_id,
                "group_id": str(gold.get("group_id") or ""),
                "join_group_id": canonical_group_join_key(
                    str(gold.get("group_id") or ""), gold.get("image_id")
                ),
                "image_id": gold.get("image_id"),
                "prompt_id": gold.get("prompt_id"),
                "gold_labels": gold_labels,
                "pred_labels": pred_labels,
                "parse_error": float(parsed.parser_status == "unparsable"),
                "parse_covered": float(parsed.parser_status != "unparsable"),
                "hallucinated": float(bool(parsed.unresolved_tokens)),
                "parser_status": parsed.parser_status,
                "unresolved_tokens": list(parsed.unresolved_tokens),
                "pred_signature": pred_labels,
                **row_metrics,
            }
        )
    enriched_rows, group_summaries = attach_group_metadata(
        "attribution_freeform", rows, gate_enabled=consistency_gate
    )
    kept_rows = [row for row in enriched_rows if row["counted_by_gate"]]
    model_name = enriched_rows[0]["model_name"] if enriched_rows else predictions_path.parent.name
    payload = _metrics_header(
        "attribution_freeform",
        model_name,
        consistency_gate,
        kept_rows,
        enriched_rows,
        group_summaries,
        {
            "recompute": RECOMPUTE_POLICY_VERSION,
            "parser": policy_version,
            "freeform_aliases": policy_version,
        },
    )
    micro_macro = _micro_macro_from_rows(
        kept_rows,
        gold_key="gold_labels",
        pred_key="pred_labels",
    )
    payload["metrics"] = {
        "set_miou": _safe_div(sum(row["iou"] for row in kept_rows), len(kept_rows)),
        "mean_iou": _safe_div(sum(row["iou"] for row in kept_rows), len(kept_rows)),
        "exact_accuracy": _safe_div(sum(row["exact"] for row in kept_rows), len(kept_rows)),
        "parse_coverage": _safe_div(sum(row["parse_covered"] for row in kept_rows), len(kept_rows)),
        "hallucinated_rate": _safe_div(sum(row["hallucinated"] for row in kept_rows), len(kept_rows)),
        **micro_macro,
    }
    return payload, enriched_rows, group_summaries


def recompute_variant_metrics(
    variant: str,
    predictions_path: Path,
    subset_jsonl: Path,
    *,
    consistency_gate: bool | None = None,
    policy_version: str = FREEFORM_POLICY_VERSION,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Dispatch recomputation for one variant."""

    variant_name = normalize_variant_name(variant)
    gate_enabled = (
        get_variant_definition(variant_name).default_consistency_gate
        if consistency_gate is None
        else bool(consistency_gate)
    )
    if variant_name == "presence":
        return score_presence(
            predictions_path,
            subset_jsonl,
            consistency_gate=gate_enabled,
        )
    if variant_name == "attribution_constrained":
        return score_attribution_constrained(
            predictions_path,
            subset_jsonl,
            consistency_gate=gate_enabled,
        )
    if variant_name == "compatibility_diversity":
        return score_compatibility(
            variant_name,
            predictions_path,
            subset_jsonl,
            consistency_gate=gate_enabled,
        )
    if variant_name == "compatibility_category":
        return score_compatibility_category(
            predictions_path,
            subset_jsonl,
            consistency_gate=gate_enabled,
        )
    if variant_name == "attribution_freeform":
        return score_attribution_freeform(
            predictions_path,
            subset_jsonl,
            consistency_gate=gate_enabled,
            policy_version=policy_version,
        )
    raise ValueError(f"Unsupported variant: {variant}")


def write_metrics_bundle(
    output_json: Path,
    payload: dict[str, Any],
    per_item_rows: list[dict[str, Any]],
    per_group_rows: list[dict[str, Any]],
) -> None:
    """Write metrics JSON plus adjacent per-item/per-group sidecars."""

    output_json.parent.mkdir(parents=True, exist_ok=True)
    stem = output_json.name.replace(".metrics.json", "").replace(".json", "")
    per_item_path = output_json.with_name(f"{stem}.per_item.jsonl")
    per_group_path = output_json.with_name(f"{stem}.per_group.json")
    payload = dict(payload)
    payload["artifacts"] = {
        "per_item_jsonl": per_item_path.name,
        "per_group_json": per_group_path.name,
    }
    save_json(payload, output_json, indent=2, ensure_ascii=False)
    with per_item_path.open("w", encoding="utf-8") as handle:
        for row in per_item_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    save_json({"groups": per_group_rows}, per_group_path, indent=2, ensure_ascii=False)


def load_per_group_rows(metrics_json_path: Path) -> list[dict[str, Any]]:
    """Load the per-group sidecar associated with one metrics JSON."""

    payload = load_json(metrics_json_path)
    artifact_name = payload.get("artifacts", {}).get("per_group_json")
    if not artifact_name:
        return []
    sidecar_path = metrics_json_path.with_name(artifact_name)
    if not sidecar_path.exists():
        return []
    return list(load_json(sidecar_path).get("groups") or [])


def load_per_item_rows(metrics_json_path: Path) -> list[dict[str, Any]]:
    """Load the per-item sidecar associated with one metrics JSON."""

    payload = load_json(metrics_json_path)
    artifact_name = payload.get("artifacts", {}).get("per_item_jsonl")
    if not artifact_name:
        return []
    sidecar_path = metrics_json_path.with_name(artifact_name)
    if not sidecar_path.exists():
        return []
    return load_jsonl(sidecar_path)


def load_metrics_index(metrics_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Index metrics by model then variant."""

    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for path in iter_metric_jsons(metrics_root):
        payload = load_json(path)
        payload["_path"] = str(path)
        model_name = str(payload.get("model_name") or "")
        variant = str(payload.get("variant") or "")
        if model_name and variant:
            index[model_name][variant] = payload
    return index


def load_baseline_index(baselines_root: Path) -> dict[str, dict[str, dict[str, Any]]]:
    """Index baseline metric payloads by baseline then variant."""

    index: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    if not baselines_root.exists():
        return index
    for path in sorted(baselines_root.rglob("baseline_*.metrics.json")):
        payload = load_json(path)
        baseline_name = str(payload.get("baseline_name") or "")
        variant = str(payload.get("variant") or "")
        if baseline_name and variant:
            payload["_path"] = str(path)
            index[baseline_name][variant] = payload
    return index


def best_model_for_variant(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    variant: str,
) -> tuple[str | None, float]:
    """Return the best model under the variant key metric."""

    metric_name = KEY_METRIC_BY_VARIANT[variant]
    best_name = None
    best_value = -math.inf
    for model_name, by_variant in metrics_index.items():
        payload = by_variant.get(variant)
        if not payload:
            continue
        value = float(payload.get("metrics", {}).get(metric_name, 0.0))
        if value > best_value:
            best_name = model_name
            best_value = value
    return best_name, best_value
