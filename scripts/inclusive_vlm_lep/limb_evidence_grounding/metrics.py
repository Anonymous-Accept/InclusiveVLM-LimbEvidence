"""Metric computations for Limb-Evidence Grounding (spec_009)."""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from . import config as benchmark_config
from .eval_config import BENCHMARK_NAME, BENCHMARK_PART

OPTION_TO_LABEL = {v: k for k, v in benchmark_config.PRESENCE_LABEL_TO_OPTION.items()}
PRESENCE_OPTION_ORDER = ["a", "b", "c", "d"]
PRESENCE_LABEL_ORDER = [
    OPTION_TO_LABEL[opt] for opt in PRESENCE_OPTION_ORDER if opt in OPTION_TO_LABEL
]

def _consistency_score(values: Sequence[Any]) -> float:
    """Return smoothed consistency score in [0,1].

    Score = 1 - (distinct-1)/(n-1); single-item groups are treated as fully consistent.
    """

    if not values:
        return 0.0
    if len(values) == 1:
        return 1.0
    distinct = len(set(values))
    return max(0.0, 1.0 - (distinct - 1) / (len(values) - 1))


def _safe_div(numerator: float, denominator: float) -> float:
    """Avoid division by zero."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def _f1(precision: float, recall: float) -> float:
    """Compute F1 with zero guard."""

    if precision == 0 or recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _init_stats(labels: Sequence[str]) -> Dict[str, Dict[str, float]]:
    """Initialize stats container."""

    return {label: {"tp": 0, "fp": 0, "fn": 0} for label in labels}


def compute_presence_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute accuracy, per-class metrics, and confusion matrix."""

    entries = [rec for rec in records if rec.get("task_name") == "recognition"]
    total = len(entries)
    correct = 0
    stats = _init_stats(PRESENCE_LABEL_ORDER)
    confusion = [[0 for _ in PRESENCE_OPTION_ORDER] for _ in PRESENCE_OPTION_ORDER]
    invalid_predictions = 0

    for rec in entries:
        target_opt = rec.get("target_answer")
        pred_opt = rec.get("prediction_parsed")

        if target_opt not in OPTION_TO_LABEL:
            continue
        target_label = OPTION_TO_LABEL[target_opt]
        pred_label = OPTION_TO_LABEL[pred_opt] if pred_opt in OPTION_TO_LABEL else None

        if pred_label is not None and pred_label == target_label:
            correct += 1

        for label in PRESENCE_LABEL_ORDER:
            if target_label == label and pred_label == label:
                stats[label]["tp"] += 1
            elif target_label == label and pred_label != label:
                stats[label]["fn"] += 1
            elif pred_label == label and target_label != label:
                stats[label]["fp"] += 1

        if pred_opt in PRESENCE_OPTION_ORDER:
            ti = PRESENCE_OPTION_ORDER.index(target_opt)
            pi = PRESENCE_OPTION_ORDER.index(pred_opt)
            confusion[ti][pi] += 1
        else:
            invalid_predictions += 1

    per_class = {}
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []

    for label in PRESENCE_LABEL_ORDER:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        support = tp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = _safe_div(correct, total)
    macro_precision = _safe_div(sum(precisions), len(precisions)) if precisions else 0.0
    macro_recall = _safe_div(sum(recalls), len(recalls)) if recalls else 0.0
    macro_f1 = _safe_div(sum(f1s), len(f1s)) if f1s else 0.0

    parse_error_count = invalid_predictions
    parse_error_rate = _safe_div(parse_error_count, total)

    return {
        "num_samples": total,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "parse_error_count": parse_error_count,
        "parse_error_rate": parse_error_rate,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": PRESENCE_OPTION_ORDER,
            "matrix": confusion,
            "invalid_predictions": invalid_predictions,
        },
    }


def _collect_label_vocab(records: Iterable[Dict[str, Any]]) -> List[str]:
    """Collect label vocabulary from targets and predictions."""

    vocab = set()
    for rec in records:
        for label in rec.get("label_vocab") or []:
            if label and label != "none":
                vocab.add(label)
        for label in rec.get("target_labels") or []:
            if label and label != "none":
                vocab.add(label)
        for label in rec.get("prediction_labels") or []:
            if label and label != "none":
                vocab.add(label)
    return sorted(vocab)


def compute_segments_metrics(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute IoU and multi-label classification metrics."""

    entries = [rec for rec in records if rec.get("task_name") == "attribution"]
    total = len(entries)
    if total == 0:
        return {
            "num_samples": 0,
            "mean_iou": 0.0,
            "micro_precision": 0.0,
            "micro_recall": 0.0,
            "micro_f1": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "parse_error_count": 0,
            "parse_error_rate": 0.0,
            "per_label": {},
        }

    label_vocab = _collect_label_vocab(entries)
    stats = _init_stats(label_vocab)
    ious: List[float] = []
    parse_errors = 0

    for rec in entries:
        if not rec.get("success", True):
            parse_errors += 1
        gt = {l for l in rec.get("target_labels") or [] if l and l != "none"}
        pred = {l for l in rec.get("prediction_labels") or [] if l and l != "none"}

        union = gt | pred
        inter = gt & pred
        iou = 1.0 if not union else _safe_div(len(inter), len(union))
        ious.append(iou)

        for label in label_vocab:
            in_gt = label in gt
            in_pred = label in pred
            if in_gt and in_pred:
                stats[label]["tp"] += 1
            elif in_gt and not in_pred:
                stats[label]["fn"] += 1
            elif in_pred and not in_gt:
                stats[label]["fp"] += 1

    per_label = {}
    micro_tp = micro_fp = micro_fn = 0
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    supports: List[int] = []

    for label in label_vocab:
        tp = stats[label]["tp"]
        fp = stats[label]["fp"]
        fn = stats[label]["fn"]
        support = tp + fn
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _f1(precision, recall)

        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

        micro_tp += tp
        micro_fp += fp
        micro_fn += fn
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        supports.append(support)

    micro_precision = _safe_div(micro_tp, micro_tp + micro_fp)
    micro_recall = _safe_div(micro_tp, micro_tp + micro_fn)
    micro_f1 = _f1(micro_precision, micro_recall)
    macro_precision = _safe_div(sum(precisions), len(precisions)) if precisions else 0.0
    macro_recall = _safe_div(sum(recalls), len(recalls)) if recalls else 0.0
    macro_f1 = _safe_div(sum(f1s), len(f1s)) if f1s else 0.0

    return {
        "num_samples": total,
        "mean_iou": _safe_div(sum(ious), len(ious)) if ious else 0.0,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "parse_error_count": parse_errors,
        "parse_error_rate": _safe_div(parse_errors, total),
        "per_label": per_label,
    }


def _presence_consistency_scores(
    records: Iterable[Dict[str, Any]],
    *,
    require_correct: bool = True,
) -> Tuple[List[float], int]:
    """Collect consistency scores for presence predictions by image."""

    groups: Dict[Any, List[Tuple[str, bool]]] = {}
    for rec in records:
        if rec.get("task_name") != "recognition":
            continue
        image_id = rec.get("image_id")
        pred = rec.get("prediction_parsed")
        if pred is None:
            continue
        groups.setdefault(image_id, []).append((pred, _presence_prompt_correct(rec)))

    scores: List[float] = []
    num_groups_excluded = 0
    for entries in groups.values():
        if require_correct and not any(is_correct for _, is_correct in entries):
            num_groups_excluded += 1
            continue
        scores.append(_consistency_score([pred for pred, _ in entries]))

    return scores, num_groups_excluded


def _segments_consistency_scores(
    records: Iterable[Dict[str, Any]],
    *,
    require_correct: bool = True,
) -> Tuple[List[float], int]:
    """Collect consistency scores for segment predictions by image."""

    groups: Dict[Any, List[Tuple[frozenset[str], bool]]] = {}
    for rec in records:
        if rec.get("task_name") != "attribution":
            continue
        image_id = rec.get("image_id")
        labels = rec.get("prediction_labels") or []
        labels_set = frozenset(l for l in labels if l)
        groups.setdefault(image_id, []).append((labels_set, _segments_prompt_correct(rec)))

    scores: List[float] = []
    num_groups_excluded = 0
    for entries in groups.values():
        if require_correct and not any(is_correct for _, is_correct in entries):
            num_groups_excluded += 1
            continue
        scores.append(_consistency_score([labels_set for labels_set, _ in entries]))

    return scores, num_groups_excluded


def _summarize_consistency_scores(
    scores: Sequence[float],
    *,
    num_groups_excluded: int = 0,
) -> Dict[str, Any]:
    """Summarize consistency scores."""

    return {
        "num_groups": len(scores),
        "num_groups_excluded": num_groups_excluded,
        "mean_consistency": _safe_div(sum(scores), len(scores)) if scores else 0.0,
    }


def _percentile_bounds(
    values: Sequence[float],
    confidence: float,
) -> Tuple[float, float]:
    """Compute percentile bounds for bootstrap distribution."""

    if not values:
        return 0.0, 0.0
    alpha = 1 - confidence
    lower = float(np.percentile(values, 100 * alpha / 2))
    upper = float(np.percentile(values, 100 * (1 - alpha / 2)))
    return lower, upper


def _summarize_bootstrap(
    values: Sequence[float],
    point_estimate: float,
    confidence: float,
) -> Dict[str, float]:
    """Summarize bootstrap samples into CI stats."""

    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    ci_lower, ci_upper = _percentile_bounds(values, confidence)
    return {
        "mean": float(point_estimate),
        "std": std,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def _bootstrap_ci_for_values(
    values: Sequence[float],
    point_estimate: float,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> Dict[str, float]:
    """Bootstrap CI for mean of scalar values."""

    if n_bootstrap <= 0:
        return {}
    if not values:
        return {
            "mean": float(point_estimate),
            "std": 0.0,
            "ci_lower": float(point_estimate),
            "ci_upper": float(point_estimate),
        }

    rng = random.Random(seed)
    n = len(values)
    boot_means: List[float] = []
    for _ in range(n_bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(_safe_div(sum(sample), n))

    return _summarize_bootstrap(boot_means, point_estimate, confidence)


def _bootstrap_metrics_for_records(
    records: Sequence[Dict[str, Any]],
    metric_keys: Sequence[str],
    point_metrics: Dict[str, Any],
    compute_fn: Callable[[Iterable[Dict[str, Any]]], Dict[str, Any]],
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> Dict[str, Dict[str, float]]:
    """Bootstrap CI for metrics computed from record samples."""

    if not records or n_bootstrap <= 0:
        return {}

    rng = random.Random(seed)
    n = len(records)
    values_map: Dict[str, List[float]] = {key: [] for key in metric_keys}

    for _ in range(n_bootstrap):
        sample = [records[rng.randrange(n)] for _ in range(n)]
        metrics = compute_fn(sample)
        for key in metric_keys:
            value = metrics.get(key)
            values_map[key].append(float(value) if value is not None else 0.0)

    ci = {}
    for key in metric_keys:
        point = float(point_metrics.get(key, 0.0))
        ci[key] = _summarize_bootstrap(values_map[key], point, confidence)
    return ci


def _aggregate_correctness(values: Sequence[bool], strategy: str) -> bool:
    """Aggregate per-prompt correctness into a single image-level label."""

    if not values:
        return False
    if strategy == "all":
        return all(values)
    if strategy == "any":
        return any(values)
    return _safe_div(sum(values), len(values)) >= 0.5


def _presence_prompt_correct(rec: Mapping[str, Any]) -> bool:
    """Return whether a single presence record is correct."""

    target_opt = rec.get("target_answer")
    pred_opt = rec.get("prediction_parsed")
    if target_opt not in OPTION_TO_LABEL:
        return False
    return pred_opt == target_opt


def _segments_prompt_correct(rec: Mapping[str, Any]) -> bool:
    """Return whether a single segments record is correct (exact label match)."""

    gt = {l for l in rec.get("target_labels") or [] if l and l != "none"}
    pred = {l for l in rec.get("prediction_labels") or [] if l and l != "none"}
    return pred == gt


def _image_level_correctness(
    records: Iterable[Dict[str, Any]],
    task_name: str,
    *,
    strategy: str,
) -> Dict[int, bool]:
    """Aggregate correctness per image for a given task."""

    groups: Dict[int, List[bool]] = {}
    for rec in records:
        if rec.get("task_name") != task_name:
            continue
        image_id = rec.get("image_id")
        if image_id is None:
            continue
        if task_name == "recognition":
            correct = _presence_prompt_correct(rec)
        else:
            correct = _segments_prompt_correct(rec)
        groups.setdefault(int(image_id), []).append(bool(correct))

    return {
        image_id: _aggregate_correctness(values, strategy)
        for image_id, values in groups.items()
        if values
    }


def _image_id_to_filename(queries: Iterable[Dict[str, Any]]) -> Dict[int, str]:
    """Map image_id to basename filename."""

    mapping: Dict[int, str] = {}
    for rec in queries:
        image_id = rec.get("image_id")
        image_path = rec.get("image_path")
        if image_id is None or not image_path:
            continue
        mapping[int(image_id)] = Path(image_path).name
    return mapping


def _prosthesis_correctness_by_filename(
    predictions: Iterable[Dict[str, Any]],
    items: Mapping[str, Dict[str, Any]],
    *,
    strategy: str,
) -> Dict[str, bool]:
    """Aggregate Prosthesis Matching correctness per image filename."""

    groups: Dict[str, List[bool]] = {}
    for pred in predictions:
        item_id = pred.get("item_id")
        if item_id not in items:
            continue
        item = items[item_id]
        image_path = item.get("person_image", {}).get("path")
        if not image_path:
            continue
        filename = Path(image_path).name

        parse_error = pred.get("parse_error", 0)
        if parse_error:
            correct = False
        else:
            pred_choices = pred.get("choices", [])
            if not pred_choices and pred.get("choice"):
                pred_choices = [pred["choice"]]
            pred_set = set(pred_choices or [])
            correct_set = set(item.get("answer", {}).get("correct_option_ids") or [])
            correct = pred_set == correct_set

        groups.setdefault(filename, []).append(bool(correct))

    return {
        filename: _aggregate_correctness(values, strategy)
        for filename, values in groups.items()
        if values
    }


def _pairwise_counts(
    a_values: Mapping[str | int, bool],
    b_values: Mapping[str | int, bool],
) -> Dict[str, int]:
    """Compute 2x2 counts for two binary variables."""

    keys = set(a_values) & set(b_values)
    both_correct = both_wrong = a_correct_b_wrong = a_wrong_b_correct = 0
    for key in keys:
        a_val = bool(a_values[key])
        b_val = bool(b_values[key])
        if a_val and b_val:
            both_correct += 1
        elif a_val and not b_val:
            a_correct_b_wrong += 1
        elif not a_val and b_val:
            a_wrong_b_correct += 1
        else:
            both_wrong += 1
    return {
        "n": len(keys),
        "both_correct": both_correct,
        "a_correct_b_wrong": a_correct_b_wrong,
        "a_wrong_b_correct": a_wrong_b_correct,
        "both_wrong": both_wrong,
    }


def _phi_from_counts(counts: Mapping[str, int]) -> float:
    """Compute phi coefficient for a 2x2 table."""

    a = counts.get("both_correct", 0)
    b = counts.get("a_correct_b_wrong", 0)
    c = counts.get("a_wrong_b_correct", 0)
    d = counts.get("both_wrong", 0)
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    if denom == 0:
        return 0.0
    return (a * d - b * c) / math.sqrt(denom)


def _mutual_info_bits(counts: Mapping[str, int]) -> float:
    """Compute mutual information (bits) for a 2x2 table."""

    n = counts.get("n", 0)
    if n == 0:
        return 0.0
    a = counts.get("both_correct", 0)
    b = counts.get("a_correct_b_wrong", 0)
    c = counts.get("a_wrong_b_correct", 0)
    d = counts.get("both_wrong", 0)

    pa = (a + b) / n
    pb = (c + d) / n
    pc = (a + c) / n
    pd = (b + d) / n

    mi = 0.0
    for count, px, py in (
        (a, pa, pc),
        (b, pa, pd),
        (c, pb, pc),
        (d, pb, pd),
    ):
        if count == 0:
            continue
        pxy = count / n
        mi += pxy * math.log(pxy / (px * py), 2)
    return mi


def _conditional_wrong_rate(counts: Mapping[str, int]) -> float | None:
    """Compute P(B wrong | A wrong) from counts."""

    a_wrong = counts.get("a_wrong_b_correct", 0) + counts.get("both_wrong", 0)
    if a_wrong == 0:
        return None
    return _safe_div(counts.get("both_wrong", 0), a_wrong)


def _task_stats(values: Mapping[str | int, bool]) -> Dict[str, int]:
    """Summarize correct/wrong counts for a task."""

    correct = sum(1 for v in values.values() if v)
    wrong = sum(1 for v in values.values() if not v)
    return {"num_images": len(values), "correct": correct, "wrong": wrong}


def _conditional_matrix(
    task_values: Mapping[str, Mapping[str | int, bool]],
) -> Dict[str, Any]:
    """Build conditional wrong matrix for multiple tasks."""

    tasks = list(task_values.keys())
    matrix: Dict[str, Dict[str, float | None]] = {t: {} for t in tasks}
    counts: Dict[str, Dict[str, Dict[str, int]]] = {t: {} for t in tasks}

    for task_a in tasks:
        for task_b in tasks:
            if task_a == task_b:
                matrix[task_a][task_b] = 1.0 if task_values[task_a] else None
                counts[task_a][task_b] = {
                    "n": len(task_values[task_a]),
                    "both_wrong": sum(1 for v in task_values[task_a].values() if not v),
                }
                continue
            pair_counts = _pairwise_counts(task_values[task_a], task_values[task_b])
            matrix[task_a][task_b] = _conditional_wrong_rate(pair_counts)
            counts[task_a][task_b] = pair_counts

    return {"tasks": tasks, "matrix": matrix, "counts": counts}


def _pairwise_analysis(
    task_values: Mapping[str, Mapping[str | int, bool]],
) -> Dict[str, Dict[str, Any]]:
    """Compute pairwise phi/MI stats for task correctness."""

    tasks = list(task_values.keys())
    results: Dict[str, Dict[str, Any]] = {}
    for i, task_a in enumerate(tasks):
        for task_b in tasks[i + 1 :]:
            counts = _pairwise_counts(task_values[task_a], task_values[task_b])
            results[f"{task_a}_vs_{task_b}"] = {
                "n": counts.get("n", 0),
                "counts": counts,
                "phi": _phi_from_counts(counts),
                "mutual_info_bits": _mutual_info_bits(counts),
            }
    return results


def compute_instance_analysis(
    results: Sequence[Dict[str, Any]],
    *,
    queries: Sequence[Dict[str, Any]] | None = None,
    prosthesis_predictions: Sequence[Dict[str, Any]] | None = None,
    prosthesis_items: Mapping[str, Dict[str, Any]] | None = None,
    correctness_strategy: str = "majority",
) -> Dict[str, Any]:
    """Compute instance-level conditional errors and correlations."""

    presence_by_image = _image_level_correctness(
        results, "presence", strategy=correctness_strategy
    )
    segments_by_image = _image_level_correctness(
        results, "segments", strategy=correctness_strategy
    )

    analysis: Dict[str, Any] = {
        "aggregation": {
            "strategy": correctness_strategy,
            "segments_correctness": "exact_match",
            "prosthesis_correctness": "exact_match",
        },
        "presence_segments": {
            "task_stats": {
                "presence": _task_stats(presence_by_image),
                "segments": _task_stats(segments_by_image),
            },
            "conditional_wrong": _conditional_matrix(
                {"presence": presence_by_image, "segments": segments_by_image}
            ),
            "pairwise": _pairwise_analysis(
                {"presence": presence_by_image, "segments": segments_by_image}
            ),
        },
    }

    if prosthesis_predictions and prosthesis_items and queries:
        image_id_to_filename = _image_id_to_filename(queries)
        presence_by_filename = {
            image_id_to_filename[img_id]: val
            for img_id, val in presence_by_image.items()
            if img_id in image_id_to_filename
        }
        segments_by_filename = {
            image_id_to_filename[img_id]: val
            for img_id, val in segments_by_image.items()
            if img_id in image_id_to_filename
        }
        prosthesis_by_filename = _prosthesis_correctness_by_filename(
            prosthesis_predictions,
            prosthesis_items,
            strategy=correctness_strategy,
        )

        task_values = {
            "presence": presence_by_filename,
            "segments": segments_by_filename,
            "prosthesis_match": prosthesis_by_filename,
        }
        analysis["presence_segments_prosthesis"] = {
            "task_stats": {
                "presence": _task_stats(presence_by_filename),
                "segments": _task_stats(segments_by_filename),
                "prosthesis_match": _task_stats(prosthesis_by_filename),
            },
            "conditional_wrong": _conditional_matrix(task_values),
            "pairwise": _pairwise_analysis(task_values),
        }

    return analysis


def compute_presence_consistency(
    records: Iterable[Dict[str, Any]],
    *,
    require_correct: bool = True,
) -> Dict[str, Any]:
    """Compute consistency across multiple queries for the same image (presence)."""

    scores, num_groups_excluded = _presence_consistency_scores(
        records,
        require_correct=require_correct,
    )
    return _summarize_consistency_scores(
        scores,
        num_groups_excluded=num_groups_excluded,
    )


def compute_segments_consistency(
    records: Iterable[Dict[str, Any]],
    *,
    require_correct: bool = True,
) -> Dict[str, Any]:
    """Compute consistency across multiple queries for the same image (segments)."""

    scores, num_groups_excluded = _segments_consistency_scores(
        records,
        require_correct=require_correct,
    )
    return _summarize_consistency_scores(
        scores,
        num_groups_excluded=num_groups_excluded,
    )


def build_metrics(
    results: Iterable[Dict[str, Any]],
    *,
    compute_bootstrap: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    queries: Sequence[Dict[str, Any]] | None = None,
    prosthesis_predictions: Sequence[Dict[str, Any]] | None = None,
    prosthesis_items: Mapping[str, Dict[str, Any]] | None = None,
    correctness_strategy: str = "majority",
) -> Dict[str, Any]:
    """Build metrics payload for a model."""

    results_list = list(results)
    model_name = None
    if results_list:
        model_name = results_list[0].get("model_name")

    presence_metrics = compute_presence_metrics(results_list)
    segments_metrics = compute_segments_metrics(results_list)
    presence_consistency = compute_presence_consistency(results_list)
    segments_consistency = compute_segments_consistency(results_list)

    payload = {
        "model_name": model_name,
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_part": BENCHMARK_PART,
        "presence": presence_metrics,
        "segments": segments_metrics,
        "presence_consistency": presence_consistency,
        "segments_consistency": segments_consistency,
    }
    if compute_bootstrap and bootstrap_samples > 0:
        presence_records = [rec for rec in results_list if rec.get("task_name") == "recognition"]
        segments_records = [rec for rec in results_list if rec.get("task_name") == "attribution"]

        presence_ci = _bootstrap_metrics_for_records(
            presence_records,
            metric_keys=[
                "accuracy",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "parse_error_rate",
            ],
            point_metrics=presence_metrics,
            compute_fn=compute_presence_metrics,
            n_bootstrap=bootstrap_samples,
            confidence=confidence_level,
            seed=bootstrap_seed,
        )
        if presence_ci:
            presence_metrics["ci"] = presence_ci

        segments_ci = _bootstrap_metrics_for_records(
            segments_records,
            metric_keys=[
                "mean_iou",
                "micro_precision",
                "micro_recall",
                "micro_f1",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "parse_error_rate",
            ],
            point_metrics=segments_metrics,
            compute_fn=compute_segments_metrics,
            n_bootstrap=bootstrap_samples,
            confidence=confidence_level,
            seed=bootstrap_seed,
        )
        if segments_ci:
            segments_metrics["ci"] = segments_ci

        presence_scores, _ = _presence_consistency_scores(results_list)
        presence_consistency_ci = _bootstrap_ci_for_values(
            presence_scores,
            presence_consistency.get("mean_consistency", 0.0),
            n_bootstrap=bootstrap_samples,
            confidence=confidence_level,
            seed=bootstrap_seed,
        )
        if presence_consistency_ci:
            presence_consistency["ci"] = {"mean_consistency": presence_consistency_ci}

        segments_scores, _ = _segments_consistency_scores(results_list)
        segments_consistency_ci = _bootstrap_ci_for_values(
            segments_scores,
            segments_consistency.get("mean_consistency", 0.0),
            n_bootstrap=bootstrap_samples,
            confidence=confidence_level,
            seed=bootstrap_seed,
        )
        if segments_consistency_ci:
            segments_consistency["ci"] = {"mean_consistency": segments_consistency_ci}

        payload["bootstrap"] = {
            "samples": bootstrap_samples,
            "confidence": confidence_level,
            "seed": bootstrap_seed,
        }

    instance_analysis = compute_instance_analysis(
        results_list,
        queries=queries,
        prosthesis_predictions=prosthesis_predictions,
        prosthesis_items=prosthesis_items,
        correctness_strategy=correctness_strategy,
    )
    payload["instance_analysis"] = instance_analysis

    return payload
