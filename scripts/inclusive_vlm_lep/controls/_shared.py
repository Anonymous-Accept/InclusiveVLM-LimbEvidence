"""Shared helpers for answer-space control experiments."""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, TextIO

from scripts.core.io_utils import load_json, load_jsonl, save_json

logger = logging.getLogger(__name__)

NONE_LABEL_ALIASES = {
    "",
    "abstain",
    "cannot be determined",
    "empty",
    "n/a",
    "no visible cue",
    "none",
    "none of the above",
    "none_visible",
    "not visible",
    "unsure",
}

CANONICAL_OPTION_LABEL_KEYS = (
    "canonical_label",
    "canonical_tag",
    "canonical_id",
    "coarse_part",
    "label",
    "tag",
    "name",
)


def load_subset(path: Path) -> list[dict[str, Any]]:
    """Load subset items from JSONL."""

    return load_jsonl(path)


def load_predictions(path: Path) -> list[dict[str, Any]]:
    """Load prediction records from JSONL."""

    return load_jsonl(path)


def save_report(data: dict[str, Any], path: Path) -> None:
    """Save a JSON report."""

    save_json(data, path, indent=2, ensure_ascii=False)


def open_jsonl_writer(path: Path) -> TextIO:
    """Open a JSONL writer and create missing parents."""

    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8")


def append_jsonl_line(handle: TextIO, record: dict[str, Any]) -> None:
    """Write one JSONL record and flush immediately."""

    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def get_record_id(item: dict[str, Any]) -> str:
    """Return the stable record identifier."""

    return str(item.get("record_id") or item.get("item_id") or "")


def get_variant(item: dict[str, Any]) -> str:
    """Return variant/task name for one item."""

    return str(
        item.get("variant")
        or item.get("task_variant")
        or item.get("answer_space_variant")
        or item.get("task_name")
        or item.get("source_task")
        or ""
    )


def infer_answer_space_type(item: dict[str, Any]) -> str:
    """Infer whether one item is constrained or free-form."""

    declared = item.get("answer_space_type")
    if declared:
        return str(declared).strip().lower()
    if item.get("option_count") is not None:
        return "constrained"
    if item.get("options"):
        return "constrained"
    return "freeform"


def get_correct_option_ids(item: dict[str, Any]) -> list[str]:
    """Return gold constrained option ids."""

    answer = item.get("answer")
    if isinstance(answer, dict) and answer.get("correct_option_ids") is not None:
        return [str(v) for v in answer.get("correct_option_ids") or []]
    return [str(v) for v in item.get("correct_option_ids") or []]


def get_canonical_target_set(item: dict[str, Any]) -> list[str]:
    """Return the canonical gold label set for free-form items."""

    value = (
        item.get("canonical_target_set")
        or item.get("target_labels")
        or item.get("canonical_labels")
        or item.get("gold_labels")
        or []
    )
    return [str(v) for v in value if str(v)]


def get_gold_count(item: dict[str, Any]) -> int:
    """Return gold answer count for one item."""

    explicit = item.get("gold_count")
    if explicit is not None:
        return int(explicit)
    answer_space_type = infer_answer_space_type(item)
    if answer_space_type == "constrained":
        return len(get_correct_option_ids(item))
    return len(get_canonical_target_set(item))


def get_options(item: dict[str, Any]) -> list[dict[str, Any]]:
    """Return normalized option dicts."""

    options = item.get("options") or []
    if not isinstance(options, list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, option in enumerate(options):
        if isinstance(option, dict):
            option_id = option.get("option_id") or option.get("id")
            if option_id is None:
                option_id = chr(ord("A") + index)
            normalized.append({**option, "option_id": str(option_id)})
        else:
            normalized.append(
                {
                    "option_id": chr(ord("A") + index),
                    "text": str(option),
                }
            )
    return normalized


def get_option_ids(item: dict[str, Any]) -> list[str]:
    """Return valid option ids for one constrained item."""

    options = get_options(item)
    if options:
        return [str(option["option_id"]) for option in options]

    option_count = item.get("option_count")
    if option_count is None:
        return []
    return [chr(ord("A") + idx) for idx in range(int(option_count))]


def _clean_none_candidate(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_")
    return " ".join(text.split())


def find_none_option_id(item: dict[str, Any]) -> str | None:
    """Return the option id that represents none/abstain if present."""

    for option in get_options(item):
        for key in ("canonical_label", "label", "tag", "name", "text", "description"):
            value = option.get(key)
            if value is None:
                continue
            if _clean_none_candidate(value) in NONE_LABEL_ALIASES:
                return str(option["option_id"])
    return None


def infer_canonical_label_for_option(option: dict[str, Any]) -> str | None:
    """Extract a stable option-level semantic label if available."""

    for key in CANONICAL_OPTION_LABEL_KEYS:
        value = option.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def infer_global_canonical_vocab(items: Iterable[dict[str, Any]]) -> list[str]:
    """Infer canonical label vocabulary from one subset."""

    labels: set[str] = set()
    for item in items:
        for label in get_canonical_target_set(item):
            labels.add(label)
        for label in item.get("canonical_label_space") or []:
            labels.add(str(label))
        for option in get_options(item):
            label = infer_canonical_label_for_option(option)
            if label:
                labels.add(label)
    return sorted(label for label in labels if label)


def get_item_canonical_label_space(
    item: dict[str, Any],
    global_vocab: list[str],
) -> list[str]:
    """Return valid canonical labels for one free-form item."""

    explicit = item.get("canonical_label_space") or item.get("valid_canonical_labels")
    if explicit:
        return [str(v) for v in explicit if str(v)]
    return list(global_vocab)


def compute_mode_int(values: Iterable[int]) -> int:
    """Return deterministic integer mode using smallest-value tie break."""

    counter = Counter(int(v) for v in values)
    if not counter:
        return 0
    highest = max(counter.values())
    return min(value for value, count in counter.items() if count == highest)


def sample_uniform_subset(
    labels: list[str],
    rng: random.Random,
    *,
    allow_empty: bool,
) -> list[str]:
    """Sample uniformly from the subset power set."""

    unique = list(dict.fromkeys(labels))
    if not unique:
        return []
    lower = 0 if allow_empty else 1
    size = rng.randint(lower, len(unique))
    if size == 0:
        return []
    return sorted(rng.sample(unique, size))


def sample_k_without_replacement(
    labels: list[str],
    k: int,
    rng: random.Random,
) -> list[str]:
    """Sample up to k labels uniformly without replacement."""

    unique = list(dict.fromkeys(labels))
    if not unique or k <= 0:
        return []
    return sorted(rng.sample(unique, min(k, len(unique))))


def compute_label_frequency(
    items: Iterable[dict[str, Any]],
    *,
    answer_space_type: str,
) -> Counter[str]:
    """Count empirical gold label frequency on the subset itself."""

    counts: Counter[str] = Counter()
    for item in items:
        if answer_space_type == "constrained":
            counts.update(get_correct_option_ids(item))
        else:
            counts.update(get_canonical_target_set(item))
    return counts


def top_labels_from_counts(counts: Counter[str], k: int) -> list[str]:
    """Return deterministic top-k labels sorted by frequency then label."""

    ordered = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [label for label, _ in ordered[: max(k, 0)]]


def classify_subset(items: list[dict[str, Any]]) -> str:
    """Return the dominant answer-space type in one subset."""

    types = {infer_answer_space_type(item) for item in items}
    if len(types) == 1:
        return next(iter(types))
    raise ValueError(f"Mixed subset answer-space types are not supported: {sorted(types)}")


def is_attribution_variant(variant: str) -> bool:
    """Heuristic check for attribution-family variants."""

    return "attribution" in variant.strip().lower()


def is_compatibility_variant(variant: str) -> bool:
    """Heuristic check for compatibility-family variants."""

    value = variant.strip().lower()
    return "compatibility" in value or "prosthesis" in value


def make_prediction_record(
    *,
    item: dict[str, Any],
    model_name: str,
    seed: int | None,
    selected_labels: list[str],
) -> dict[str, Any]:
    """Build a prediction record compatible with constrained and free-form flows."""

    record: dict[str, Any] = {
        "record_id": get_record_id(item),
        "item_id": get_record_id(item),
        "variant": get_variant(item),
        "model_name": model_name,
        "seed": seed,
        "answer_space_type": infer_answer_space_type(item),
        "answer_count_predicted": len(selected_labels),
        "parse_error": 0,
    }

    if infer_answer_space_type(item) == "constrained":
        record["selected_option_ids"] = list(selected_labels)
        record["choices"] = list(selected_labels)
    else:
        record["selected_canonical_labels"] = list(selected_labels)
        record["normalized_predictions"] = list(selected_labels)
        record["prediction_labels"] = list(selected_labels)
        record["unresolved_tokens"] = []
        record["parser_status"] = "ok"
    return record


def extract_prediction_labels(record: dict[str, Any]) -> list[str]:
    """Extract normalized selected labels from one prediction record."""

    if record.get("answer_space_type") == "constrained":
        values = (
            record.get("selected_option_ids")
            or record.get("choices")
            or record.get("prediction_option_ids")
            or []
        )
    else:
        values = (
            record.get("selected_canonical_labels")
            or record.get("normalized_predictions")
            or record.get("prediction_labels")
            or []
        )
    return [str(value) for value in values if str(value)]


def compute_set_metrics(
    gold: set[str],
    pred: set[str],
) -> dict[str, float]:
    """Compute standard set prediction metrics for one item."""

    true_positives = len(gold & pred)
    precision = true_positives / len(pred) if pred else (1.0 if not gold else 0.0)
    recall = true_positives / len(gold) if gold else (1.0 if not pred else 0.0)
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    exact = 1.0 if pred == gold else 0.0
    partial = 1.0 if true_positives > 0 or (not gold and not pred) else 0.0
    return {
        "exact_accuracy": exact,
        "partial_accuracy": partial,
        "avg_precision": precision,
        "avg_recall": recall,
        "avg_f1": f1,
    }


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    """Load JSON if the path exists."""

    if not path.exists():
        return None
    return load_json(path)


def iter_json_files(roots: Iterable[Path]) -> Iterator[Path]:
    """Yield JSON files recursively under a list of roots."""

    seen: set[Path] = set()
    for root in roots:
        if root.is_file():
            if root.suffix == ".json" and root not in seen:
                seen.add(root)
                yield root
            continue
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.json")):
            if path not in seen:
                seen.add(path)
                yield path
