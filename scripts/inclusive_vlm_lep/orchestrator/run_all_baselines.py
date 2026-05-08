"""Run trivial baselines for the InclusiveVLM-LEP paper-facing variants."""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from scripts.core.io_utils import load_jsonl, save_json
from scripts.core.logging_utils import configure_logging, get_logger

from ._shared import (
    DEFAULT_BASELINES_ROOT,
    DEFAULT_SUBSET_BY_VARIANT,
    FREEFORM_POLICY_VERSION,
    VARIANT_ORDER,
    recompute_variant_metrics,
    write_metrics_bundle,
)

LOGGER = get_logger(__name__)

BASELINES = [
    "random",
    "label_freq_prior",
    "answer_count_prior",
    "always_none",
    "top_k_frequent_tags",
]


def _mode(values: list[int]) -> int:
    counts = Counter(values)
    if not counts:
        return 0
    max_count = max(counts.values())
    return min(value for value, count in counts.items() if count == max_count)


def _write_predictions(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _presence_predictions(
    subset: list[dict[str, Any]],
    baseline: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    options = ["a", "b", "c", "d"]
    target_counts = Counter(str(item.get("target_answer") or "").strip().lower() for item in subset)
    top_label = sorted(target_counts.items(), key=lambda pair: (-pair[1], pair[0]))[0][0]
    records: list[dict[str, Any]] = []
    for item in subset:
        if baseline == "random":
            pred = rng.choice(options)
        elif baseline == "label_freq_prior":
            pred = top_label
        elif baseline == "answer_count_prior":
            pred = options[0]
        elif baseline == "always_none":
            pred = None
        else:
            raise ValueError(f"Incompatible baseline for presence: {baseline}")
        records.append(
            {
                "model_name": baseline,
                "task_name": "recognition",
                "image_id": item.get("image_id"),
                "group_id": item.get("group_id"),
                "prompt_id": item.get("prompt_id"),
                "prediction_parsed": pred,
                "prediction_raw": "" if pred is None else pred,
            }
        )
    return records


def _top_k_labels(subset: list[dict[str, Any]], k: int) -> list[str]:
    counts: Counter[str] = Counter()
    for item in subset:
        counts.update(str(label) for label in item.get("target_labels", []) if str(label))
        counts.update(
            str(label) for label in item.get("canonical_target_set", []) if str(label)
        )
    ranked = sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))
    return [label for label, _ in ranked[:k]]


def _attribution_constrained_predictions(
    subset: list[dict[str, Any]],
    baseline: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    vocab = sorted({str(label) for item in subset for label in item.get("label_vocab", [])})
    mode_k = max(_mode([len(item.get("target_labels", [])) for item in subset]), 1)
    top_labels = _top_k_labels(subset, mode_k)
    records: list[dict[str, Any]] = []
    for item in subset:
        label_vocab = [str(label) for label in item.get("label_vocab", [])]
        if baseline == "random":
            size = rng.randint(0, len(label_vocab))
            pred_labels = sorted(rng.sample(label_vocab, size)) if size else []
        elif baseline == "label_freq_prior":
            pred_labels = [label for label in top_labels if label in set(label_vocab)]
        elif baseline == "answer_count_prior":
            pred_labels = label_vocab[:mode_k]
        elif baseline == "always_none":
            pred_labels = []
        else:
            raise ValueError(
                f"Incompatible baseline for attribution_constrained: {baseline}"
            )
        records.append(
            {
                "model_name": baseline,
                "task_name": "attribution",
                "image_id": item.get("image_id"),
                "group_id": item.get("group_id"),
                "prompt_id": item.get("prompt_id"),
                "prediction_labels": pred_labels,
                "prediction_raw": json.dumps({"labels": pred_labels}, ensure_ascii=False),
            }
        )
    return records


def _compatibility_predictions(
    subset: list[dict[str, Any]],
    baseline: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    mode_k = max(
        _mode([len(item.get("answer", {}).get("correct_option_ids", [])) for item in subset]),
        1,
    )
    counts: Counter[str] = Counter()
    for item in subset:
        counts.update(str(choice) for choice in item.get("answer", {}).get("correct_option_ids", []))
    ranked = [choice for choice, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))]
    records: list[dict[str, Any]] = []
    for item in subset:
        choices = [str(option.get("option_id")) for option in item.get("options", [])]
        if baseline == "random":
            size = rng.randint(0, len(choices))
            pred = sorted(rng.sample(choices, size)) if size else []
        elif baseline == "label_freq_prior":
            pred = [choice for choice in ranked if choice in set(choices)][:mode_k]
        elif baseline == "answer_count_prior":
            pred = choices[:mode_k]
        elif baseline == "always_none":
            pred = []
        else:
            raise ValueError(f"Incompatible baseline for compatibility: {baseline}")
        records.append(
            {
                "model_name": baseline,
                "item_id": item.get("item_id"),
                "choices": pred,
                "parse_error": 0,
            }
        )
    return records


def _freeform_predictions(
    subset: list[dict[str, Any]],
    baseline: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    mode_k = max(_mode([int(item.get("gold_count", 0)) for item in subset]), 1)
    top_labels = _top_k_labels(subset, max(mode_k, 3))
    global_vocab = sorted(
        {str(label) for item in subset for label in item.get("canonical_target_set", []) if str(label)}
    )
    records: list[dict[str, Any]] = []
    for item in subset:
        if baseline == "random":
            size = rng.randint(1, len(global_vocab)) if global_vocab else 0
            pred = sorted(rng.sample(global_vocab, size)) if size else []
        elif baseline == "label_freq_prior":
            pred = top_labels[:mode_k]
        elif baseline == "answer_count_prior":
            pred = global_vocab[:mode_k]
        elif baseline == "always_none":
            pred = []
        elif baseline == "top_k_frequent_tags":
            pred = top_labels[: min(2, len(top_labels))]
        else:
            raise ValueError(f"Unsupported freeform baseline: {baseline}")
        records.append(
            {
                "model_name": baseline,
                "record_id": item.get("record_id"),
                "raw_output": ", ".join(pred),
            }
        )
    return records


def _subset_for_variant(variant: str) -> list[dict[str, Any]]:
    path = DEFAULT_SUBSET_BY_VARIANT[variant]
    records = load_jsonl(path)
    if variant == "presence":
        return [record for record in records if str(record.get("task_name")) == "recognition"]
    if variant == "attribution_constrained":
        return [record for record in records if str(record.get("task_name")) == "attribution"]
    return records


def _baseline_records(
    variant: str,
    baseline: str,
    rng: random.Random,
) -> list[dict[str, Any]]:
    subset = _subset_for_variant(variant)
    if variant == "presence":
        return _presence_predictions(subset, baseline, rng)
    if variant == "attribution_constrained":
        return _attribution_constrained_predictions(subset, baseline, rng)
    if variant in {"compatibility_diversity", "compatibility_category"}:
        return _compatibility_predictions(subset, baseline, rng)
    if variant == "attribution_freeform":
        return _freeform_predictions(subset, baseline, rng)
    raise ValueError(f"Unsupported variant: {variant}")


def baseline_is_compatible(variant: str, baseline: str) -> bool:
    """Return whether a baseline should be run for a variant."""

    return not (baseline == "top_k_frequent_tags" and variant != "attribution_freeform")


def run_one_baseline(
    baseline: str,
    variant: str,
    output_dir: Path,
    *,
    seed: int,
) -> Path | None:
    """Run one baseline/variant pair and save metrics JSON."""

    if not baseline_is_compatible(variant, baseline):
        LOGGER.info("Skipping incompatible combo: %s x %s", baseline, variant)
        return None
    rng = random.Random(seed)
    records = _baseline_records(variant, baseline, rng)
    with tempfile.TemporaryDirectory(prefix="inclusive_vlm_lep_baseline_") as tmpdir:
        predictions_path = Path(tmpdir) / f"{baseline}_{variant}.predictions.jsonl"
        _write_predictions(predictions_path, records)
        subset_path = DEFAULT_SUBSET_BY_VARIANT[variant]
        payload, per_item_rows, per_group_rows = recompute_variant_metrics(
            variant,
            predictions_path,
            subset_path,
            consistency_gate=None,
            policy_version=FREEFORM_POLICY_VERSION,
        )
    payload["baseline_name"] = baseline
    output_path = output_dir / f"baseline_{baseline}_{variant}.metrics.json"
    write_metrics_bundle(output_path, payload, per_item_rows, per_group_rows)
    return output_path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_BASELINES_ROOT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for baseline in BASELINES:
        for variant in VARIANT_ORDER:
            output = run_one_baseline(
                baseline,
                variant,
                args.output_dir,
                seed=args.seed,
            )
            if output is not None:
                written += 1
    save_json(
        {
            "seed": args.seed,
            "written_metrics": written,
            "variants": VARIANT_ORDER,
            "baselines": BASELINES,
        },
        args.output_dir / "summary.json",
        indent=2,
        ensure_ascii=False,
    )
    LOGGER.info("Wrote %d baseline metrics files under %s", written, args.output_dir)


if __name__ == "__main__":
    main()
