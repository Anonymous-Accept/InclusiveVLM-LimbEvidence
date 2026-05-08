"""Build a unified markdown/json summary for control experiments."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.core.logging_utils import configure_logging
from scripts.inclusive_vlm_lep.controls._shared import (
    iter_json_files,
    load_json_if_exists,
    save_report,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "controls"


def _is_baseline(name: str) -> bool:
    return name.startswith("baseline_")


def _metric_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _get_or_create_row(
    rows: dict[tuple[str, str], dict[str, Any]],
    *,
    variant: str,
    model_name: str,
) -> dict[str, Any]:
    key = (variant or "unknown_variant", model_name or "unknown_model")
    if key not in rows:
        rows[key] = {
            "variant": key[0],
            "model_name": key[1],
            "row_type": "baseline" if _is_baseline(key[1]) else "model",
        }
    return rows[key]


def _ingest_metrics_json(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> bool:
    data = load_json_if_exists(path)
    if not isinstance(data, dict):
        return False
    overall = data.get("overall")
    if not isinstance(overall, dict):
        return False

    metadata = data.get("metadata") or {}
    variant = (
        data.get("variant")
        or data.get("task_variant")
        or metadata.get("variant")
        or path.parent.name
    )
    model_name = (
        data.get("model_name")
        or metadata.get("model_name")
        or path.stem
    )
    row = _get_or_create_row(rows, variant=variant, model_name=model_name)
    for key in (
        "exact_accuracy",
        "partial_accuracy",
        "avg_precision",
        "avg_recall",
        "avg_f1",
        "parse_error_rate",
    ):
        if key in overall:
            row[key] = overall.get(key)
    row.setdefault("sources", []).append(str(path))
    return True


def _ingest_parse_coverage_json(
    path: Path,
    rows: dict[tuple[str, str], dict[str, Any]],
) -> bool:
    data = load_json_if_exists(path)
    if not isinstance(data, dict) or data.get("report_type") != "parse_coverage":
        return False

    variant = str(data.get("variant") or "unknown_variant")
    for model_name, metrics in (data.get("per_model") or {}).items():
        row = _get_or_create_row(rows, variant=variant, model_name=str(model_name))
        for key in (
            "parse_success_rate",
            "unresolved_token_rate",
            "hallucination_rate",
            "omission_rate",
        ):
            if key in metrics:
                row[key] = metrics.get(key)
        row.setdefault("sources", []).append(str(path))
    return True


def _ingest_shuffle_json(path: Path, rows: dict[tuple[str, str], dict[str, Any]]) -> bool:
    data = load_json_if_exists(path)
    if not isinstance(data, dict) or data.get("analysis_type") != "option_order_shuffle":
        return False

    variant = str(data.get("variant") or "unknown_variant")
    for model_name, report in (data.get("per_model") or {}).items():
        summary = report.get("summary") or {}
        row = _get_or_create_row(rows, variant=variant, model_name=str(model_name))
        row["shuffle_selected_set_jaccard_mean"] = summary.get("selected_set_jaccard_mean")
        row["shuffle_exact_accuracy_std_mean"] = summary.get("exact_accuracy_std_mean")
        row["shuffle_avg_f1_std_mean"] = summary.get("avg_f1_std_mean")
        row.setdefault("sources", []).append(str(path))
    return True


def _build_markdown(rows_by_variant: dict[str, list[dict[str, Any]]]) -> str:
    lines = [
        "# InclusiveVLM-LEP Control Summary",
        "",
        "How to read:",
        "- `baseline_random` samples uniformly from the valid answer space.",
        "- `baseline_label_freq_prior` reuses only empirical label popularity on the evaluated subset.",
        "- `baseline_answer_count_prior` matches only the dominant gold answer count.",
        "- `baseline_always_none` is the abstain floor.",
        "- `baseline_top_k_frequent_tags` is an attribution-only fixed-tag prior.",
        "- Real models should beat these floors while remaining stable under shuffles and parser diagnostics.",
        "",
    ]

    if not rows_by_variant:
        rows_by_variant = {
            "Attribution-FreeForm": [
                {
                    "model_name": "<model-or-baseline>",
                    "row_type": "template",
                }
            ],
            "Compatibility-Category": [
                {
                    "model_name": "<model-or-baseline>",
                    "row_type": "template",
                }
            ],
        }

    for variant in sorted(rows_by_variant):
        lines.extend(
            [
                f"## {variant}",
                "",
                "| System | Type | Exact | Partial | Precision | Recall | F1 | ParseErr | ParseSuccess | Hallucination | Omission | ShuffleJaccard | F1-Stability |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        sorted_rows = sorted(
            rows_by_variant[variant],
            key=lambda row: (0 if row["row_type"] == "baseline" else 1, row["model_name"]),
        )
        for row in sorted_rows:
            lines.append(
                "| {model_name} | {row_type} | {exact_accuracy} | {partial_accuracy} | "
                "{avg_precision} | {avg_recall} | {avg_f1} | {parse_error_rate} | "
                "{parse_success_rate} | {hallucination_rate} | {omission_rate} | "
                "{shuffle_selected_set_jaccard_mean} | {shuffle_avg_f1_std_mean} |".format(
                    model_name=row["model_name"],
                    row_type=row["row_type"],
                    exact_accuracy=_metric_cell(row.get("exact_accuracy")),
                    partial_accuracy=_metric_cell(row.get("partial_accuracy")),
                    avg_precision=_metric_cell(row.get("avg_precision")),
                    avg_recall=_metric_cell(row.get("avg_recall")),
                    avg_f1=_metric_cell(row.get("avg_f1")),
                    parse_error_rate=_metric_cell(row.get("parse_error_rate")),
                    parse_success_rate=_metric_cell(row.get("parse_success_rate")),
                    hallucination_rate=_metric_cell(row.get("hallucination_rate")),
                    omission_rate=_metric_cell(row.get("omission_rate")),
                    shuffle_selected_set_jaccard_mean=_metric_cell(
                        row.get("shuffle_selected_set_jaccard_mean")
                    ),
                    shuffle_avg_f1_std_mean=_metric_cell(
                        row.get("shuffle_avg_f1_std_mean")
                    ),
                )
            )
        lines.append("")
    return "\n".join(lines)


def build_control_summary(
    *,
    metrics_roots: list[Path],
    shuffle_analysis_roots: list[Path],
    parse_coverage_roots: list[Path],
    output_dir: Path = DEFAULT_OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Build markdown and JSON summary artifacts."""

    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for path in iter_json_files(metrics_roots):
        _ingest_metrics_json(path, rows)
    for path in iter_json_files(shuffle_analysis_roots):
        _ingest_shuffle_json(path, rows)
    for path in iter_json_files(parse_coverage_roots):
        _ingest_parse_coverage_json(path, rows)

    rows_by_variant: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        rows_by_variant.setdefault(row["variant"], []).append(row)
    if not rows_by_variant:
        rows_by_variant = {
            "Attribution-FreeForm": [
                {
                    "variant": "Attribution-FreeForm",
                    "model_name": "<model-or-baseline>",
                    "row_type": "template",
                }
            ],
            "Compatibility-Category": [
                {
                    "variant": "Compatibility-Category",
                    "model_name": "<model-or-baseline>",
                    "row_type": "template",
                }
            ],
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "control_summary.md"
    json_path = output_dir / "control_summary.json"

    markdown_text = _build_markdown(rows_by_variant)
    markdown_path.write_text(markdown_text, encoding="utf-8")
    save_report(
        {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "task_variants": rows_by_variant,
        },
        json_path,
    )
    logger.info("Wrote markdown summary to %s", markdown_path)
    logger.info("Wrote JSON summary to %s", json_path)
    return markdown_path, json_path


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Directories or JSON files containing evaluator metrics.",
    )
    parser.add_argument(
        "--shuffle-analysis-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Directories or JSON files containing shuffle sensitivity analysis.",
    )
    parser.add_argument(
        "--parse-coverage-roots",
        type=Path,
        nargs="*",
        default=[],
        help="Directories or JSON files containing parse coverage reports.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for control_summary.{md,json}.",
    )
    args = parser.parse_args()

    configure_logging()
    build_control_summary(
        metrics_roots=list(args.metrics_roots),
        shuffle_analysis_roots=list(args.shuffle_analysis_roots),
        parse_coverage_roots=list(args.parse_coverage_roots),
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
