"""Aggregate InclusiveVLM-LEP metrics, baselines, and findings into one markdown report."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from scripts.core.logging_utils import configure_logging, get_logger

from ._shared import (
    DEFAULT_BASELINES_ROOT,
    DEFAULT_FINDINGS_ROOT,
    DEFAULT_MAIN_RESULTS_PATH,
    DEFAULT_METRICS_ROOT,
    DISPLAY_NAME_BY_VARIANT,
    KEY_METRIC_BY_VARIANT,
    VARIANT_ORDER,
    best_model_for_variant,
    load_baseline_index,
    load_metrics_index,
    load_per_item_rows,
    markdown_table,
    metric_cell,
    write_markdown,
)

LOGGER = get_logger(__name__)


def _baseline_floor_rows(
    baseline_index: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    rows = []
    for baseline_name, by_variant in sorted(baseline_index.items()):
        row = {"model_name": f"baseline:{baseline_name}"}
        for variant in VARIANT_ORDER:
            payload = by_variant.get(variant)
            row[variant] = (
                metric_cell(payload.get("metrics", {}).get(KEY_METRIC_BY_VARIANT[variant]))
                if payload
                else "NA"
            )
        rows.append(row)
    return rows


def _model_rows(metrics_index: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for model_name, by_variant in sorted(metrics_index.items()):
        row = {"model_name": model_name}
        for variant in VARIANT_ORDER:
            payload = by_variant.get(variant)
            row[variant] = (
                metric_cell(payload.get("metrics", {}).get(KEY_METRIC_BY_VARIANT[variant]))
                if payload
                else "NA"
            )
        rows.append(row)
    return rows


def _finding_paragraph(
    finding_id: str,
    findings_root: Path,
) -> str:
    md_path = findings_root / f"finding_{finding_id}.md"
    return f"See `{md_path.name}` for the full table and figure artifacts."


def _write_failure_examples(
    output_path: Path,
    variant: str,
    model_name: str,
    metrics_payload: dict[str, Any],
) -> Path:
    metrics_json_path = Path(metrics_payload["_path"])
    rows = load_per_item_rows(metrics_json_path)
    wrong_rows = [row for row in rows if float(row.get("exact", 0.0)) < 1.0]
    wrong_rows = wrong_rows[:3]
    sidecar_path = output_path.with_name(
        f"{output_path.stem}_failures_{variant}.md"
    )
    lines = [f"# {DISPLAY_NAME_BY_VARIANT[variant]} failure cases", ""]
    lines.append(f"- top_model: `{model_name}`")
    lines.append(f"- source_metrics: `{metrics_json_path}`")
    lines.append("")
    if not wrong_rows:
        lines.append("No failure examples available.")
    else:
        for row in wrong_rows:
            lines.extend(
                [
                    f"## {row.get('item_id')}",
                    f"- group_id: `{row.get('group_id')}`",
                    f"- gold: `{row.get('gold_labels') or row.get('gold_answer')}`",
                    f"- prediction: `{row.get('pred_labels') or row.get('pred_answer')}`",
                    f"- exact: `{row.get('exact')}`",
                    "",
                ]
            )
    write_markdown(sidecar_path, "\n".join(lines) + "\n")
    return sidecar_path


def build_report(
    metrics_root: Path,
    baselines_root: Path,
    findings_root: Path,
    output_path: Path,
) -> str:
    """Build the markdown report and write failure-example sidecars."""

    metrics_index = load_metrics_index(metrics_root)
    baseline_index = load_baseline_index(baselines_root)
    model_rows = _model_rows(metrics_index)
    baseline_rows = _baseline_floor_rows(baseline_index)

    table_columns = [("model_name", "Model")] + [
        (variant, DISPLAY_NAME_BY_VARIANT[variant]) for variant in VARIANT_ORDER
    ]
    sections = [
        "# InclusiveVLM-LEP Main Results",
        "",
        "## Model Matrix",
        "",
        markdown_table(model_rows, table_columns),
        "",
        "## Baseline Floors",
        "",
        markdown_table(baseline_rows, table_columns),
        "",
        "## Findings Summary",
        "",
    ]
    finding_map = {
        "f1_persistent_failure": "Persistent failure remains visible across variants rather than collapsing under the easiest answer space. "
        + _finding_paragraph("f1_persistent_failure", findings_root),
        "f2_attribution_vs_presence": "Attribution remains systematically weaker than coarse presence recognition. "
        + _finding_paragraph("f2_attribution_vs_presence", findings_root),
        "f3_presence_attribution_coupling": "Presence correctness only partially transfers to attribution correctness at the group level. "
        + _finding_paragraph("f3_presence_attribution_coupling", findings_root),
        "f4_compatibility_plausible_vs_exact": "Partial compatibility often exceeds exact recovery, exposing plausibility without full-set correctness. "
        + _finding_paragraph("f4_compatibility_plausible_vs_exact", findings_root),
        "f5_consistency_vs_correctness": "Consistency and correctness diverge; consistent wrong behavior remains visible in the group cross-tabs. "
        + _finding_paragraph("f5_consistency_vs_correctness", findings_root),
    }
    for title, paragraph in finding_map.items():
        sections.extend([f"### {title}", "", paragraph, ""])

    sections.extend(["## Failure Cases", ""])
    for variant in VARIANT_ORDER:
        model_name, _ = best_model_for_variant(metrics_index, variant)
        if model_name is None:
            sections.extend([f"### {DISPLAY_NAME_BY_VARIANT[variant]}", "", "No model metrics available.", ""])
            continue
        payload = metrics_index[model_name].get(variant)
        if payload is None:
            continue
        sidecar = _write_failure_examples(output_path, variant, model_name, payload)
        sections.extend(
            [
                f"### {DISPLAY_NAME_BY_VARIANT[variant]}",
                "",
                f"Top model: `{model_name}`. Example failures: [{sidecar.name}]({sidecar.name})",
                "",
            ]
        )
    text = "\n".join(sections).rstrip() + "\n"
    write_markdown(output_path, text)
    return text


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--baselines-root", type=Path, default=DEFAULT_BASELINES_ROOT)
    parser.add_argument("--findings-root", type=Path, default=DEFAULT_FINDINGS_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_MAIN_RESULTS_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    report = build_report(
        args.metrics_root,
        args.baselines_root,
        args.findings_root,
        args.output,
    )
    if args.dry_run:
        print(report)
    LOGGER.info("Wrote aggregate report to %s", args.output)


if __name__ == "__main__":
    main()
