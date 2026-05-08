"""Generate the InclusiveVLM-LEP paper findings tables and figures."""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.core.io_utils import save_json
from scripts.core.logging_utils import configure_logging, get_logger

from ._shared import (
    DEFAULT_FINDINGS_ROOT,
    DEFAULT_METRICS_ROOT,
    DISPLAY_NAME_BY_VARIANT,
    KEY_METRIC_BY_VARIANT,
    VARIANT_ORDER,
    load_metric_payloads,
    load_metrics_index,
    load_per_group_rows,
    markdown_table,
    metric_cell,
    write_csv,
    write_markdown,
)

LOGGER = get_logger(__name__)


def _save_plot(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()


def _metrics_by_variant(metrics_index: dict[str, dict[str, dict[str, Any]]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for model_name, by_variant in metrics_index.items():
        for variant, payload in by_variant.items():
            score = float(payload.get("metrics", {}).get(KEY_METRIC_BY_VARIANT[variant], 0.0))
            rows[variant].append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "score": score,
                    "metrics": payload.get("metrics", {}),
                    "_path": Path(payload["_path"]),
                }
            )
    return rows


def run_f1(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Persistent failure table across all variants."""

    rows = []
    for model_name in sorted(metrics_index):
        row: dict[str, Any] = {"model_name": model_name}
        for variant in VARIANT_ORDER:
            payload = metrics_index[model_name].get(variant)
            value = None
            if payload:
                value = payload.get("metrics", {}).get(KEY_METRIC_BY_VARIANT[variant])
            row[variant] = metric_cell(value)
        rows.append(row)
    csv_rows = [
        {
            "model_name": row["model_name"],
            **{variant: row[variant] for variant in VARIANT_ORDER},
        }
        for row in rows
    ]
    csv_path = output_dir / "finding_f1_persistent_failure.csv"
    md_path = output_dir / "finding_f1_persistent_failure.md"
    fig_path = output_dir / "finding_f1_persistent_failure.png"
    write_csv(csv_path, csv_rows, ["model_name", *VARIANT_ORDER])
    write_markdown(
        md_path,
        "# F1 Persistent Failure\n\n"
        + markdown_table(
            rows,
            [("model_name", "Model")]
            + [(variant, DISPLAY_NAME_BY_VARIANT[variant]) for variant in VARIANT_ORDER],
        )
        + "\n",
    )
    plt.figure(figsize=(10, max(4, len(rows) * 0.35)))
    for idx, variant in enumerate(VARIANT_ORDER):
        values = [
            float(metrics_index[row["model_name"]].get(variant, {}).get("metrics", {}).get(KEY_METRIC_BY_VARIANT[variant], 0.0))
            for row in rows
        ]
        plt.plot(values, range(len(rows)), marker="o", label=DISPLAY_NAME_BY_VARIANT[variant])
    plt.yticks(range(len(rows)), [row["model_name"] for row in rows])
    plt.xlabel("Variant score")
    plt.ylabel("Model")
    plt.legend(fontsize=8)
    _save_plot(fig_path)
    return {"csv": str(csv_path), "markdown": str(md_path), "figure": str(fig_path)}


def run_f2(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Presence vs attribution gap."""

    rows = []
    for model_name, by_variant in sorted(metrics_index.items()):
        presence = by_variant.get("presence")
        attribution = by_variant.get("attribution_constrained")
        if not presence or not attribution:
            continue
        presence_acc = float(presence.get("metrics", {}).get("exact_accuracy", 0.0))
        attribution_miou = float(attribution.get("metrics", {}).get("set_miou", 0.0))
        abs_gap = presence_acc - attribution_miou
        rel_gap = abs_gap / presence_acc if presence_acc else 0.0
        rows.append(
            {
                "model_name": model_name,
                "presence_accuracy": round(presence_acc, 4),
                "attribution_set_miou": round(attribution_miou, 4),
                "absolute_gap": round(abs_gap, 4),
                "relative_gap": round(rel_gap, 4),
            }
        )
    csv_path = output_dir / "finding_f2_attribution_vs_presence.csv"
    md_path = output_dir / "finding_f2_attribution_vs_presence.md"
    fig_path = output_dir / "finding_f2_attribution_vs_presence.png"
    write_csv(
        csv_path,
        rows,
        [
            "model_name",
            "presence_accuracy",
            "attribution_set_miou",
            "absolute_gap",
            "relative_gap",
        ],
    )
    write_markdown(
        md_path,
        "# F2 Attribution vs Presence\n\n"
        + markdown_table(
            rows,
            [
                ("model_name", "Model"),
                ("presence_accuracy", "Presence Acc"),
                ("attribution_set_miou", "Attribution mIoU"),
                ("absolute_gap", "Abs Gap"),
                ("relative_gap", "Rel Gap"),
            ],
        )
        + "\n",
    )
    plt.figure(figsize=(6, 5))
    x = [row["presence_accuracy"] for row in rows]
    y = [row["attribution_set_miou"] for row in rows]
    plt.scatter(x, y)
    for row in rows:
        plt.annotate(row["model_name"], (row["presence_accuracy"], row["attribution_set_miou"]), fontsize=7)
    plt.xlabel("Presence accuracy")
    plt.ylabel("Attribution-Constrained set-mIoU")
    _save_plot(fig_path)
    return {"csv": str(csv_path), "markdown": str(md_path), "figure": str(fig_path)}


def _load_group_pairs(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    variant_a: str,
    variant_b: str,
) -> dict[str, list[tuple[dict[str, Any], dict[str, Any]]]]:
    grouped: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for model_name, by_variant in metrics_index.items():
        payload_a = by_variant.get(variant_a)
        payload_b = by_variant.get(variant_b)
        if not payload_a or not payload_b:
            continue
        rows_a = {
            str(row["join_group_id"]): row
            for row in load_per_group_rows(Path(payload_a["_path"]))
        }
        rows_b = {
            str(row["join_group_id"]): row
            for row in load_per_group_rows(Path(payload_b["_path"]))
        }
        shared = sorted(set(rows_a) & set(rows_b))
        grouped[model_name] = [(rows_a[key], rows_b[key]) for key in shared]
    return grouped


def run_f3(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Presence-attribution correctness coupling."""

    paired = _load_group_pairs(metrics_index, "presence", "attribution_constrained")
    rows = []
    for model_name, pairs in sorted(paired.items()):
        p_correct = [b for a, b in pairs if a["group_any_correct"]]
        p_wrong = [b for a, b in pairs if not a["group_any_correct"]]
        a_correct = [a for a, b in pairs if b["group_any_correct"]]
        a_wrong = [a for a, b in pairs if not b["group_any_correct"]]
        rows.append(
            {
                "model_name": model_name,
                "num_groups": len(pairs),
                "p_a_correct_given_p_correct": round(
                    sum(1 for row in p_correct if row["group_any_correct"]) / len(p_correct),
                    4,
                )
                if p_correct
                else 0.0,
                "p_a_correct_given_p_wrong": round(
                    sum(1 for row in p_wrong if row["group_any_correct"]) / len(p_wrong),
                    4,
                )
                if p_wrong
                else 0.0,
                "p_p_correct_given_a_correct": round(
                    sum(1 for row in a_correct if row["group_any_correct"]) / len(a_correct),
                    4,
                )
                if a_correct
                else 0.0,
                "p_p_correct_given_a_wrong": round(
                    sum(1 for row in a_wrong if row["group_any_correct"]) / len(a_wrong),
                    4,
                )
                if a_wrong
                else 0.0,
            }
        )
    csv_path = output_dir / "finding_f3_presence_attribution_coupling.csv"
    md_path = output_dir / "finding_f3_presence_attribution_coupling.md"
    fig_path = output_dir / "finding_f3_presence_attribution_coupling.png"
    write_csv(
        csv_path,
        rows,
        [
            "model_name",
            "num_groups",
            "p_a_correct_given_p_correct",
            "p_a_correct_given_p_wrong",
            "p_p_correct_given_a_correct",
            "p_p_correct_given_a_wrong",
        ],
    )
    write_markdown(
        md_path,
        "# F3 Presence-Attribution Coupling\n\n"
        + markdown_table(
            rows,
            [
                ("model_name", "Model"),
                ("num_groups", "Groups"),
                ("p_a_correct_given_p_correct", "P(A|P correct)"),
                ("p_a_correct_given_p_wrong", "P(A|P wrong)"),
                ("p_p_correct_given_a_correct", "P(P|A correct)"),
                ("p_p_correct_given_a_wrong", "P(P|A wrong)"),
            ],
        )
        + "\n",
    )
    plt.figure(figsize=(8, 5))
    x = range(len(rows))
    plt.bar(
        [value - 0.2 for value in x],
        [row["p_a_correct_given_p_correct"] for row in rows],
        width=0.4,
        label="P(A|P correct)",
    )
    plt.bar(
        [value + 0.2 for value in x],
        [row["p_a_correct_given_p_wrong"] for row in rows],
        width=0.4,
        label="P(A|P wrong)",
    )
    plt.xticks(list(x), [row["model_name"] for row in rows], rotation=45, ha="right")
    plt.ylabel("Conditional probability")
    plt.legend(fontsize=8)
    _save_plot(fig_path)
    return {"csv": str(csv_path), "markdown": str(md_path), "figure": str(fig_path)}


def run_f4(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Exact vs partial compatibility accuracy gap."""

    rows = []
    for model_name, by_variant in sorted(metrics_index.items()):
        for variant in ["compatibility_diversity", "compatibility_category"]:
            payload = by_variant.get(variant)
            if not payload:
                continue
            exact = float(payload.get("metrics", {}).get("exact_accuracy", 0.0))
            partial = float(payload.get("metrics", {}).get("partial_accuracy", 0.0))
            rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "exact_accuracy": round(exact, 4),
                    "partial_accuracy": round(partial, 4),
                    "gap": round(partial - exact, 4),
                }
            )
    csv_path = output_dir / "finding_f4_compatibility_plausible_vs_exact.csv"
    md_path = output_dir / "finding_f4_compatibility_plausible_vs_exact.md"
    fig_path = output_dir / "finding_f4_compatibility_plausible_vs_exact.png"
    write_csv(csv_path, rows, ["model_name", "variant", "exact_accuracy", "partial_accuracy", "gap"])
    write_markdown(
        md_path,
        "# F4 Compatibility Plausible vs Exact\n\n"
        + markdown_table(
            rows,
            [
                ("model_name", "Model"),
                ("variant", "Variant"),
                ("exact_accuracy", "Exact"),
                ("partial_accuracy", "Partial"),
                ("gap", "Gap"),
            ],
        )
        + "\n",
    )
    plt.figure(figsize=(6, 5))
    for variant in ["compatibility_diversity", "compatibility_category"]:
        subset = [row for row in rows if row["variant"] == variant]
        plt.scatter(
            [row["exact_accuracy"] for row in subset],
            [row["partial_accuracy"] for row in subset],
            label=DISPLAY_NAME_BY_VARIANT[variant],
        )
    plt.xlabel("Exact accuracy")
    plt.ylabel("Partial accuracy")
    plt.legend(fontsize=8)
    _save_plot(fig_path)
    return {"csv": str(csv_path), "markdown": str(md_path), "figure": str(fig_path)}


def run_f5(
    metrics_index: dict[str, dict[str, dict[str, Any]]],
    output_dir: Path,
) -> dict[str, Any]:
    """Consistency vs correctness 2x2 tables."""

    rows = []
    for model_name, by_variant in sorted(metrics_index.items()):
        for variant, payload in sorted(by_variant.items()):
            group_rows = load_per_group_rows(Path(payload["_path"]))
            if not group_rows:
                continue
            cc = sum(1 for row in group_rows if row["group_consistent"] and row["group_any_correct"])
            ci = sum(1 for row in group_rows if row["group_consistent"] and not row["group_any_correct"])
            ic = sum(1 for row in group_rows if not row["group_consistent"] and row["group_any_correct"])
            ii = sum(1 for row in group_rows if not row["group_consistent"] and not row["group_any_correct"])
            total = len(group_rows)
            rows.append(
                {
                    "model_name": model_name,
                    "variant": variant,
                    "num_groups": total,
                    "consistent_and_correct": cc,
                    "consistent_and_wrong": ci,
                    "inconsistent_and_correct": ic,
                    "inconsistent_and_wrong": ii,
                    "consistent_and_correct_rate": round(cc / total, 4) if total else 0.0,
                }
            )
    csv_path = output_dir / "finding_f5_consistency_vs_correctness.csv"
    md_path = output_dir / "finding_f5_consistency_vs_correctness.md"
    fig_path = output_dir / "finding_f5_consistency_vs_correctness.png"
    write_csv(
        csv_path,
        rows,
        [
            "model_name",
            "variant",
            "num_groups",
            "consistent_and_correct",
            "consistent_and_wrong",
            "inconsistent_and_correct",
            "inconsistent_and_wrong",
            "consistent_and_correct_rate",
        ],
    )
    write_markdown(
        md_path,
        "# F5 Consistency vs Correctness\n\n"
        + markdown_table(
            rows,
            [
                ("model_name", "Model"),
                ("variant", "Variant"),
                ("num_groups", "Groups"),
                ("consistent_and_correct", "Consistent+Correct"),
                ("consistent_and_wrong", "Consistent+Wrong"),
                ("inconsistent_and_correct", "Inconsistent+Correct"),
                ("inconsistent_and_wrong", "Inconsistent+Wrong"),
            ],
        )
        + "\n",
    )
    top_rows = rows[: min(20, len(rows))]
    plt.figure(figsize=(10, max(4, len(top_rows) * 0.3)))
    plt.barh(
        range(len(top_rows)),
        [row["consistent_and_correct_rate"] for row in top_rows],
    )
    plt.yticks(
        range(len(top_rows)),
        [f"{row['model_name']}::{row['variant']}" for row in top_rows],
    )
    plt.xlabel("Rate of consistent+correct groups")
    _save_plot(fig_path)
    return {"csv": str(csv_path), "markdown": str(md_path), "figure": str(fig_path)}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FINDINGS_ROOT)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_index = load_metrics_index(args.metrics_root)
    outputs = {
        "f1": run_f1(metrics_index, args.output_dir),
        "f2": run_f2(metrics_index, args.output_dir),
        "f3": run_f3(metrics_index, args.output_dir),
        "f4": run_f4(metrics_index, args.output_dir),
        "f5": run_f5(metrics_index, args.output_dir),
        "metrics_files": [payload["_path"] for payload in load_metric_payloads(args.metrics_root)],
    }
    save_json(outputs, args.output_dir / "summary.json", indent=2, ensure_ascii=False)
    LOGGER.info("Wrote findings outputs under %s", args.output_dir)


if __name__ == "__main__":
    main()
