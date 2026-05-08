"""Enhanced evaluation utilities.

评测功能增强 | Evaluation Enhancements

This module provides:
- Multi-run aggregation with statistical analysis
- Confidence interval computation
- Metrics comparison across models
- Basic visualization helpers
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MetricStats",
    "aggregate_runs",
    "compute_confidence_interval",
    "compare_models",
    "format_metric_table",
]


@dataclass
class MetricStats:
    """Statistical summary for a metric across multiple runs.

    指标统计 | Metric Statistics

    Attributes:
        mean: Mean value.
        std: Standard deviation.
        ci_lower: Lower bound of 95% confidence interval.
        ci_upper: Upper bound of 95% confidence interval.
        n_runs: Number of runs.
        values: Raw values from each run.
    """

    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    n_runs: int
    values: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "n_runs": self.n_runs,
            "values": [round(v, 4) for v in self.values],
        }

    def __str__(self) -> str:
        """Format as string with CI."""
        return (
            f"{self.mean:.2%} ± {self.std:.2%} (95% CI: [{self.ci_lower:.2%},"
            f" {self.ci_upper:.2%}])"
        )


def compute_confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float, float]:
    """Compute confidence interval for a list of values.

    计算置信区间 | Compute Confidence Interval

    Uses t-distribution for small samples (n < 30).

    Args:
        values: List of metric values.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (mean, std, ci_lower, ci_upper).
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    n = len(values)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if n > 1 else 0.0

    if n < 2:
        return mean, std, mean, mean

    # Use t-distribution for small samples
    from scipy import stats

    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha / 2, df=n - 1)
    margin = t_critical * std / math.sqrt(n)

    ci_lower = mean - margin
    ci_upper = mean + margin

    return mean, std, ci_lower, ci_upper


def aggregate_runs(
    metrics_list: list[dict[str, Any]],
    metric_keys: list[str] | None = None,
) -> dict[str, MetricStats]:
    """Aggregate metrics from multiple runs.

    多运行聚合 | Multi-Run Aggregation

    Args:
        metrics_list: List of metrics dictionaries from each run.
        metric_keys: Specific keys to aggregate. If None, aggregates all
                    numeric keys in 'overall'.

    Returns:
        Dictionary mapping metric name to MetricStats.
    """
    if not metrics_list:
        return {}

    # Determine keys to aggregate
    if metric_keys is None:
        # Auto-detect numeric keys from first metrics dict
        first = metrics_list[0]
        if "overall" in first:
            metric_keys = [
                k
                for k, v in first["overall"].items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            ]
        else:
            metric_keys = [
                k
                for k, v in first.items()
                if isinstance(v, (int, float)) and not k.startswith("_")
            ]

    # Collect values for each metric
    aggregated = {}
    for key in metric_keys:
        values = []
        for metrics in metrics_list:
            if "overall" in metrics:
                val = metrics["overall"].get(key)
            else:
                val = metrics.get(key)

            if val is not None and isinstance(val, (int, float)):
                values.append(float(val))

        if values:
            mean, std, ci_lower, ci_upper = compute_confidence_interval(values)
            aggregated[key] = MetricStats(
                mean=mean,
                std=std,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_runs=len(values),
                values=values,
            )

    return aggregated


def compare_models(
    model_metrics: dict[str, dict[str, MetricStats]],
    primary_metric: str = "exact_accuracy",
) -> dict[str, Any]:
    """Compare metrics across multiple models.

    模型对比 | Model Comparison

    Args:
        model_metrics: Dict mapping model name to aggregated metrics.
        primary_metric: Primary metric for ranking (default: exact_accuracy).

    Returns:
        Comparison summary with rankings and statistical tests.
    """
    if not model_metrics:
        return {}

    # Rank models by primary metric
    ranked = []
    for model_name, metrics in model_metrics.items():
        if primary_metric in metrics:
            stat = metrics[primary_metric]
            ranked.append({
                "model": model_name,
                "mean": stat.mean,
                "std": stat.std,
                "ci_lower": stat.ci_lower,
                "ci_upper": stat.ci_upper,
                "n_runs": stat.n_runs,
            })

    # Sort by mean (descending)
    ranked.sort(key=lambda x: x["mean"], reverse=True)

    # Add rank
    for i, item in enumerate(ranked):
        item["rank"] = i + 1

    # Pairwise significance tests (if scipy available)
    significance = {}
    try:
        from scipy import stats

        models = list(model_metrics.keys())
        for i, model_a in enumerate(models):
            for model_b in models[i + 1 :]:
                if primary_metric not in model_metrics[model_a]:
                    continue
                if primary_metric not in model_metrics[model_b]:
                    continue

                values_a = model_metrics[model_a][primary_metric].values
                values_b = model_metrics[model_b][primary_metric].values

                if len(values_a) < 2 or len(values_b) < 2:
                    continue

                # Welch's t-test (unequal variances)
                result = stats.ttest_ind(values_a, values_b, equal_var=False)
                t_stat = float(result.statistic)
                p_val = float(result.pvalue)
                significance[f"{model_a}_vs_{model_b}"] = {
                    "t_statistic": round(t_stat, 4),
                    "p_value": round(p_val, 4),
                    "significant_at_0.05": p_val < 0.05,
                }
    except ImportError:
        pass

    return {
        "primary_metric": primary_metric,
        "rankings": ranked,
        "pairwise_significance": significance,
    }


def format_metric_table(
    model_metrics: dict[str, dict[str, MetricStats]],
    metrics_to_show: list[str] | None = None,
) -> str:
    """Format metrics as a markdown table.

    生成指标表格 | Generate Metric Table

    Args:
        model_metrics: Dict mapping model name to aggregated metrics.
        metrics_to_show: Specific metrics to include in table.

    Returns:
        Markdown table string.
    """
    if not model_metrics:
        return "No metrics available."

    # Determine metrics to show
    if metrics_to_show is None:
        # Get union of all metrics
        all_keys = set()
        for metrics in model_metrics.values():
            all_keys.update(metrics.keys())
        metrics_to_show = sorted(all_keys)

    # Build header
    header = "| Model | " + " | ".join(metrics_to_show) + " |"
    separator = "|" + "|".join(["---"] * (len(metrics_to_show) + 1)) + "|"

    # Build rows
    rows = []
    for model_name, metrics in sorted(model_metrics.items()):
        cells = [model_name]
        for metric_key in metrics_to_show:
            if metric_key in metrics:
                stat = metrics[metric_key]
                # Format as percentage if looks like a rate
                if "rate" in metric_key or "accuracy" in metric_key:
                    cells.append(f"{stat.mean:.1%} ± {stat.std:.1%}")
                else:
                    cells.append(f"{stat.mean:.2f} ± {stat.std:.2f}")
            else:
                cells.append("-")
        rows.append("| " + " | ".join(cells) + " |")

    return "\n".join([header, separator] + rows)


def load_run_metrics(
    runs_dir: Path, pattern: str = "metrics.json"
) -> list[dict[str, Any]]:
    """Load metrics from multiple run directories.

    加载多运行指标 | Load Multi-Run Metrics

    Args:
        runs_dir: Directory containing run subdirectories.
        pattern: Filename pattern for metrics files.

    Returns:
        List of metrics dictionaries.
    """
    metrics_list = []
    runs_dir = Path(runs_dir)

    for metrics_path in runs_dir.rglob(pattern):
        try:
            with open(metrics_path, encoding="utf-8") as f:
                metrics = json.load(f)
                metrics["_source"] = str(metrics_path)
                metrics_list.append(metrics)
        except (OSError, json.JSONDecodeError) as e:
            print(f"Warning: Failed to load {metrics_path}: {e}")

    return metrics_list


def bootstrap_confidence_interval(
    values: list[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    random_state: int | None = None,
) -> tuple[float, float, float, float]:
    """Compute bootstrap confidence interval.

    Bootstrap 置信区间 | Bootstrap Confidence Interval

    Useful when sample size is small or distribution is non-normal.

    Args:
        values: List of metric values.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (mean, std, ci_lower, ci_upper).
    """
    if not values:
        return 0.0, 0.0, 0.0, 0.0

    rng = np.random.RandomState(random_state)
    values_arr = np.array(values)
    n = len(values_arr)

    # Generate bootstrap samples
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values_arr, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    mean = float(np.mean(values_arr))
    std = float(np.std(values_arr, ddof=1)) if n > 1 else 0.0

    # Percentile confidence interval
    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return mean, std, ci_lower, ci_upper
