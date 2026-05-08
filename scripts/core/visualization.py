"""Visualization utilities for benchmark results.

可视化工具 | Visualization Utilities

This module provides comprehensive plotting functions for benchmark metrics:
- Model comparison charts
- Confusion matrices
- Category breakdowns
- Radar/spider charts for multi-metric comparison
- Performance over time
- Distribution analysis

Requires matplotlib (optional dependency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

__all__ = [
    "plot_metric_comparison",
    "plot_confusion_matrix",
    "plot_breakdown_by_category",
    "plot_per_num_correct_breakdown",
    "plot_radar_chart",
    "plot_metric_heatmap",
    "plot_threshold_curve",
    "plot_performance_summary",
    "create_benchmark_report",
    "save_plot",
]


def _check_matplotlib() -> bool:
    """Check if matplotlib is available."""
    try:
        import importlib.util
        return importlib.util.find_spec("matplotlib") is not None
    except ImportError:
        return False


def plot_metric_comparison(
    model_metrics: dict[str, dict[str, Any]],
    metric_name: str = "exact_accuracy",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Any:
    """Plot metric comparison across models with error bars.

    绘制模型对比图 | Plot Model Comparison

    Args:
        model_metrics: Dict mapping model name to metrics dict with 'mean' and 'std'.
        metric_name: Name of metric to plot.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt

    # Extract data
    models = []
    means = []
    stds = []

    for model_name, metrics in model_metrics.items():
        if isinstance(metrics, dict):
            if "mean" in metrics:
                means.append(metrics["mean"])
                stds.append(metrics.get("std", 0))
            elif metric_name in metrics:
                metric_data = metrics[metric_name]
                if hasattr(metric_data, "mean"):
                    means.append(metric_data.mean)
                    stds.append(metric_data.std)
                elif isinstance(metric_data, dict):
                    means.append(metric_data.get("mean", metric_data.get("value", 0)))
                    stds.append(metric_data.get("std", 0))
                else:
                    means.append(float(metric_data))
                    stds.append(0)
            else:
                continue
            models.append(model_name)

    if not models:
        raise ValueError(f"No data found for metric '{metric_name}'")

    # Sort by mean
    sorted_data = sorted(zip(models, means, stds, strict=True), key=lambda x: x[1], reverse=True)
    models, means, stds = zip(*sorted_data, strict=True)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x = range(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"Model Comparison: {metric_name}")

    # Add value labels
    for bar, mean, _std in zip(bars, means, stds, strict=True):
        height = bar.get_height()
        if "rate" in metric_name or "accuracy" in metric_name:
            label = f"{mean:.1%}"
        else:
            label = f"{mean:.2f}"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


def plot_confusion_matrix(
    confusion: dict[str, dict[str, int]],
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (8, 6),
    cmap: str = "Blues",
) -> Any:
    """Plot confusion matrix heatmap.

    绘制混淆矩阵 | Plot Confusion Matrix

    Args:
        confusion: Nested dict with confusion[target][pred] = count.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract labels and matrix
    targets = list(confusion.keys())
    if not targets:
        raise ValueError("Empty confusion matrix")

    preds = list(confusion[targets[0]].keys())

    # Clean up labels
    targets_clean = [t.replace("target=", "") for t in targets]
    preds_clean = [p.replace("pred=", "") for p in preds]

    # Build matrix
    matrix = np.zeros((len(targets), len(preds)))
    for i, target in enumerate(targets):
        for j, pred in enumerate(preds):
            matrix[i, j] = confusion[target].get(pred, 0)

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(matrix, cmap=cmap)

    # Labels
    ax.set_xticks(range(len(preds_clean)))
    ax.set_yticks(range(len(targets_clean)))
    ax.set_xticklabels(preds_clean, rotation=45, ha="right")
    ax.set_yticklabels(targets_clean)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Target")
    ax.set_title(title)

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(len(targets)):
        for j in range(len(preds)):
            value = int(matrix[i, j])
            if value > 0:
                text_color = "white" if matrix[i, j] > matrix.max() / 2 else "black"
                ax.text(j, i, str(value), ha="center", va="center", color=text_color)

    plt.tight_layout()
    return fig


def plot_breakdown_by_category(
    breakdown: dict[str, dict[str, Any]],
    metric_name: str = "exact_accuracy",
    title: str | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> Any:
    """Plot metrics breakdown by category.

    绘制分类指标分布 | Plot Breakdown by Category

    Args:
        breakdown: Dict mapping category to metrics.
        metric_name: Name of metric to plot.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data
    categories = []
    values = []
    counts = []

    for category, metrics in breakdown.items():
        if metric_name in metrics:
            categories.append(category)
            values.append(metrics[metric_name])
            counts.append(metrics.get("n", 0))

    if not categories:
        raise ValueError(f"No data found for metric '{metric_name}'")

    # Create plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(categories))
    width = 0.6

    bars = ax.bar(x, values, width, color="steelblue", alpha=0.8)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylabel(metric_name.replace("_", " ").title())
    ax.set_title(title or f"Breakdown by Category: {metric_name}")

    # Add value and count labels
    for bar, val, count in zip(bars, values, counts, strict=True):
        height = bar.get_height()
        if "rate" in metric_name or "accuracy" in metric_name:
            label = f"{val:.1%}\n(n={count})"
        else:
            label = f"{val:.2f}\n(n={count})"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    return fig


def save_plot(
    fig: Any,
    output_path: Path,
    dpi: int = 150,
    bbox_inches: str = "tight",
) -> None:
    """Save plot to file.

    保存图表 | Save Plot

    Args:
        fig: matplotlib Figure object.
        output_path: Output file path (.png, .pdf, .svg supported).
        dpi: Resolution for raster formats.
        bbox_inches: Bounding box mode.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches=bbox_inches)
    print(f"Plot saved to {output_path}")


def plot_per_num_correct_breakdown(
    breakdown: dict[str, dict[str, Any]],
    figsize: tuple[int, int] = (12, 5),
) -> Any:
    """Plot metrics breakdown by number of correct answers.

    绘制按正确答案数量的分布 | Plot Breakdown by Num Correct

    Args:
        breakdown: Dict mapping num_correct (as str) to metrics.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract and sort data
    sorted_items = sorted(breakdown.items(), key=lambda x: int(x[0]))

    num_correct = [int(k) for k, _ in sorted_items]
    exact_acc = [v.get("exact_accuracy", 0) for _, v in sorted_items]
    partial_acc = [v.get("partial_accuracy", 0) for _, v in sorted_items]
    counts = [v.get("n", 0) for _, v in sorted_items]

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Accuracy by num_correct
    x = np.arange(len(num_correct))
    width = 0.35

    ax1.bar(
        x - width / 2, exact_acc, width, label="Exact Match", color="steelblue"
    )
    ax1.bar(
        x + width / 2, partial_acc, width, label="Partial Match", color="lightsteelblue"
    )

    ax1.set_xlabel("Number of Correct Answers")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy by Number of Correct Answers")
    ax1.set_xticks(x)
    ax1.set_xticklabels(num_correct)
    ax1.legend()
    ax1.set_ylim(0, 1.1)

    # Right: Sample distribution
    ax2.bar(x, counts, color="gray", alpha=0.7)
    ax2.set_xlabel("Number of Correct Answers")
    ax2.set_ylabel("Number of Samples")
    ax2.set_title("Sample Distribution")
    ax2.set_xticks(x)
    ax2.set_xticklabels(num_correct)

    # Add count labels
    for bar, count in zip(ax2.patches, counts, strict=True):
        height = bar.get_height()
        ax2.annotate(
            str(count),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Advanced Visualization Functions
# ---------------------------------------------------------------------------


def plot_radar_chart(
    model_metrics: dict[str, dict[str, float]],
    metrics: Sequence[str] | None = None,
    title: str = "Model Comparison (Radar)",
    figsize: tuple[int, int] = (8, 8),
) -> Any:
    """Plot radar/spider chart for multi-metric model comparison.

    绘制雷达图对比 | Plot Radar Chart

    Args:
        model_metrics: Dict mapping model name to metrics dict.
        metrics: List of metric names to include (uses all if None).
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    # Determine metrics to plot
    if metrics is None:
        all_metrics: set[str] = set()
        for m in model_metrics.values():
            all_metrics.update(m.keys())
        metrics = sorted(all_metrics)

    num_vars = len(metrics)
    if num_vars < 3:
        raise ValueError("Radar chart requires at least 3 metrics")

    # Compute angle for each metric
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_metrics)))

    for idx, (model_name, metric_dict) in enumerate(model_metrics.items()):
        values = [metric_dict.get(m, 0) for m in metrics]
        values += values[:1]  # Close the polygon

        ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=colors[idx])
        ax.fill(angles, values, alpha=0.1, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace("_", "\n") for m in metrics], size=8)
    ax.set_title(title, size=12, y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))

    return fig


def plot_metric_heatmap(
    results: dict[str, dict[str, float]],
    row_label: str = "Model",
    col_label: str = "Metric",
    title: str = "Metrics Heatmap",
    figsize: tuple[int, int] = (12, 6),
    cmap: str = "RdYlGn",
    annotate: bool = True,
) -> Any:
    """Plot heatmap of metrics across models.

    绘制指标热力图 | Plot Metrics Heatmap

    Args:
        results: Dict mapping row name (model) to dict of metrics.
        row_label: Label for rows.
        col_label: Label for columns.
        title: Plot title.
        figsize: Figure size.
        cmap: Colormap name.
        annotate: Whether to show values in cells.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    # Extract data
    rows = list(results.keys())
    cols = sorted(
        set(col for row_data in results.values() for col in row_data.keys())
    )

    # Build matrix
    matrix = np.zeros((len(rows), len(cols)))
    for i, row in enumerate(rows):
        for j, col in enumerate(cols):
            matrix[i, j] = results[row].get(col, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(rows)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(rows, fontsize=9)
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)

    plt.colorbar(im, ax=ax, shrink=0.8)

    if annotate:
        for i in range(len(rows)):
            for j in range(len(cols)):
                val = matrix[i, j]
                if not np.isnan(val):
                    text_color = "white" if val > 0.5 else "black"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            color=text_color, fontsize=8)

    plt.tight_layout()
    return fig


def plot_threshold_curve(
    thresholds: Sequence[float],
    metrics: dict[str, Sequence[float]],
    title: str = "Threshold vs Metrics",
    figsize: tuple[int, int] = (10, 6),
) -> Any:
    """Plot metrics against threshold values.

    绘制阈值曲线 | Plot Threshold Curve

    Useful for analyzing precision-recall tradeoffs.

    Args:
        thresholds: List of threshold values.
        metrics: Dict mapping metric name to values at each threshold.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for metric_name, values in metrics.items():
        ax.plot(thresholds, values, "o-", label=metric_name, linewidth=2)

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_performance_summary(
    model_results: dict[str, dict[str, Any]],
    primary_metric: str = "exact_accuracy",
    secondary_metrics: Sequence[str] | None = None,
    title: str = "Performance Summary",
    figsize: tuple[int, int] = (14, 6),
) -> Any:
    """Create comprehensive performance summary with multiple subplots.

    绘制性能综合摘要 | Plot Performance Summary

    Args:
        model_results: Dict mapping model name to results dict.
        primary_metric: Main metric for bar chart.
        secondary_metrics: Additional metrics for table.
        title: Plot title.
        figsize: Figure size.

    Returns:
        matplotlib Figure object.
    """
    if not _check_matplotlib():
        raise ImportError(
            "matplotlib is required for plotting. Install with: pip install matplotlib"
        )

    import matplotlib.pyplot as plt
    import numpy as np

    if secondary_metrics is None:
        secondary_metrics = ["partial_accuracy", "f1_score"]

    # Sort models by primary metric
    sorted_models = sorted(
        model_results.items(),
        key=lambda x: x[1].get(primary_metric, 0),
        reverse=True,
    )

    models = [m for m, _ in sorted_models]
    primary_values = [r.get(primary_metric, 0) for _, r in sorted_models]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Left: Primary metric bar chart
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(models)))
    bars = ax1.barh(range(len(models)), primary_values, color=colors)
    ax1.set_yticks(range(len(models)))
    ax1.set_yticklabels(models)
    ax1.set_xlabel(primary_metric.replace("_", " ").title())
    ax1.set_title(f"Models Ranked by {primary_metric}")
    ax1.invert_yaxis()

    for bar, val in zip(bars, primary_values, strict=True):
        ax1.text(
            val + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2%}" if val <= 1 else f"{val:.2f}",
            va="center", fontsize=9
        )

    # Right: Secondary metrics table
    cell_text = []
    for model, results in sorted_models:
        row = [model]
        for metric in secondary_metrics:
            val = results.get(metric, "N/A")
            if isinstance(val, float):
                row.append(f"{val:.3f}" if val <= 1 else f"{val:.1f}")
            else:
                row.append(str(val))
        cell_text.append(row)

    ax2.axis("off")
    table = ax2.table(
        cellText=cell_text,
        colLabels=["Model"] + list(secondary_metrics),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax2.set_title("Secondary Metrics")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    return fig


def create_benchmark_report(
    results: dict[str, Any],
    output_dir: Path,
    benchmark_name: str = "Benchmark",
) -> list[Path]:
    """Create comprehensive benchmark report with multiple plots.

    创建基准测试报告 | Create Benchmark Report

    Generates multiple visualizations and saves them to output directory.

    Args:
        results: Benchmark results dict containing model results and metadata.
        output_dir: Directory to save plots.
        benchmark_name: Name for report title.

    Returns:
        List of saved file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_files: list[Path] = []

    # Model comparison
    if "models" in results or "model_results" in results:
        model_data = results.get("models") or results.get("model_results", {})
        if model_data:
            try:
                fig = plot_metric_comparison(model_data, title=f"{benchmark_name}: Model Comparison")
                path = output_dir / "model_comparison.png"
                save_plot(fig, path)
                saved_files.append(path)
            except Exception:
                pass

    # Breakdown by category
    if "breakdown" in results or "by_category" in results:
        breakdown_data = results.get("breakdown") or results.get("by_category", {})
        if breakdown_data:
            try:
                fig = plot_breakdown_by_category(breakdown_data, title=f"{benchmark_name}: Category Breakdown")
                path = output_dir / "category_breakdown.png"
                save_plot(fig, path)
                saved_files.append(path)
            except Exception:
                pass

    # Confusion matrix
    if "confusion" in results or "confusion_matrix" in results:
        confusion_data = results.get("confusion") or results.get("confusion_matrix", {})
        if confusion_data:
            try:
                fig = plot_confusion_matrix(confusion_data, title=f"{benchmark_name}: Confusion Matrix")
                path = output_dir / "confusion_matrix.png"
                save_plot(fig, path)
                saved_files.append(path)
            except Exception:
                pass

    # Prosthesis Matching specific: num_correct breakdown
    if "by_num_correct" in results:
        try:
            fig = plot_per_num_correct_breakdown(results["by_num_correct"])
            path = output_dir / "by_num_correct.png"
            save_plot(fig, path)
            saved_files.append(path)
        except Exception:
            pass

    return saved_files
