"""CLI utilities for benchmark evaluation pipelines.

CLI 工具模块 | CLI Utilities Module

This module provides common CLI helper functions used across benchmarks:
- Execution flag resolution
- Common argument parsers
- Output path resolution

These utilities ensure consistent behavior across all benchmark CLI tools.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "resolve_execution_flags",
    "resolve_output_path",
    "validate_path_exists",
]


def resolve_execution_flags(
    run_inference: bool | None,
    run_metrics: bool | None,
) -> tuple[bool, bool]:
    """Resolve boolean execution flags where None means unspecified.

    解析执行标志 | Resolve Execution Flags

    This function implements consistent logic for determining which pipeline
    stages to run based on user-specified flags:

    Logic:
    - If both None: run both steps (full pipeline)
    - If one specified: infer the other as opposite (partial pipeline)
    - If both specified: use as-is (explicit control)

    Args:
        run_inference: Whether to run inference step (None = auto).
        run_metrics: Whether to run metrics step (None = auto).

    Returns:
        Tuple of (run_inference, run_metrics) boolean values.

    Examples:
        >>> resolve_execution_flags(None, None)
        (True, True)  # Run both

        >>> resolve_execution_flags(True, None)
        (True, False)  # Only inference

        >>> resolve_execution_flags(None, True)
        (False, True)  # Only metrics

        >>> resolve_execution_flags(True, True)
        (True, True)  # Both explicitly
    """
    if run_inference is None and run_metrics is None:
        # Default: run full pipeline
        return True, True

    if run_inference is None:
        # run_metrics specified: infer run_inference
        return not run_metrics, bool(run_metrics)

    if run_metrics is None:
        # run_inference specified: infer run_metrics
        return bool(run_inference), not run_inference

    # Both explicitly specified
    return bool(run_inference), bool(run_metrics)


def resolve_output_path(
    output_path: Path | None,
    default_dir: Path,
    model_name: str,
    suffix: str = ".jsonl",
    create_dirs: bool = True,
) -> Path:
    """Resolve output path with default fallback.

    解析输出路径 | Resolve Output Path

    If output_path is None, generates a default path in the format:
    {default_dir}/{sanitized_model_name}{suffix}

    Args:
        output_path: User-specified output path (None for auto).
        default_dir: Default directory for output files.
        model_name: Model name for generating filename.
        suffix: File suffix (e.g., ".jsonl", ".json").
        create_dirs: Whether to create parent directories.

    Returns:
        Resolved output path.

    Example:
        >>> resolve_output_path(None, Path("outputs"), "gpt-4o-mini", ".jsonl")
        Path("outputs/gpt-4o-mini.jsonl")
    """
    if output_path is not None:
        resolved = Path(output_path)
    else:
        # Sanitize model name for use in filename
        safe_name = model_name.replace("/", "_").replace(":", "_").replace(" ", "_")
        resolved = default_dir / f"{safe_name}{suffix}"

    if create_dirs:
        resolved.parent.mkdir(parents=True, exist_ok=True)

    return resolved


def validate_path_exists(
    path: Path,
    description: str = "File",
    must_be_file: bool = True,
) -> Path:
    """Validate that a path exists and optionally is a file.

    验证路径存在 | Validate Path Exists

    Args:
        path: Path to validate.
        description: Description for error message (e.g., "Dataset", "Config").
        must_be_file: Whether path must be a file (not directory).

    Returns:
        The validated path.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ValueError: If must_be_file=True but path is a directory.

    Example:
        >>> validate_path_exists(Path("data/items.jsonl"), "Dataset file")
        Path("data/items.jsonl")
    """
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")

    if must_be_file and path.is_dir():
        raise ValueError(f"{description} is a directory, expected file: {path}")

    return path


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in file paths.

    净化模型名称 | Sanitize Model Name

    Replaces characters that are problematic in file paths.

    Args:
        model_name: Original model name.

    Returns:
        Sanitized name safe for file paths.

    Example:
        >>> sanitize_model_name("meta-llama/Llama-3.2-11B-Vision")
        "meta-llama_Llama-3.2-11B-Vision"
    """
    replacements = [
        ("/", "_"),
        ("\\", "_"),
        (":", "_"),
        (" ", "_"),
        ("*", "_"),
        ("?", "_"),
        ('"', "_"),
        ("<", "_"),
        (">", "_"),
        ("|", "_"),
    ]
    result = model_name
    for old, new in replacements:
        result = result.replace(old, new)
    return result


def format_run_summary(
    model_name: str,
    total_samples: int,
    elapsed_seconds: float,
    output_path: Path | None = None,
    metrics: dict[str, float] | None = None,
) -> str:
    """Format a human-readable run summary.

    格式化运行摘要 | Format Run Summary

    Args:
        model_name: Name of the model used.
        total_samples: Number of samples processed.
        elapsed_seconds: Total runtime in seconds.
        output_path: Path to output file (if any).
        metrics: Dict of metric_name -> value (if any).

    Returns:
        Formatted summary string.
    """
    lines = [
        "=" * 60,
        "Run Summary | 运行摘要",
        "=" * 60,
        f"Model: {model_name}",
        f"Samples: {total_samples}",
        f"Time: {elapsed_seconds:.1f}s ({elapsed_seconds/60:.1f}m)",
    ]

    if total_samples > 0 and elapsed_seconds > 0:
        rate = total_samples / elapsed_seconds
        lines.append(f"Rate: {rate:.2f} samples/s")

    if output_path:
        lines.append(f"Output: {output_path}")

    if metrics:
        lines.append("-" * 60)
        lines.append("Metrics:")
        for name, value in metrics.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")

    lines.append("=" * 60)
    return "\n".join(lines)
