"""Unified output directory utilities for all benchmarks.

统一输出目录工具 | Unified Output Directory Utilities

Provides consistent path resolution across all benchmark evaluations.

Usage:
    from scripts.core.output_utils import (
        get_benchmark_root,
        get_run_dir,
        get_predictions_path,
        get_metrics_path,
        get_log_path,
        ensure_run_dir,
        StreamingWriter,
    )

    # Get paths
    run_dir = get_run_dir("prosthesis_match", "qwen3-vl-30b")

    # Use streaming writer for real-time output
    with StreamingWriter(run_dir / "predictions.jsonl") as writer:
        for item in items:
            result = process(item)
            writer.write(result)
"""

from __future__ import annotations

import json
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Set

__all__ = [
    "get_benchmark_root",
    "get_run_dir",
    "get_predictions_path",
    "get_metrics_path",
    "get_log_path",
    "ensure_run_dir",
    "StreamingWriter",
    "write_manifest",
    "load_completed_ids",
    "BENCHMARK_WORK_DIRS",
]

logger = logging.getLogger(__name__)

# Project root (InclusiveVLM-LEP/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# outputs root
WORK_DIRS_ROOT = PROJECT_ROOT / "outputs"

# Benchmark to outputs subdirectory mapping
# 基准测试到工作目录的映射
BENCHMARK_WORK_DIRS: Dict[str, str] = {
    "limb_evidence_grounding": "inclusive_vlm_lep/limb_evidence_grounding",
    "prosthesis_match": "inclusive_vlm_lep/prosthesis_matching",
    "attribution_freeform": "inclusive_vlm_lep/attribution_freeform",
}


def get_benchmark_root(benchmark_name: str) -> Path:
    """Get the outputs root for a benchmark.

    获取基准测试的工作目录根路径

    Args:
        benchmark_name: Benchmark identifier (e.g., 'prosthesis_match', 'limb_evidence_grounding')

    Returns:
        Path to benchmark work directory

    Raises:
        ValueError: If benchmark_name is not recognized
    """
    if benchmark_name not in BENCHMARK_WORK_DIRS:
        raise ValueError(
            f"Unknown benchmark: {benchmark_name}. "
            f"Available: {list(BENCHMARK_WORK_DIRS.keys())}"
        )
    return WORK_DIRS_ROOT / BENCHMARK_WORK_DIRS[benchmark_name]


def get_run_dir(benchmark_name: str, model_name: str) -> Path:
    """Get the run directory for a model.

    获取模型运行目录

    Args:
        benchmark_name: Benchmark identifier
        model_name: Model name

    Returns:
        Path to model run directory (outputs/{benchmark}/runs/{model})
    """
    return get_benchmark_root(benchmark_name) / "runs" / model_name


def get_predictions_path(
    benchmark_name: str,
    model_name: str,
    filename: str = "predictions.jsonl",
) -> Path:
    """Get the predictions file path.

    获取预测文件路径

    Args:
        benchmark_name: Benchmark identifier
        model_name: Model name
        filename: Predictions filename (default: predictions.jsonl)

    Returns:
        Path to predictions file
    """
    return get_run_dir(benchmark_name, model_name) / filename


def get_metrics_path(
    benchmark_name: str,
    model_name: str,
    suffix: str = ".metrics.json",
) -> Path:
    """Get the metrics file path.

    获取指标文件路径

    Args:
        benchmark_name: Benchmark identifier
        model_name: Model name
        suffix: Metrics file suffix (default: .metrics.json)

    Returns:
        Path to metrics file (outputs/{benchmark}/metrics/{model}{suffix})
    """
    metrics_dir = get_benchmark_root(benchmark_name) / "metrics"
    return metrics_dir / f"{model_name}{suffix}"


def get_log_path(
    benchmark_name: str,
    model_name: str,
    prefix: str = "eval_",
) -> Path:
    """Get the log file path.

    获取日志文件路径

    Args:
        benchmark_name: Benchmark identifier
        model_name: Model name
        prefix: Log file prefix (default: eval_)

    Returns:
        Path to log file (outputs/{benchmark}/logs/{prefix}{model}.log)
    """
    logs_dir = get_benchmark_root(benchmark_name) / "logs"
    return logs_dir / f"{prefix}{model_name}.log"


def ensure_run_dir(benchmark_name: str, model_name: str) -> Path:
    """Ensure run directory exists and return its path.

    确保运行目录存在并返回路径

    Args:
        benchmark_name: Benchmark identifier
        model_name: Model name

    Returns:
        Path to created run directory
    """
    run_dir = get_run_dir(benchmark_name, model_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def load_completed_ids(
    predictions_path: Path,
    id_key: str = "item_id",
) -> Set[str]:
    """Load IDs of completed predictions for resume support.

    加载已完成的预测 ID 以支持断点续传

    Args:
        predictions_path: Path to predictions JSONL file
        id_key: Key name for the ID field (default: item_id)

    Returns:
        Set of completed item IDs
    """
    completed: Set[str] = set()
    if not predictions_path.exists():
        return completed

    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                item_id = record.get(id_key)
                if item_id:
                    completed.add(str(item_id))
            except json.JSONDecodeError:
                continue

    return completed


def write_manifest(
    run_dir: Path,
    benchmark_name: str,
    model_name: str,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    output_files: Optional[Dict[str, str]] = None,
) -> Path:
    """Write run manifest file.

    写入运行清单文件

    Args:
        run_dir: Run directory path
        benchmark_name: Benchmark identifier
        model_name: Model name
        config: Configuration dictionary
        stats: Statistics dictionary
        output_files: Optional mapping of file types to filenames

    Returns:
        Path to manifest file
    """
    manifest = {
        "benchmark_name": benchmark_name,
        "model_name": model_name,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "stats": stats,
        "output_files": output_files or {
            "predictions": "predictions.jsonl",
            "failures": "failures.jsonl",
        },
    }

    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    return manifest_path


class StreamingWriter:
    """Streaming JSONL writer with auto-flush for real-time output.

    流式 JSONL 写入器，支持自动刷新以实现实时输出

    Features:
    - Auto-flush after each write
    - Resume support (append mode)
    - Atomic writes with exception handling

    Usage:
        with StreamingWriter(path) as writer:
            for item in items:
                result = process(item)
                writer.write(result)
    """

    def __init__(
        self,
        path: Path,
        mode: str = "a",
        ensure_ascii: bool = False,
    ):
        """Initialize streaming writer.

        Args:
            path: Output file path
            mode: File mode ('a' for append, 'w' for overwrite)
            ensure_ascii: JSON ensure_ascii setting
        """
        self.path = Path(path)
        self.mode = mode
        self.ensure_ascii = ensure_ascii
        self._file = None
        self._count = 0

    def __enter__(self) -> "StreamingWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.path, self.mode, encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def write(self, record: Dict[str, Any]) -> None:
        """Write a single record and flush.

        写入单条记录并刷新

        Args:
            record: Record dictionary to write
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")

        line = json.dumps(record, ensure_ascii=self.ensure_ascii)
        self._file.write(line + "\n")
        self._file.flush()
        self._count += 1

    def write_batch(self, records: list[Dict[str, Any]]) -> None:
        """Write multiple records and flush once.

        写入多条记录并刷新一次

        Args:
            records: List of record dictionaries
        """
        if self._file is None:
            raise RuntimeError("Writer not opened. Use 'with' statement.")

        for record in records:
            line = json.dumps(record, ensure_ascii=self.ensure_ascii)
            self._file.write(line + "\n")
        self._file.flush()
        self._count += len(records)

    @property
    def count(self) -> int:
        """Number of records written."""
        return self._count


@contextmanager
def atomic_write(path: Path) -> Iterator[StreamingWriter]:
    """Context manager for atomic file writing with temp file.

    原子文件写入上下文管理器

    Writes to a temp file and renames on success to prevent corruption.

    Args:
        path: Target file path

    Yields:
        StreamingWriter instance
    """
    import tempfile

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory for atomic rename
    temp_fd, temp_path = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}_",
        suffix=path.suffix,
    )

    try:
        # Close the fd, we'll open it properly
        import os
        os.close(temp_fd)

        writer = StreamingWriter(Path(temp_path), mode="w")
        with writer:
            yield writer

        # Success: rename temp to target
        Path(temp_path).rename(path)
        logger.debug(f"Atomic write completed: {path}")

    except Exception:
        # Failure: cleanup temp file
        try:
            Path(temp_path).unlink(missing_ok=True)
        except Exception:
            pass
        raise
