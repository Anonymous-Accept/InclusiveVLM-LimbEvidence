"""Evaluation configuration for Limb-Evidence Grounding."""

from __future__ import annotations

from pathlib import Path

from . import config as benchmark_config

__all__ = [
    "PROJECT_ROOT",
    "BENCHMARK_NAME",
    "BENCHMARK_PART",
    "TASK_WORK_DIRS",
    "DEFAULT_WORK_DIR",
    "DEFAULT_QUERIES_PATH",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_RESULTS_DIR",  # Legacy alias for RUNS_DIR
    "DEFAULT_METRICS_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_TASKS",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SEED",
    "get_task_work_dir",
    "get_model_run_dir",
    "get_model_predictions_path",
    "get_model_metrics_path",
    "get_model_log_path",
]

# Re-export from config
PROJECT_ROOT = benchmark_config.PROJECT_ROOT
BENCHMARK_NAME = benchmark_config.BENCHMARK_NAME
BENCHMARK_PART = benchmark_config.BENCHMARK_PART
DEFAULT_WORK_DIR = benchmark_config.DEFAULT_WORK_DIR
DEFAULT_QUERIES_PATH = benchmark_config.DEFAULT_QUERIES_PATH

TASK_WORK_DIRS = {
    "recognition": DEFAULT_WORK_DIR,
    "presence": DEFAULT_WORK_DIR,
    "attribution": DEFAULT_WORK_DIR,
    "attribution_constrained": DEFAULT_WORK_DIR,
}

# Evaluation paths - unified structure
DEFAULT_RUNS_DIR = DEFAULT_WORK_DIR / "runs"
DEFAULT_RESULTS_DIR = DEFAULT_RUNS_DIR  # Legacy alias
DEFAULT_METRICS_DIR = DEFAULT_WORK_DIR / "metrics"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"

# Default evaluation settings
DEFAULT_TASKS = ("recognition", "attribution")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 64
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------


def get_task_work_dir(task_name: str = "recognition") -> Path:
    """Get the canonical work directory for one Limb-Evidence Grounding task."""

    normalized = task_name.strip().lower()
    try:
        return TASK_WORK_DIRS[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported Limb-Evidence Grounding task: {task_name}") from exc


def get_model_run_dir(model_name: str, task_name: str = "recognition") -> Path:
    """Get model-specific run directory."""
    return get_task_work_dir(task_name) / "runs" / model_name


def get_model_predictions_path(model_name: str, task_name: str = "recognition") -> Path:
    """Get model-specific predictions file path."""
    return get_model_run_dir(model_name, task_name) / "predictions.jsonl"


def get_model_metrics_path(model_name: str, task_name: str = "recognition") -> Path:
    """Get model-specific metrics file path."""
    return get_task_work_dir(task_name) / "metrics" / f"{model_name}.metrics.json"


def get_model_log_path(model_name: str, task_name: str = "recognition") -> Path:
    """Get model-specific log file path."""
    return get_task_work_dir(task_name) / "logs" / f"eval_{model_name}.log"
