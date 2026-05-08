"""
Shared core utilities for InclusiveVLM-LEP.

This module provides common configuration, I/O, logging, model-backend, and
evaluation helpers for the reviewer-facing limb-evidence preservation code.

Submodules:
- benchmark_base: Abstract base classes for benchmarks
- config: Unified configuration schema
- exceptions: Centralized exception handling
- cli_utils: CLI helper functions
- utils: Common utility functions
- logging_utils: Logging configuration utilities
- io_utils: JSON/JSONL/YAML I/O utilities
- progress: Progress reporting utilities
- model_registry: Unified model configuration registry
- backends: Vision-language model backends (vLLM, OpenAI, Gemini)
- evaluation: Enhanced evaluation utilities
- visualization: Plotting utilities

Example:
    >>> from scripts.core import configure_logging, load_jsonl, InferenceConfig
    >>> configure_logging("INFO")
    >>> items = load_jsonl("data/items.jsonl")
"""

from __future__ import annotations

__all__ = [
    # Exceptions
    "BenchmarkError",
    "ConfigurationError",
    "DatasetError",
    "InferenceError",
    "EvaluationError",
    "FileIOError",
    "error_context",
    # Configuration
    "BaseConfig",
    "InferenceConfig",
    "MetricsConfig",
    "PipelineConfig",
    "DatasetConfig",
    # CLI utilities
    "resolve_execution_flags",
    "resolve_output_path",
    "validate_path_exists",
    "sanitize_model_name",
    "format_run_summary",
    # Logging
    "configure_logging",
    "get_logger",
    "LOG_FORMAT_STANDARD",
    "LOG_FORMAT_DETAILED",
    # I/O
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "load_yaml",
    "iter_jsonl",
    "append_jsonl",
    "load_jsonl_cached",
    "load_json_cached",
    "clear_io_cache",
    # Progress and Performance
    "ProgressReporter",
    "ProgressTracker",
    "UnifiedProgressBar",
    "PerfTimer",
    "PerfStats",
    "format_eta",
    "format_duration",
    "format_rate",
    "track_progress",
    # Model registry
    "ModelConfig",
    "MODEL_REGISTRY",
    "get_model_config",
    # Base classes (legacy, use config module instead)
    "BenchmarkConfig",
    "InferenceConfigBase",
    "BaseBenchmark",
    "PredictionRecordBase",
    # Utilities
    "chunked",
    "ensure_dir",
    # Evaluation
    "MetricStats",
    "aggregate_runs",
    "compute_confidence_interval",
    "compare_models",
    "format_metric_table",
    # Output utilities
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

# Exceptions
# Legacy base classes (for backward compatibility)
from .benchmark_base import (
    BaseBenchmark,
    BenchmarkConfig,
    InferenceConfigBase,
    PredictionRecordBase,
)

# CLI utilities
from .cli_utils import (
    format_run_summary,
    resolve_execution_flags,
    resolve_output_path,
    sanitize_model_name,
    validate_path_exists,
)

# Configuration
from .config import (
    BaseConfig,
    DatasetConfig,
    InferenceConfig,
    MetricsConfig,
    PipelineConfig,
)

# Evaluation
from .evaluation import (
    MetricStats,
    aggregate_runs,
    compare_models,
    compute_confidence_interval,
    format_metric_table,
)
from .exceptions import (
    BenchmarkError,
    ConfigurationError,
    DatasetError,
    EvaluationError,
    FileIOError,
    InferenceError,
    error_context,
)

# I/O
from .io_utils import (
    append_jsonl,
    clear_io_cache,
    iter_jsonl,
    load_json,
    load_json_cached,
    load_jsonl,
    load_jsonl_cached,
    load_yaml,
    save_json,
    save_jsonl,
)

# Logging
from .logging_utils import (
    LOG_FORMAT_DETAILED,
    LOG_FORMAT_STANDARD,
    configure_logging,
    get_logger,
)

# Model registry
from .model_registry import MODEL_REGISTRY, ModelConfig, get_model_config

# Output utilities
from .output_utils import (
    BENCHMARK_WORK_DIRS,
    StreamingWriter,
    ensure_run_dir,
    get_benchmark_root,
    get_log_path,
    get_metrics_path,
    get_predictions_path,
    get_run_dir,
    load_completed_ids,
    write_manifest,
)

# Progress and Performance
from .progress import (
    PerfStats,
    PerfTimer,
    ProgressReporter,
    ProgressTracker,
    UnifiedProgressBar,
    format_duration,
    format_eta,
    format_rate,
    track_progress,
)

# Utilities
from .utils import (
    chunked,
    ensure_dir,
)
