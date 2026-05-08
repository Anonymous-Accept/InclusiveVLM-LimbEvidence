"""Unified benchmark configuration schema.

统一基准配置 Schema | Unified Benchmark Configuration Schema

This module provides a comprehensive configuration system for all benchmarks:
- Type-safe configuration with dataclasses
- Validation with clear error messages
- Serialization/deserialization support
- Environment variable expansion
- Path resolution utilities

Configuration Hierarchy:
    BaseConfig (shared across all tools)
    ├── BenchmarkConfig (evaluation-specific)
    │   ├── InferenceConfig
    │   ├── MetricsConfig
    │   └── PipelineConfig
    └── ModelConfig (in model_registry.py)

Example:
    >>> from scripts.core.config import InferenceConfig
    >>> config = InferenceConfig(
    ...     model_name="qwen3-vl-30b",
    ...     max_tokens=512,
    ...     temperature=0.0,
    ... )
    >>> config.validate()
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from .exceptions import InvalidConfigError, MissingConfigError

__all__ = [
    "BaseConfig",
    "InferenceConfig",
    "MetricsConfig",
    "PipelineConfig",
    "DatasetConfig",
]


# ---------------------------------------------------------------------------
# Configuration Utilities
# ---------------------------------------------------------------------------


def _get_project_root() -> Path:
    """Get project root directory.

    Searches upward for a directory containing certain marker files.
    """
    current = Path(__file__).resolve().parent
    markers = ["requirements.txt", "pyproject.toml", ".git"]

    for _ in range(5):  # Max 5 levels up
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent

    # Fallback: 2 levels up from this file
    return Path(__file__).resolve().parents[2]


def _expand_env_vars(value: str) -> str:
    """Expand environment variables in string."""
    return os.path.expandvars(value)


def _resolve_path(
    path: Path | str | None,
    base: Path | None = None,
    must_exist: bool = False,
) -> Path | None:
    """Resolve a path relative to base directory.

    Args:
        path: Path to resolve (can be relative or absolute).
        base: Base directory for relative paths.
        must_exist: Whether to validate existence.

    Returns:
        Resolved absolute path, or None if input is None.

    Raises:
        FileNotFoundError: If must_exist=True and path doesn't exist.
    """
    if path is None:
        return None

    path = Path(_expand_env_vars(str(path)))

    if not path.is_absolute() and base is not None:
        path = base / path

    path = path.resolve()

    if must_exist and not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return path


# ---------------------------------------------------------------------------
# Base Configuration
# ---------------------------------------------------------------------------


@dataclass
class BaseConfig:
    """Base configuration shared across all tools.

    基础配置 | Base Configuration

    Provides common infrastructure:
    - Project root detection
    - Path resolution
    - Serialization
    - Validation interface
    """

    # Class-level defaults (can be overridden by subclasses)
    _PROJECT_ROOT: ClassVar[Path | None] = None

    # Common fields
    project_root: Path = field(default_factory=_get_project_root)
    log_level: str = "INFO"
    seed: int = 42

    def __post_init__(self) -> None:
        """Post-initialization processing."""
        # Ensure project_root is Path
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise InvalidConfigError(
                "log_level",
                self.log_level,
                f"one of {valid_levels}",
            )

    def validate(self) -> None:
        """Validate configuration.

        Override in subclasses to add specific validation.
        Raises ConfigurationError on validation failure.
        """
        pass  # Base implementation does nothing

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert Path objects to strings
        for key, value in list(d.items()):
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseConfig:
        """Create config from dictionary.

        Subclasses should override to handle their specific path fields.
        """
        return cls(**data)

    def resolve_path(
        self,
        path: Path | str | None,
        must_exist: bool = False,
    ) -> Path | None:
        """Resolve path relative to project root."""
        return _resolve_path(path, self.project_root, must_exist)


# ---------------------------------------------------------------------------
# Inference Configuration
# ---------------------------------------------------------------------------


@dataclass
class InferenceConfig(BaseConfig):
    """Configuration for model inference.

    推理配置 | Inference Configuration

    Settings for running VLM inference including:
    - Model selection and backend
    - Generation parameters
    - Batch processing
    - Output constraints
    """

    # Model identification
    model_name: str = ""

    # Backend configuration
    backend_type: str = "vllm"  # vllm, openai, gemini
    vllm_mode: str = "server"  # server, local

    # Generation parameters
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1  # -1 means disabled

    # Batch processing
    batch_size: int = 1
    max_concurrent: int = 8
    timeout_seconds: float = 120.0

    # Output constraints
    assistant_prefill: bool = False
    guided_decoding: bool = False
    stop_sequences: list[str] = field(default_factory=list)

    # Execution control
    resume: bool = False
    overwrite: bool = False
    max_samples: int | None = None
    dry_run: bool = False

    # Output paths
    output_path: Path | None = None
    log_path: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        # Convert string paths
        if isinstance(self.output_path, str):
            self.output_path = Path(self.output_path)
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)

    def validate(self) -> None:
        """Validate inference configuration."""
        super().validate()

        if not self.model_name:
            raise MissingConfigError("model_name", suggestion="Specify --model-name")

        valid_backends = {"vllm", "openai", "gemini"}
        if self.backend_type not in valid_backends:
            raise InvalidConfigError(
                "backend_type",
                self.backend_type,
                f"one of {valid_backends}",
            )

        if self.max_tokens < 1:
            raise InvalidConfigError(
                "max_tokens",
                self.max_tokens,
                "positive integer",
            )

        if not 0.0 <= self.temperature <= 2.0:
            raise InvalidConfigError(
                "temperature",
                self.temperature,
                "float between 0.0 and 2.0",
            )

        if not 0.0 <= self.top_p <= 1.0:
            raise InvalidConfigError(
                "top_p",
                self.top_p,
                "float between 0.0 and 1.0",
            )


# ---------------------------------------------------------------------------
# Metrics Configuration
# ---------------------------------------------------------------------------


@dataclass
class MetricsConfig(BaseConfig):
    """Configuration for metric computation.

    指标配置 | Metrics Configuration

    Settings for evaluation metric computation.
    """

    # Input paths
    predictions_path: Path | None = None
    ground_truth_path: Path | None = None

    # Output paths
    metrics_dir: Path | None = None
    reports_dir: Path | None = None

    # Metric options
    compute_per_category: bool = True
    compute_confidence_intervals: bool = True
    bootstrap_samples: int = 1000
    confidence_level: float = 0.95

    # Visualization
    generate_plots: bool = False
    plot_format: str = "png"  # png, pdf, svg

    def __post_init__(self) -> None:
        super().__post_init__()

        # Convert string paths
        path_fields = ["predictions_path", "ground_truth_path", "metrics_dir", "reports_dir"]
        for field_name in path_fields:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

    def validate(self) -> None:
        """Validate metrics configuration."""
        super().validate()

        if not 0.0 < self.confidence_level < 1.0:
            raise InvalidConfigError(
                "confidence_level",
                self.confidence_level,
                "float between 0.0 and 1.0 (exclusive)",
            )

        if self.bootstrap_samples < 100:
            raise InvalidConfigError(
                "bootstrap_samples",
                self.bootstrap_samples,
                "at least 100",
            )


# ---------------------------------------------------------------------------
# Pipeline Configuration
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig(BaseConfig):
    """Configuration for full evaluation pipeline.

    流水线配置 | Pipeline Configuration

    Combines inference and metrics configuration for
    end-to-end benchmark evaluation.
    """

    # Model identification
    model_name: str = ""

    # Pipeline stages
    run_inference: bool = True
    run_metrics: bool = True

    # Paths
    dataset_path: Path | None = None
    output_dir: Path | None = None
    work_dir: Path | None = None

    # Inference settings (flattened for CLI convenience)
    max_tokens: int = 512
    temperature: float = 0.0
    batch_size: int = 1
    backend_type: str = "vllm"

    # Execution control
    resume: bool = False
    overwrite: bool = False
    max_samples: int | None = None

    # Logging
    log_path: Path | None = None
    progress_interval: int = 50

    def __post_init__(self) -> None:
        super().__post_init__()

        # Set default work_dir
        if self.work_dir is None:
            self.work_dir = self.project_root / "outputs"
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)

        # Convert other path fields
        if isinstance(self.dataset_path, str):
            self.dataset_path = Path(self.dataset_path)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.log_path, str):
            self.log_path = Path(self.log_path)

    def validate(self) -> None:
        """Validate pipeline configuration."""
        super().validate()

        if not self.model_name:
            raise MissingConfigError("model_name")

        if not self.run_inference and not self.run_metrics:
            raise InvalidConfigError(
                "run_inference/run_metrics",
                "both False",
                "at least one must be True",
            )

    def to_inference_config(self) -> InferenceConfig:
        """Extract inference configuration."""
        return InferenceConfig(
            project_root=self.project_root,
            model_name=self.model_name,
            backend_type=self.backend_type,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            batch_size=self.batch_size,
            resume=self.resume,
            overwrite=self.overwrite,
            max_samples=self.max_samples,
            output_path=self.output_dir / "predictions.jsonl" if self.output_dir else None,
            log_path=self.log_path,
            log_level=self.log_level,
            seed=self.seed,
        )

    def to_metrics_config(self) -> MetricsConfig:
        """Extract metrics configuration."""
        return MetricsConfig(
            project_root=self.project_root,
            predictions_path=self.output_dir / "predictions.jsonl" if self.output_dir else None,
            metrics_dir=self.output_dir / "metrics" if self.output_dir else None,
            reports_dir=self.output_dir / "reports" if self.output_dir else None,
            log_level=self.log_level,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# Dataset Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig(BaseConfig):
    """Configuration for dataset loading.

    数据集配置 | Dataset Configuration

    Settings for loading and preprocessing benchmark datasets.
    """

    # Paths
    data_root: Path | None = None
    items_path: Path | None = None
    images_dir: Path | None = None
    annotations_path: Path | None = None

    # Preprocessing
    max_samples: int | None = None
    shuffle: bool = False
    filter_invalid: bool = True

    # Caching
    use_cache: bool = True
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        super().__post_init__()

        # Convert string paths
        path_fields = ["data_root", "items_path", "images_dir", "annotations_path", "cache_dir"]
        for field_name in path_fields:
            value = getattr(self, field_name)
            if isinstance(value, str):
                setattr(self, field_name, Path(value))

        # Set default cache_dir
        if self.cache_dir is None and self.data_root is not None:
            self.cache_dir = self.data_root / ".cache"

    def validate(self) -> None:
        """Validate dataset configuration."""
        super().validate()

        if self.items_path is not None and not self.items_path.exists():
            raise FileNotFoundError(f"Dataset items file not found: {self.items_path}")

        if self.max_samples is not None and self.max_samples < 1:
            raise InvalidConfigError(
                "max_samples",
                self.max_samples,
                "positive integer or None",
            )
