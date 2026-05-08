"""Base classes for benchmark evaluation pipelines.

基准测试抽象基类 | Benchmark Abstract Base Classes

This module provides abstract base classes for:
- Benchmark configuration (BenchmarkConfig)
- Evaluation pipeline (BaseBenchmark)

All benchmark modules should extend these classes to ensure
consistent interface and behavior.
"""

from __future__ import annotations

import argparse
import logging
import sys
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, TypeVar

__all__ = [
    "BenchmarkConfig",
    "InferenceConfigBase",
    "BaseBenchmark",
    "PredictionRecordBase",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration Base Classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Base configuration for all benchmarks.

    基准配置基类 | Benchmark Configuration Base Class

    Common settings shared across all benchmark types:
    - Model identification
    - Path configuration
    - Logging settings
    - Execution control flags

    Subclasses should add benchmark-specific fields.
    """

    # Model identification
    model_name: str

    # Base paths (computed from project root)
    project_root: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[2]
    )
    work_dir: Path | None = None

    # Logging
    log_level: str = "INFO"
    log_path: Path | None = None

    # Execution control
    resume: bool = False
    overwrite: bool = False
    max_samples: int | None = None
    seed: int = 42

    def __post_init__(self) -> None:
        """Resolve paths after initialization."""
        if self.work_dir is None:
            self.work_dir = self.project_root / "outputs"
        if isinstance(self.work_dir, str):
            self.work_dir = Path(self.work_dir)
        if isinstance(self.project_root, str):
            self.project_root = Path(self.project_root)

    def to_dict(self) -> dict[str, Any]:
        """Convert config to dictionary (for serialization)."""
        d = asdict(self)
        # Convert Path objects to strings
        for key, value in d.items():
            if isinstance(value, Path):
                d[key] = str(value)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkConfig:
        """Create config from dictionary."""
        # Convert string paths back to Path objects
        path_fields = {"project_root", "work_dir", "log_path"}
        for key in path_fields:
            if key in data and data[key] is not None:
                data[key] = Path(data[key])
        return cls(**data)


@dataclass
class InferenceConfigBase(BenchmarkConfig):
    """Base configuration for inference pipelines.

    推理配置基类 | Inference Configuration Base Class

    Extends BenchmarkConfig with inference-specific settings:
    - Model backend configuration
    - Generation parameters
    - Batch processing settings
    """

    # Model backend
    vllm_mode: str = "server"  # "server" or "local"
    backend_type: str = "vllm"  # "vllm", "openai", "gemini"

    # Generation parameters
    max_tokens: int = 256
    temperature: float = 0.0
    top_p: float = 1.0

    # Batch processing
    batch_size: int = 1
    max_concurrent: int = 8

    # Output constraints
    assistant_prefill: bool = False
    guided_decoding: bool = False


@dataclass
class PredictionRecordBase:
    """Base class for prediction records.

    预测记录基类 | Prediction Record Base Class

    Common fields for all prediction types:
    - item_id: Unique identifier for the item
    - model_id: Model that made the prediction
    - raw_text: Raw model output
    - parse_error: Whether parsing failed
    - latency_ms: Inference latency in milliseconds
    - timestamp: When prediction was made
    """

    item_id: str
    model_id: str
    raw_text: str
    parse_error: int = 0
    latency_ms: int = 0
    timestamp: str = ""

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Benchmark Pipeline Base Class
# ---------------------------------------------------------------------------

ConfigT = TypeVar("ConfigT", bound=BenchmarkConfig)
PredictionT = TypeVar("PredictionT", bound=PredictionRecordBase)


class BaseBenchmark(ABC, Generic[ConfigT, PredictionT]):
    """Abstract base class for benchmark evaluation pipelines.

    基准评测流水线基类 | Benchmark Evaluation Pipeline Base Class

    Provides common infrastructure for:
    - Logging configuration
    - Directory management
    - Progress tracking
    - Result persistence

    Subclasses must implement:
    - run_inference(): Execute model inference
    - compute_metrics(): Calculate evaluation metrics
    - parse_args(): Parse command line arguments
    """

    # Class-level configuration
    BENCHMARK_NAME: str = "BaseBenchmark"
    BENCHMARK_VERSION: str = "1.0.0"

    def __init__(self, config: ConfigT) -> None:
        """Initialize benchmark with configuration.

        Args:
            config: Benchmark configuration object.
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.BENCHMARK_NAME}")
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging for the benchmark."""
        handlers = [logging.StreamHandler(sys.stdout)]

        if self.config.log_path is not None:
            self.config.log_path.parent.mkdir(parents=True, exist_ok=True)
            handlers.append(logging.FileHandler(self.config.log_path, encoding="utf-8"))

        logging.basicConfig(
            level=getattr(logging, self.config.log_level.upper(), logging.INFO),
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            handlers=handlers,
            force=True,
        )

    def _ensure_directories(self) -> None:
        """Ensure output directories exist."""
        if self.config.work_dir:
            self.config.work_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run_inference(self) -> list[PredictionT]:
        """Run model inference on benchmark items.

        Returns:
            List of prediction records.
        """
        pass

    @abstractmethod
    def compute_metrics(self, predictions: list[PredictionT]) -> dict[str, Any]:
        """Compute evaluation metrics from predictions.

        Args:
            predictions: List of prediction records.

        Returns:
            Dictionary of metric names to values.
        """
        pass

    def run(self) -> dict[str, Any]:
        """Run full evaluation pipeline (inference + metrics).

        Returns:
            Dictionary containing:
            - predictions: List of prediction dicts
            - metrics: Dict of computed metrics
            - manifest: Run metadata
        """
        self._ensure_directories()

        self.logger.info("=" * 70)
        self.logger.info(f"{self.BENCHMARK_NAME} Evaluation Pipeline")
        self.logger.info("=" * 70)
        self.logger.info(f"Model: {self.config.model_name}")
        self.logger.info(f"Config: {self.config.to_dict()}")

        # Run inference
        self.logger.info("Starting inference...")
        predictions = self.run_inference()
        self.logger.info(f"Completed {len(predictions)} predictions.")

        # Compute metrics
        self.logger.info("Computing metrics...")
        metrics = self.compute_metrics(predictions)
        self.logger.info(f"Metrics: {metrics}")

        # Create manifest
        manifest = {
            "benchmark_name": self.BENCHMARK_NAME,
            "benchmark_version": self.BENCHMARK_VERSION,
            "model_name": self.config.model_name,
            "timestamp": datetime.now().isoformat(),
            "config": self.config.to_dict(),
            "total_predictions": len(predictions),
            "metrics_summary": {
                k: v for k, v in metrics.items() if isinstance(v, (int, float, str))
            },
        }

        self.logger.info("=" * 70)
        self.logger.info("Evaluation complete!")
        self.logger.info("=" * 70)

        return {
            "predictions": [p.to_dict() for p in predictions],
            "metrics": metrics,
            "manifest": manifest,
        }

    @classmethod
    @abstractmethod
    def parse_args(cls) -> argparse.Namespace:
        """Parse command line arguments.

        Returns:
            Parsed arguments namespace.
        """
        pass

    @classmethod
    def create_base_parser(cls, description: str) -> argparse.ArgumentParser:
        """Create base argument parser with common options.

        Args:
            description: Parser description.

        Returns:
            ArgumentParser with common options.
        """
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )

        # Required
        parser.add_argument(
            "--model-name",
            required=True,
            help="Model key in registry.",
        )

        # Common options
        parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            help="Limit number of samples (for debugging).",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume from checkpoint.",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing outputs.",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="Random seed. Default: 42",
        )
        parser.add_argument(
            "--log-level",
            type=str,
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR"],
            help="Logging level.",
        )
        parser.add_argument(
            "--log-path",
            type=Path,
            default=None,
            help="Log file path.",
        )

        return parser

    @classmethod
    def add_inference_args(cls, parser: argparse.ArgumentParser) -> None:
        """Add inference-specific arguments to parser.

        Args:
            parser: ArgumentParser to extend.
        """
        inference_group = parser.add_argument_group("Inference Options")

        inference_group.add_argument(
            "--vllm-mode",
            type=str,
            choices=["server", "local"],
            default="server",
            help="vLLM mode. Default: server",
        )
        inference_group.add_argument(
            "--max-tokens",
            type=int,
            default=256,
            help="Maximum tokens for output. Default: 256",
        )
        inference_group.add_argument(
            "--temperature",
            type=float,
            default=0.0,
            help="Sampling temperature. Default: 0.0",
        )
        inference_group.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batch size for parallel inference. Default: 1",
        )
        inference_group.add_argument(
            "--max-concurrent",
            type=int,
            default=8,
            help="Max concurrent requests. Default: 8",
        )
        inference_group.add_argument(
            "--assistant-prefill",
            action="store_true",
            help="Enable assistant prefill for output constraints.",
        )
        inference_group.add_argument(
            "--guided-decoding",
            action="store_true",
            help="Enable guided decoding for output constraints.",
        )
