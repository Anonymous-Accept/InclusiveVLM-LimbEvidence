"""Centralized exception handling for benchmark pipelines.

集中式异常处理模块 | Centralized Exception Handling Module

This module provides a unified exception hierarchy for all benchmarks,
enabling better error reporting, debugging, and user-friendly messages.

Exception Hierarchy:
    BenchmarkError (base)
    ├── ConfigurationError      # Configuration/settings issues
    ├── DatasetError            # Dataset loading/validation issues
    │   ├── DatasetNotFoundError
    │   └── DatasetFormatError
    ├── InferenceError          # Model inference issues
    │   ├── ModelNotFoundError
    │   ├── APIError
    │   └── TimeoutError
    ├── EvaluationError         # Metric computation issues
    └── IOError                 # File I/O issues

Example:
    >>> from scripts.core.exceptions import DatasetNotFoundError
    >>> raise DatasetNotFoundError("items.jsonl", search_paths=["/data", "/cache"])
    DatasetNotFoundError: Dataset file not found: items.jsonl
    Searched paths:
      - /data
      - /cache
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

__all__ = [
    # Base
    "BenchmarkError",
    # Configuration
    "ConfigurationError",
    "MissingConfigError",
    "InvalidConfigError",
    # Dataset
    "DatasetError",
    "DatasetNotFoundError",
    "DatasetFormatError",
    "DatasetValidationError",
    # Inference
    "InferenceError",
    "ModelNotFoundError",
    "APIError",
    "APITimeoutError",
    "APIRateLimitError",
    # Evaluation
    "EvaluationError",
    "MetricComputationError",
    # I/O
    "FileIOError",
    "FileNotFoundError",
    "FileParseError",
]


# ---------------------------------------------------------------------------
# Base Exception
# ---------------------------------------------------------------------------


class BenchmarkError(Exception):
    """Base exception for all benchmark operations.

    基准测试基础异常 | Benchmark Base Exception

    All benchmark-specific exceptions inherit from this class,
    enabling easy catching of any benchmark-related error.

    Attributes:
        message: Human-readable error description.
        details: Optional dict with additional context.
    """

    def __init__(
        self,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initialize benchmark error.

        Args:
            message: Error message.
            details: Optional additional context dict.
        """
        self.message = message
        self.details = details or {}
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        """Format the full error message with details."""
        if not self.details:
            return self.message
        detail_lines = [f"  {k}: {v}" for k, v in self.details.items()]
        return f"{self.message}\nDetails:\n" + "\n".join(detail_lines)


# ---------------------------------------------------------------------------
# Configuration Errors
# ---------------------------------------------------------------------------


class ConfigurationError(BenchmarkError):
    """Configuration or settings related errors.

    配置错误 | Configuration Error

    Raised when configuration files are invalid, missing required fields,
    or contain incompatible settings.
    """

    pass


class MissingConfigError(ConfigurationError):
    """Required configuration field is missing.

    缺失配置项 | Missing Config Field

    Example:
        >>> raise MissingConfigError("model_name", config_file="config.yaml")
    """

    def __init__(
        self,
        field_name: str,
        config_file: str | Path | None = None,
        suggestion: str | None = None,
    ) -> None:
        """Initialize missing config error.

        Args:
            field_name: Name of the missing field.
            config_file: Config file path if known.
            suggestion: Suggested fix or default value.
        """
        message = f"Required configuration field missing: '{field_name}'"
        details = {"field": field_name}

        if config_file:
            details["config_file"] = str(config_file)
        if suggestion:
            details["suggestion"] = suggestion

        super().__init__(message, details)


class InvalidConfigError(ConfigurationError):
    """Configuration value is invalid or out of range.

    无效配置值 | Invalid Config Value
    """

    def __init__(
        self,
        field_name: str,
        value: Any,
        expected: str,
        config_file: str | Path | None = None,
    ) -> None:
        """Initialize invalid config error.

        Args:
            field_name: Name of the invalid field.
            value: The invalid value provided.
            expected: Description of expected value/format.
            config_file: Config file path if known.
        """
        message = f"Invalid configuration value for '{field_name}'"
        details = {
            "field": field_name,
            "got": repr(value),
            "expected": expected,
        }

        if config_file:
            details["config_file"] = str(config_file)

        super().__init__(message, details)


# ---------------------------------------------------------------------------
# Dataset Errors
# ---------------------------------------------------------------------------


class DatasetError(BenchmarkError):
    """Dataset loading or validation errors.

    数据集错误 | Dataset Error

    Base class for all dataset-related errors.
    """

    pass


class DatasetNotFoundError(DatasetError):
    """Dataset file or directory not found.

    数据集未找到 | Dataset Not Found
    """

    def __init__(
        self,
        path: str | Path,
        search_paths: list[str | Path] | None = None,
        dataset_name: str | None = None,
    ) -> None:
        """Initialize dataset not found error.

        Args:
            path: The path that was not found.
            search_paths: List of paths that were searched.
            dataset_name: Name of the dataset if known.
        """
        if dataset_name:
            message = f"Dataset '{dataset_name}' not found: {path}"
        else:
            message = f"Dataset file not found: {path}"

        details: dict[str, Any] = {"path": str(path)}

        if search_paths:
            details["searched_paths"] = [str(p) for p in search_paths]
        if dataset_name:
            details["dataset"] = dataset_name

        super().__init__(message, details)


class DatasetFormatError(DatasetError):
    """Dataset file has invalid format.

    数据集格式错误 | Dataset Format Error
    """

    def __init__(
        self,
        path: str | Path,
        expected_format: str,
        actual_format: str | None = None,
        line_number: int | None = None,
    ) -> None:
        """Initialize dataset format error.

        Args:
            path: Path to the problematic file.
            expected_format: Expected format (e.g., "JSONL", "COCO JSON").
            actual_format: Detected format if known.
            line_number: Line number where error occurred (for line-based formats).
        """
        message = f"Invalid dataset format in: {path}"
        details: dict[str, Any] = {
            "path": str(path),
            "expected_format": expected_format,
        }

        if actual_format:
            details["actual_format"] = actual_format
        if line_number:
            details["line_number"] = line_number

        super().__init__(message, details)


class DatasetValidationError(DatasetError):
    """Dataset content validation failed.

    数据集验证失败 | Dataset Validation Failed
    """

    def __init__(
        self,
        message: str,
        missing_fields: list[str] | None = None,
        invalid_records: list[int] | None = None,
    ) -> None:
        """Initialize dataset validation error.

        Args:
            message: Validation error description.
            missing_fields: List of missing required fields.
            invalid_records: List of invalid record indices.
        """
        details = {}

        if missing_fields:
            details["missing_fields"] = missing_fields
        if invalid_records:
            details["invalid_record_indices"] = invalid_records[:10]  # Limit to 10
            if len(invalid_records) > 10:
                details["total_invalid"] = len(invalid_records)

        super().__init__(message, details)


# ---------------------------------------------------------------------------
# Inference Errors
# ---------------------------------------------------------------------------


class InferenceError(BenchmarkError):
    """Model inference related errors.

    推理错误 | Inference Error

    Base class for model execution and API errors.
    """

    pass


class ModelNotFoundError(InferenceError):
    """Specified model not found in registry or on disk.

    模型未找到 | Model Not Found
    """

    def __init__(
        self,
        model_name: str,
        available_models: list[str] | None = None,
    ) -> None:
        """Initialize model not found error.

        Args:
            model_name: Name of the model that was not found.
            available_models: List of available model names.
        """
        message = f"Model not found: '{model_name}'"
        details: dict[str, Any] = {"model_name": model_name}

        if available_models:
            details["available_models"] = available_models[:20]  # Limit display

        super().__init__(message, details)


class APIError(InferenceError):
    """External API call failed.

    API 调用错误 | API Error
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        api_name: str | None = None,
        response_body: str | None = None,
    ) -> None:
        """Initialize API error.

        Args:
            message: Error message.
            status_code: HTTP status code if applicable.
            api_name: Name of the API (e.g., "OpenAI", "Gemini").
            response_body: Raw response body for debugging.
        """
        details = {}

        if status_code:
            details["status_code"] = status_code
        if api_name:
            details["api"] = api_name
        if response_body:
            # Truncate long responses
            truncated = response_body[:500] + "..." if len(response_body) > 500 else response_body
            details["response"] = truncated

        super().__init__(message, details)


class APITimeoutError(InferenceError):
    """API request timed out.

    API 超时 | API Timeout
    """

    def __init__(
        self,
        timeout_seconds: float,
        api_name: str | None = None,
        model_name: str | None = None,
    ) -> None:
        """Initialize API timeout error.

        Args:
            timeout_seconds: Configured timeout value.
            api_name: Name of the API.
            model_name: Model being called.
        """
        message = f"API request timed out after {timeout_seconds}s"
        details = {"timeout": f"{timeout_seconds}s"}

        if api_name:
            details["api"] = api_name
        if model_name:
            details["model"] = model_name

        super().__init__(message, details)


class APIRateLimitError(InferenceError):
    """API rate limit exceeded.

    API 速率限制 | API Rate Limit
    """

    def __init__(
        self,
        retry_after: float | None = None,
        api_name: str | None = None,
    ) -> None:
        """Initialize rate limit error.

        Args:
            retry_after: Seconds to wait before retrying.
            api_name: Name of the API.
        """
        message = "API rate limit exceeded"
        details = {}

        if retry_after:
            details["retry_after"] = f"{retry_after}s"
        if api_name:
            details["api"] = api_name

        super().__init__(message, details)


# ---------------------------------------------------------------------------
# Evaluation Errors
# ---------------------------------------------------------------------------


class EvaluationError(BenchmarkError):
    """Metric computation or evaluation errors.

    评估错误 | Evaluation Error
    """

    pass


class MetricComputationError(EvaluationError):
    """Failed to compute metrics.

    指标计算失败 | Metric Computation Failed
    """

    def __init__(
        self,
        metric_name: str,
        reason: str,
        sample_count: int | None = None,
    ) -> None:
        """Initialize metric computation error.

        Args:
            metric_name: Name of the metric that failed.
            reason: Why computation failed.
            sample_count: Number of samples involved.
        """
        message = f"Failed to compute metric '{metric_name}': {reason}"
        details: dict[str, Any] = {"metric": metric_name, "reason": reason}

        if sample_count is not None:
            details["sample_count"] = sample_count

        super().__init__(message, details)


# ---------------------------------------------------------------------------
# File I/O Errors
# ---------------------------------------------------------------------------


class FileIOError(BenchmarkError):
    """File I/O operation failed.

    文件 I/O 错误 | File I/O Error
    """

    pass


class FileNotFoundError(FileIOError):
    """File not found at specified path.

    文件未找到 | File Not Found

    Note: This shadows the builtin FileNotFoundError.
    Use builtins.FileNotFoundError if you need the builtin.
    """

    def __init__(
        self,
        path: str | Path,
        operation: str = "read",
    ) -> None:
        """Initialize file not found error.

        Args:
            path: Path that was not found.
            operation: Operation that was attempted.
        """
        message = f"File not found: {path}"
        details = {
            "path": str(path),
            "operation": operation,
        }
        super().__init__(message, details)


class FileParseError(FileIOError):
    """Failed to parse file content.

    文件解析错误 | File Parse Error
    """

    def __init__(
        self,
        path: str | Path,
        file_format: str,
        line_number: int | None = None,
        parse_error: str | None = None,
    ) -> None:
        """Initialize file parse error.

        Args:
            path: Path to the file.
            file_format: Expected format (JSON, YAML, etc.).
            line_number: Line where parsing failed.
            parse_error: Original parse error message.
        """
        message = f"Failed to parse {file_format} file: {path}"
        details: dict[str, Any] = {
            "path": str(path),
            "format": file_format,
        }

        if line_number:
            details["line"] = line_number
        if parse_error:
            details["error"] = parse_error

        super().__init__(message, details)


# ---------------------------------------------------------------------------
# Context Manager for Error Handling
# ---------------------------------------------------------------------------


class error_context:
    """Context manager for enhanced error handling.

    错误上下文管理器 | Error Context Manager

    Wraps code blocks to catch exceptions and re-raise with additional context.

    Example:
        >>> with error_context("Loading dataset", file=path):
        ...     data = load_jsonl(path)

    If an error occurs, it will include the context information.
    """

    def __init__(self, operation: str, **context: Any) -> None:
        """Initialize error context.

        Args:
            operation: Description of the operation being performed.
            **context: Additional context key-value pairs.
        """
        self.operation = operation
        self.context = context

    def __enter__(self) -> error_context:
        return self

    def __exit__(self, exc_type: type | None, exc_val: Exception | None, exc_tb: Any) -> bool:
        if exc_val is None:
            return False

        # If already a BenchmarkError, add context
        if isinstance(exc_val, BenchmarkError):
            exc_val.details["operation"] = self.operation
            exc_val.details.update(self.context)
            return False

        # Wrap other exceptions
        wrapped = BenchmarkError(
            f"{self.operation} failed: {exc_val}",
            details={
                "operation": self.operation,
                "original_error": type(exc_val).__name__,
                **self.context,
            },
        )
        raise wrapped from exc_val
