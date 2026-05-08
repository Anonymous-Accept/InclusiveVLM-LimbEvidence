"""Logging configuration utilities for all benchmarks.

日志配置工具模块 | Logging Configuration Utilities Module

This module provides centralized logging configuration for all benchmarks,
ensuring consistent log formatting and output across the codebase.

Features:
- Console logging with colored output (if available)
- Optional file logging with rotation
- Configurable log levels
- Named logger instances

Usage:
    >>> from scripts.core import configure_logging, get_logger
    >>> configure_logging("INFO", log_path=Path("run.log"))
    >>> logger = get_logger(__name__)
    >>> logger.info("Processing started")
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

__all__ = [
    "configure_logging",
    "get_logger",
    "LOG_FORMAT_STANDARD",
    "LOG_FORMAT_DETAILED",
    "LOG_FORMAT_MINIMAL",
]

# Standard log formats
LOG_FORMAT_STANDARD = "%(asctime)s - %(levelname)s - %(message)s"
LOG_FORMAT_DETAILED = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
LOG_FORMAT_MINIMAL = "%(levelname)s: %(message)s"


def configure_logging(
    level: str = "INFO",
    format_string: str | None = None,
    log_path: Path | str | None = None,
    force: bool = True,
    add_console: bool = True,
) -> None:
    """Configure logging with console and optional file output.

    配置日志 | Configure Logging

    This is the canonical logging configuration function. All benchmark
    modules should use this instead of defining their own.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        format_string: Custom format string. If None, uses LOG_FORMAT_STANDARD.
        log_path: Optional path for file logging. Parent dirs created automatically.
        force: Whether to force reconfiguration of existing handlers.
        add_console: Whether to add console (stdout) handler.

    Examples:
        Basic console logging:
        >>> configure_logging("INFO")

        With file output:
        >>> configure_logging("DEBUG", log_path=Path("logs/run.log"))

        Custom format:
        >>> configure_logging("INFO", format_string=LOG_FORMAT_DETAILED)

        File only (no console):
        >>> configure_logging("INFO", log_path="run.log", add_console=False)
    """
    if format_string is None:
        format_string = LOG_FORMAT_STANDARD

    handlers: list[logging.Handler] = []

    # Console handler
    if add_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(console_handler)

    # File handler
    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(format_string))
        handlers.append(file_handler)

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=format_string,
        handlers=handlers,
        force=force,
    )


def get_logger(
    name: str,
    level: str | None = None,
) -> logging.Logger:
    """Get a named logger instance.

    获取命名日志器 | Get Named Logger

    Args:
        name: Logger name (typically __name__).
        level: Optional logging level override. If None, inherits from root.

    Returns:
        Configured logger instance.

    Examples:
        >>> logger = get_logger(__name__)
        >>> logger.info("Module loaded")

        >>> debug_logger = get_logger("debug", level="DEBUG")
        >>> debug_logger.debug("Detailed info")
    """
    logger = logging.getLogger(name)
    if level is not None:
        logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def log_exception(
    logger: logging.Logger,
    message: str,
    exc: Exception,
    level: str = "ERROR",
) -> None:
    """Log an exception with context.

    记录异常 | Log Exception

    Args:
        logger: Logger instance.
        message: Context message.
        exc: Exception to log.
        level: Log level (default ERROR).

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_exception(logger, "Operation failed", e)
    """
    log_func = getattr(logger, level.lower(), logger.error)
    log_func(f"{message}: {type(exc).__name__}: {exc}", exc_info=True)
