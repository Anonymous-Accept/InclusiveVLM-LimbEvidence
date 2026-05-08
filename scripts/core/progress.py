"""Progress reporting utilities for long-running tasks.

进度报告工具 | Progress Reporting Utilities

This module provides unified progress tracking and performance monitoring
for all benchmark pipelines, with support for:
- Logging-based progress reports (ProgressReporter)
- Rich/tqdm progress bars (UnifiedProgressBar)
- Performance timing and metrics (PerfTimer, PerfStats)
- ETA estimation with exponential moving average

Example:
    >>> from scripts.core.progress import ProgressReporter, PerfTimer
    >>> # Logging-based progress
    >>> with ProgressReporter(total=1000, prefix="Inference") as reporter:
    ...     for i, item in enumerate(items):
    ...         process(item)
    ...         reporter.update(i + 1)
    >>>
    >>> # Performance timing
    >>> with PerfTimer("model_inference") as timer:
    ...     result = model.generate(prompt, image)
    >>> print(f"Took {timer.elapsed:.2f}s")
"""

from __future__ import annotations

import logging
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Generator, Iterable, Iterator, TypeVar

__all__ = [
    "ProgressReporter",
    "ProgressTracker",
    "UnifiedProgressBar",
    "PerfTimer",
    "PerfStats",
    "format_eta",
    "format_duration",
    "format_rate",
    "track_progress",
]

LOGGER = logging.getLogger(__name__)
T = TypeVar("T")


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.

    Args:
        seconds: Duration in seconds.

    Returns:
        Formatted string like "1h 23m 45s" or "45.2s".

    Example:
        >>> format_duration(3725.5)
        '1h 2m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def format_eta(elapsed: float, completed: int, total: int) -> str:
    """
    Calculate and format estimated time of arrival.

    Args:
        elapsed: Elapsed time in seconds.
        completed: Number of completed items.
        total: Total number of items.

    Returns:
        Formatted ETA string or "N/A" if cannot be calculated.

    Example:
        >>> format_eta(60.0, 10, 100)
        '9m 0s'
    """
    if completed <= 0:
        return "N/A"
    rate = elapsed / completed
    remaining = (total - completed) * rate
    return format_duration(remaining)


class ProgressReporter:
    """
    Progress reporter for long-running tasks with ETA estimation.

    Attributes:
        total: Total number of items to process.
        interval: Number of items between progress reports.
        prefix: Prefix string for log messages.

    Example:
        >>> reporter = ProgressReporter(total=1000, interval=100)
        >>> for i, item in enumerate(items):
        ...     process(item)
        ...     reporter.update(i + 1)
        >>> reporter.finish()
    """

    def __init__(
        self,
        total: int,
        interval: int = 100,
        prefix: str = "Progress",
        logger: logging.Logger | None = None,
    ) -> None:
        """
        Initialize the progress reporter.

        Args:
            total: Total number of items to process.
            interval: Number of items between progress reports.
            prefix: Prefix string for log messages.
            logger: Logger instance. If None, uses module logger.
        """
        self.total = total
        self.interval = interval
        self.prefix = prefix
        self.logger = logger or LOGGER
        self._start_time: float | None = None
        self._last_report: int = 0

    def start(self) -> None:
        """Start the progress timer."""
        self._start_time = time.time()
        self._last_report = 0
        self.logger.info("%s: Starting (%d items)", self.prefix, self.total)

    def update(self, completed: int, force: bool = False) -> None:
        """
        Update progress and optionally log status.

        Args:
            completed: Number of completed items.
            force: Force logging even if interval not reached.
        """
        if self._start_time is None:
            self.start()

        if force or completed - self._last_report >= self.interval:
            elapsed = time.time() - self._start_time
            pct = (completed / self.total) * 100 if self.total > 0 else 0
            eta = format_eta(elapsed, completed, self.total)
            self.logger.info(
                "%s: %d/%d (%.1f%%) - Elapsed: %s - ETA: %s",
                self.prefix,
                completed,
                self.total,
                pct,
                format_duration(elapsed),
                eta,
            )
            self._last_report = completed

    def finish(self) -> float:
        """
        Mark progress as complete and log final status.

        Returns:
            Total elapsed time in seconds.
        """
        if self._start_time is None:
            return 0.0

        elapsed = time.time() - self._start_time
        self.logger.info(
            "%s: Complete (%d items in %s)",
            self.prefix,
            self.total,
            format_duration(elapsed),
        )
        return elapsed

    def __enter__(self) -> "ProgressReporter":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.finish()


# ---------------------------------------------------------------------------
# Progress Tracker with EMA
# ---------------------------------------------------------------------------


class ProgressTracker:
    """
    Lightweight progress tracker with EMA-based rate estimation.

    轻量级进度跟踪器 | Lightweight Progress Tracker

    Uses exponential moving average for smoother ETA estimates.

    Attributes:
        total: Total items to process.
        current: Current completed count.
        ema_alpha: Smoothing factor for EMA (default: 0.1).

    Example:
        >>> tracker = ProgressTracker(total=100)
        >>> for i in range(100):
        ...     do_work()
        ...     tracker.tick()
        ...     print(tracker.status())
    """

    def __init__(
        self,
        total: int,
        ema_alpha: float = 0.1,
    ) -> None:
        """Initialize progress tracker.

        Args:
            total: Total number of items.
            ema_alpha: EMA smoothing factor (0-1). Lower = smoother.
        """
        self.total = total
        self.current = 0
        self.ema_alpha = ema_alpha
        self._start_time: float = time.time()
        self._last_time: float = self._start_time
        self._ema_rate: float | None = None

    def tick(self, n: int = 1) -> None:
        """Update progress by n items."""
        now = time.time()
        delta = now - self._last_time

        if delta > 0 and n > 0:
            instant_rate = n / delta
            if self._ema_rate is None:
                self._ema_rate = instant_rate
            else:
                self._ema_rate = (
                    self.ema_alpha * instant_rate + (1 - self.ema_alpha) * self._ema_rate
                )

        self.current += n
        self._last_time = now

    @property
    def elapsed(self) -> float:
        """Total elapsed time in seconds."""
        return time.time() - self._start_time

    @property
    def percent(self) -> float:
        """Completion percentage (0-100)."""
        return (self.current / self.total * 100) if self.total > 0 else 0.0

    @property
    def rate(self) -> float:
        """Estimated rate (items/second)."""
        if self._ema_rate is not None:
            return self._ema_rate
        if self.elapsed > 0:
            return self.current / self.elapsed
        return 0.0

    @property
    def eta_seconds(self) -> float | None:
        """Estimated time remaining in seconds."""
        remaining = self.total - self.current
        if remaining <= 0:
            return 0.0
        if self.rate > 0:
            return remaining / self.rate
        return None

    @property
    def eta(self) -> str:
        """Formatted ETA string."""
        eta_sec = self.eta_seconds
        if eta_sec is None:
            return "N/A"
        return format_duration(eta_sec)

    def status(self) -> str:
        """Get formatted status string."""
        return (
            f"{self.current}/{self.total} ({self.percent:.1f}%) "
            f"[{format_rate(self.rate)}, ETA: {self.eta}]"
        )


def format_rate(rate: float) -> str:
    """Format rate as items/second or items/minute.

    Args:
        rate: Rate in items per second.

    Returns:
        Formatted rate string.
    """
    if rate >= 1.0:
        return f"{rate:.1f} it/s"
    elif rate > 0:
        return f"{rate * 60:.1f} it/min"
    return "0 it/s"


# ---------------------------------------------------------------------------
# Unified Progress Bar (Rich/tqdm backend)
# ---------------------------------------------------------------------------


class UnifiedProgressBar:
    """
    Unified progress bar supporting rich and tqdm backends.

    统一进度条 | Unified Progress Bar

    Automatically selects the best available backend (rich > tqdm > logging).

    Example:
        >>> with UnifiedProgressBar(total=100, desc="Processing") as pbar:
        ...     for item in items:
        ...         process(item)
        ...         pbar.update(1)
    """

    def __init__(
        self,
        total: int,
        desc: str = "Progress",
        backend: str | None = None,
        disable: bool = False,
        leave: bool = True,
        unit: str = "it",
        **kwargs: Any,
    ) -> None:
        """Initialize unified progress bar.

        Args:
            total: Total items to process.
            desc: Description label.
            backend: Force backend ("rich", "tqdm", "logging"). Auto if None.
            disable: Disable progress display.
            leave: Keep bar on screen after completion.
            unit: Unit name for display.
            **kwargs: Additional backend-specific options.
        """
        self.total = total
        self.desc = desc
        self.disable = disable
        self.leave = leave
        self.unit = unit
        self._backend_name = backend
        self._backend: Any = None
        self._kwargs = kwargs
        self._current = 0
        self._start_time: float | None = None

    def _init_backend(self) -> None:
        """Initialize the progress bar backend."""
        if self.disable:
            return

        backend = self._backend_name

        # Auto-detect backend
        if backend is None:
            if _has_rich():
                backend = "rich"
            elif _has_tqdm():
                backend = "tqdm"
            else:
                backend = "logging"

        self._backend_name = backend

        if backend == "rich":
            self._init_rich()
        elif backend == "tqdm":
            self._init_tqdm()
        else:
            self._init_logging()

    def _init_rich(self) -> None:
        """Initialize Rich progress bar."""
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        self._backend = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=not self.leave,
        )
        self._backend.start()
        self._task_id = self._backend.add_task(self.desc, total=self.total)

    def _init_tqdm(self) -> None:
        """Initialize tqdm progress bar."""
        from tqdm import tqdm

        self._backend = tqdm(
            total=self.total,
            desc=self.desc,
            leave=self.leave,
            unit=self.unit,
            **self._kwargs,
        )

    def _init_logging(self) -> None:
        """Initialize logging-based progress (fallback)."""
        self._backend = ProgressReporter(
            total=self.total,
            interval=max(1, self.total // 20),  # ~5% intervals
            prefix=self.desc,
        )
        self._backend.start()

    def update(self, n: int = 1) -> None:
        """Update progress by n items."""
        self._current += n
        if self._backend is None or self.disable:
            return

        if self._backend_name == "rich":
            self._backend.update(self._task_id, advance=n)
        elif self._backend_name == "tqdm":
            self._backend.update(n)
        else:
            self._backend.update(self._current)

    def set_description(self, desc: str) -> None:
        """Update the description text."""
        self.desc = desc
        if self._backend is None or self.disable:
            return

        if self._backend_name == "rich":
            self._backend.update(self._task_id, description=desc)
        elif self._backend_name == "tqdm":
            self._backend.set_description(desc)

    def close(self) -> None:
        """Close the progress bar."""
        if self._backend is None:
            return

        if self._backend_name == "rich":
            self._backend.stop()
        elif self._backend_name == "tqdm":
            self._backend.close()
        else:
            self._backend.finish()

    def __enter__(self) -> "UnifiedProgressBar":
        """Context manager entry."""
        self._start_time = time.time()
        self._init_backend()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


def _has_rich() -> bool:
    """Check if rich is available."""
    try:
        import rich.progress  # noqa: F401

        return True
    except ImportError:
        return False


def _has_tqdm() -> bool:
    """Check if tqdm is available."""
    try:
        import tqdm  # noqa: F401

        return True
    except ImportError:
        return False


def track_progress(
    iterable: Iterable[T],
    total: int | None = None,
    desc: str = "Processing",
    **kwargs: Any,
) -> Iterator[T]:
    """
    Wrap an iterable with a progress bar.

    迭代器进度包装 | Iterable Progress Wrapper

    Args:
        iterable: Iterable to wrap.
        total: Total count (auto-detected if possible).
        desc: Progress bar description.
        **kwargs: Passed to UnifiedProgressBar.

    Yields:
        Items from the iterable.

    Example:
        >>> for item in track_progress(items, desc="Processing"):
        ...     process(item)
    """
    if total is None:
        try:
            total = len(iterable)  # type: ignore
        except TypeError:
            total = 0

    with UnifiedProgressBar(total=total, desc=desc, **kwargs) as pbar:
        for item in iterable:
            yield item
            pbar.update(1)


# ---------------------------------------------------------------------------
# Performance Timing
# ---------------------------------------------------------------------------


@dataclass
class PerfTimer:
    """
    Performance timer for measuring code execution time.

    性能计时器 | Performance Timer

    Can be used as a context manager or manually via start()/stop().

    Attributes:
        name: Timer name for identification.
        elapsed: Elapsed time in seconds (after stop).

    Example:
        >>> with PerfTimer("inference") as timer:
        ...     result = model.generate(prompt)
        >>> print(f"Took {timer.elapsed:.2f}s")
    """

    name: str = "timer"
    elapsed: float = 0.0
    _start: float | None = field(default=None, repr=False)

    def start(self) -> "PerfTimer":
        """Start the timer."""
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed time."""
        if self._start is not None:
            self.elapsed = time.perf_counter() - self._start
            self._start = None
        return self.elapsed

    def __enter__(self) -> "PerfTimer":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.stop()


@dataclass
class PerfStats:
    """
    Performance statistics collector for multiple measurements.

    性能统计收集器 | Performance Statistics Collector

    Collects timing data and computes statistics (mean, min, max, etc.).

    Example:
        >>> stats = PerfStats()
        >>> for batch in batches:
        ...     with stats.timer("inference"):
        ...         result = model(batch)
        ...     with stats.timer("postprocess"):
        ...         output = postprocess(result)
        >>> print(stats.summary())
    """

    _timings: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    _counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @contextmanager
    def timer(self, name: str) -> Generator[PerfTimer, None, None]:
        """Context manager to time a named operation.

        Args:
            name: Operation name.

        Yields:
            PerfTimer instance.
        """
        t = PerfTimer(name)
        t.start()
        try:
            yield t
        finally:
            t.stop()
            self._timings[name].append(t.elapsed)
            self._counts[name] += 1

    def record(self, name: str, value: float) -> None:
        """Manually record a timing value.

        Args:
            name: Metric name.
            value: Timing value in seconds.
        """
        self._timings[name].append(value)
        self._counts[name] += 1

    def get_stats(self, name: str) -> dict[str, float]:
        """Get statistics for a named metric.

        Args:
            name: Metric name.

        Returns:
            Dict with count, total, mean, min, max.
        """
        values = self._timings.get(name, [])
        if not values:
            return {"count": 0, "total": 0, "mean": 0, "min": 0, "max": 0}

        return {
            "count": len(values),
            "total": sum(values),
            "mean": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    def summary(self) -> str:
        """Get formatted summary of all metrics.

        Returns:
            Multi-line summary string.
        """
        lines = ["Performance Summary:"]
        for name in sorted(self._timings.keys()):
            s = self.get_stats(name)
            lines.append(
                f"  {name}: {s['count']} calls, "
                f"total={s['total']:.2f}s, "
                f"mean={s['mean']*1000:.1f}ms, "
                f"min={s['min']*1000:.1f}ms, "
                f"max={s['max']*1000:.1f}ms"
            )
        return "\n".join(lines)

    def to_dict(self) -> dict[str, dict[str, float]]:
        """Export all statistics as a dictionary."""
        return {name: self.get_stats(name) for name in self._timings.keys()}
