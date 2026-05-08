"""Common utilities for benchmark evaluation pipelines.

通用工具模块 | Common Utilities Module

This module provides shared utilities for all benchmarks:
- File I/O operations (legacy, prefer io_utils)
- Path resolution
- Chunking utilities

Note: Progress tracking has moved to scripts.core.progress.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TypeVar

# Re-export progress utilities for backward compatibility
from .progress import ProgressTracker, format_duration, format_eta

__all__ = [
    "load_jsonl",
    "save_jsonl",
    "load_json",
    "save_json",
    "ensure_dir",
    "chunked",
    # Re-exports from progress module
    "ProgressTracker",
    "format_eta",
    "format_duration",
]


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file.

    Args:
        path: Path to JSONL file.

    Returns:
        List of parsed JSON objects.
    """
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(
    records: list[dict[str, Any]],
    path: Path,
    append: bool = False,
    ensure_ascii: bool = False,
) -> None:
    """Save records to a JSONL file.

    Args:
        records: List of JSON-serializable objects.
        path: Output path.
        append: Whether to append to existing file.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with open(path, mode, encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Parsed JSON object.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(
    data: dict[str, Any],
    path: Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Save data to a JSON file.

    Args:
        data: JSON-serializable object.
        path: Output path.
        indent: Indentation level for pretty printing.
        ensure_ascii: Whether to escape non-ASCII characters.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating if necessary.

    Args:
        path: Directory path.

    Returns:
        The same path (for chaining).
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Iteration Utilities
# ---------------------------------------------------------------------------

T = TypeVar("T")


def chunked(iterable: Iterable[T], size: int) -> Iterator[list[T]]:
    """Split iterable into chunks of given size.

    Args:
        iterable: Input iterable.
        size: Maximum chunk size.

    Yields:
        Lists of up to `size` items.

    Example:
        >>> list(chunked([1, 2, 3, 4, 5], 2))
        [[1, 2], [3, 4], [5]]
    """
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
