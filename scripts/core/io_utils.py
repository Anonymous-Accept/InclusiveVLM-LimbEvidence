"""I/O utilities for JSON, JSONL, and YAML files.

I/O 工具模块 | I/O Utilities Module

This module provides unified file I/O operations for all benchmarks:
- JSON file loading/saving
- JSONL (JSON Lines) streaming
- YAML configuration files
- Caching support for repeated reads
- Robust error handling

All I/O functions in other modules should import from here to ensure
consistent behavior and error handling.

Example:
    >>> from scripts.core.io_utils import load_jsonl, save_json
    >>> items = load_jsonl("data/items.jsonl")
    >>> save_json({"count": len(items)}, "stats.json")
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path
from typing import Any

from .exceptions import FileNotFoundError, FileParseError

__all__ = [
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "iter_jsonl",
    "append_jsonl",
    "load_yaml",
    "save_yaml",
    "load_jsonl_cached",
    "load_json_cached",
    "clear_io_cache",
]


# ---------------------------------------------------------------------------
# JSON Operations
# ---------------------------------------------------------------------------


def load_json(path: str | Path) -> Any:
    """Load a JSON file.

    加载 JSON 文件 | Load JSON File

    Args:
        path: Path to the JSON file.

    Returns:
        Parsed JSON content (dict, list, or primitive).

    Raises:
        FileNotFoundError: If the file does not exist.
        FileParseError: If the file is not valid JSON.

    Example:
        >>> data = load_json("config.json")
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path, operation="load_json")

    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise FileParseError(path, "JSON", parse_error=str(e)) from e


def save_json(
    data: Any,
    path: str | Path,
    indent: int = 2,
    ensure_ascii: bool = False,
) -> None:
    """Save data to a JSON file.

    保存 JSON 文件 | Save JSON File

    Args:
        data: Data to serialize (must be JSON-serializable).
        path: Output file path.
        indent: Indentation level for pretty-printing.
        ensure_ascii: Whether to escape non-ASCII characters.

    Example:
        >>> save_json({"key": "value"}, "output.json")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=ensure_ascii)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load all records from a JSONL file.

    加载 JSONL 文件 | Load JSONL File

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Raises:
        FileNotFoundError: If the file does not exist.
        FileParseError: If a line is not valid JSON.

    Example:
        >>> records = load_jsonl("data.jsonl")
        >>> print(len(records))
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path, operation="load_jsonl")

    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise FileParseError(
                        path, "JSONL", line_number=line_num, parse_error=str(e)
                    ) from e
    except FileParseError:
        raise
    except Exception as e:
        raise FileParseError(path, "JSONL", parse_error=str(e)) from e

    return records


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """
    Iterate over records in a JSONL file (memory-efficient).

    Args:
        path: Path to the JSONL file.

    Yields:
        Parsed JSON objects one at a time.

    Example:
        >>> for record in iter_jsonl("large_data.jsonl"):
        ...     process(record)
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def save_jsonl(
    records: list[dict[str, Any]],
    path: str | Path,
    ensure_ascii: bool = False,
) -> None:
    """
    Save records to a JSONL file.

    Args:
        records: List of JSON-serializable dictionaries.
        path: Output file path.
        ensure_ascii: Whether to escape non-ASCII characters.

    Example:
        >>> save_jsonl([{"id": 1}, {"id": 2}], "output.jsonl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            line = json.dumps(record, ensure_ascii=ensure_ascii)
            f.write(line + "\n")


def append_jsonl(
    record: dict[str, Any],
    path: str | Path,
    ensure_ascii: bool = False,
) -> None:
    """
    Append a single record to a JSONL file.

    Args:
        record: JSON-serializable dictionary.
        path: Output file path.
        ensure_ascii: Whether to escape non-ASCII characters.

    Example:
        >>> append_jsonl({"id": 3}, "output.jsonl")
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        line = json.dumps(record, ensure_ascii=ensure_ascii)
        f.write(line + "\n")


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If PyYAML is not installed.

    Example:
        >>> config = load_yaml("config.yaml")
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for YAML support. Install with: pip install pyyaml"
        ) from e

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(
    data: dict[str, Any],
    path: str | Path,
    default_flow_style: bool = False,
) -> None:
    """
    Save data to a YAML file.

    Args:
        data: Dictionary to serialize.
        path: Output file path.
        default_flow_style: Whether to use flow style (inline) formatting.

    Raises:
        ImportError: If PyYAML is not installed.

    Example:
        >>> save_yaml({"key": "value"}, "output.yaml")
    """
    try:
        import yaml
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for YAML support. Install with: pip install pyyaml"
        ) from e

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=default_flow_style)

# ---------------------------------------------------------------------------
# Cached Loading Functions
# ---------------------------------------------------------------------------


@lru_cache(maxsize=16)
def _load_jsonl_cached_impl(path_str: str) -> tuple[dict[str, Any], ...]:
    """Internal cached implementation (returns tuple for hashability)."""
    return tuple(load_jsonl(path_str))


def load_jsonl_cached(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL file with caching for repeated reads.

    缓存加载 JSONL | Cached JSONL Loading

    Use this when the same file will be loaded multiple times
    (e.g., in evaluation pipelines with multiple runs).

    Note: Cache is keyed by absolute path string.

    Args:
        path: Path to the JSONL file.

    Returns:
        List of parsed JSON objects.

    Example:
        >>> items = load_jsonl_cached("data/items.jsonl")
        >>> # Second call returns cached result
        >>> items_again = load_jsonl_cached("data/items.jsonl")
    """
    path = Path(path).resolve()
    return list(_load_jsonl_cached_impl(str(path)))


@lru_cache(maxsize=32)
def load_json_cached(path_str: str) -> Any:
    """Load JSON file with caching for repeated reads.

    缓存加载 JSON | Cached JSON Loading

    Args:
        path_str: Path to the JSON file (as string for hashability).

    Returns:
        Parsed JSON content.

    Example:
        >>> config = load_json_cached(str(Path("config.json").resolve()))
    """
    return load_json(path_str)


def clear_io_cache() -> None:
    """Clear all I/O caches.

    清除 I/O 缓存 | Clear I/O Cache

    Call this when files may have changed on disk.
    """
    _load_jsonl_cached_impl.cache_clear()
    load_json_cached.cache_clear()


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def count_jsonl_lines(path: str | Path) -> int:
    """Count non-empty lines in a JSONL file without loading all data.

    统计 JSONL 行数 | Count JSONL Lines

    Args:
        path: Path to the JSONL file.

    Returns:
        Number of non-empty lines.
    """
    path = Path(path)
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def file_exists(path: str | Path) -> bool:
    """Check if file exists (convenience wrapper).

    检查文件是否存在 | Check File Exists

    Args:
        path: Path to check.

    Returns:
        True if file exists, False otherwise.
    """
    return Path(path).exists()
