"""Package-level evaluation defaults shared across backends and sub-benchmarks."""

from __future__ import annotations

__all__ = [
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_SEED",
]

DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 64
DEFAULT_SEED = 42
