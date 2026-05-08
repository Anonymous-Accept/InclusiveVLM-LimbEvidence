"""
Prosthesis Matching benchmark code.

This package evaluates benchmark-defined compatibility-style visual option
matching from visible body-device configuration evidence. It is a diagnostic
evaluation component, not prosthesis recommendation or clinical suitability
tooling.
"""

from __future__ import annotations

__all__ = [
    "build_options_pool",
    "build_prosthesis_match_items",
    "render_option_grids",
    "run_inference",
    "eval_prosthesis_match",
    "evaluate",
    "config",
    "dataset_adapter",
]
