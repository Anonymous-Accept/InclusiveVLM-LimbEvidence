"""Model registry for vision-language benchmark.

This module re-exports the unified model registry from scripts.core
with any benchmark-specific additions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

__all__ = [
    "ModelConfig",
    "MODEL_REGISTRY",
    "get_model_config",
    "resolve_base_url",
    "VisionLanguageModel",
    "create_model",
    "DEFAULT_API_HOST",
    "DEFAULT_API_PORT",
    "DEFAULT_API_BASE",
    # Build functions for backward compatibility
    "build_model",
    "build_model_config",
    "build_model_from_config",
]

# Re-export backend factory
from scripts.core.backends import create_backend

# Re-export from unified core registry
from scripts.core.model_registry import (
    DEFAULT_API_BASE,
    DEFAULT_API_HOST,
    DEFAULT_API_PORT,
    MODEL_REGISTRY,
    ModelConfig,
    VisionLanguageModel,
    build_model,
    build_model_config,
    build_model_from_config,
    get_model_config,
    resolve_base_url,
)


def create_model(
    model_name: str,
    mode: str = "server",
    **kwargs: Any,
) -> VisionLanguageModel:
    """
    Create a vision-language model instance.

    Args:
        model_name: Registered model name.
        mode: Backend mode ("server" or "local").
        **kwargs: Additional backend configuration.

    Returns:
        VisionLanguageModel instance.

    Example:
        >>> model = create_model("qwen3-vl-8b-instruct-fp8")
        >>> result = model.generate("image.jpg", "Describe this image.")
    """
    return create_backend(model_name, mode=mode, **kwargs)
