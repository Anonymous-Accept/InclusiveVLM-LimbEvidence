"""
Unified backends for vision-language model inference.

This module provides backends for:
- vLLM (server and local modes)
- OpenAI GPT vision models
- Google Gemini multimodal models

Example:
    >>> from scripts.core.backends import create_backend
    >>> backend = create_backend("qwen3-vl-8b-instruct-fp8")
    >>> result = backend.generate("image.jpg", "Describe this image.")
"""

from __future__ import annotations

__all__ = [
    "BaseBackend",
    "VLLMBackend",
    "OpenAIBackend",
    "GeminiBackend",
    "create_backend",
    "BackendError",
]

from .base import BackendError, BaseBackend
from .gemini_backend import GeminiBackend
from .openai_backend import OpenAIBackend
from .vllm_backend import VLLMBackend


def create_backend(
    model_name: str,
    mode: str = "server",
    **kwargs,
):
    """
    Create a backend instance for the specified model.

    Args:
        model_name: Registered model name.
        mode: Backend mode ("server" for API, "local" for local vLLM).
        **kwargs: Additional backend configuration.

    Returns:
        Backend instance (VLLMBackend, OpenAIBackend, or GeminiBackend).

    Raises:
        ValueError: If model backend is unsupported.

    Example:
        >>> backend = create_backend("qwen3-vl-8b-instruct-fp8", mode="server")
        >>> result = backend.generate("image.jpg", "Describe this image.")
    """
    from ..model_registry import get_model_config

    config = get_model_config(model_name)
    if kwargs:
        config = dict(config)
        for key in kwargs:
            config.pop(key, None)
    backend_type = config.get("backend", "vllm")

    if backend_type == "vllm":
        return VLLMBackend(model_name=model_name, mode=mode, **config, **kwargs)
    elif backend_type == "openai":
        return OpenAIBackend(model_name=model_name, **config, **kwargs)
    elif backend_type == "gemini":
        return GeminiBackend(model_name=model_name, **config, **kwargs)
    else:
        raise ValueError(f"Unsupported backend type: {backend_type}")
