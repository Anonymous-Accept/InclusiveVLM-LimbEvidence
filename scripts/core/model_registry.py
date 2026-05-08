"""Unified model configuration registry for all benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlparse, urlunparse

__all__ = [
    "ModelConfig",
    "MODEL_REGISTRY",
    "get_model_config",
    "resolve_base_url",
    "VisionLanguageModel",
    "DEFAULT_API_HOST",
    "DEFAULT_API_PORT",
    "DEFAULT_API_BASE",
    # Build functions for backward compatibility
    "build_model",
    "build_model_config",
    "build_model_from_config",
    # Listing utilities
    "list_models_by_backend",
    "list_models_by_family",
]

# ---------------------------------------------------------------------------
# Environment-based defaults for vLLM server
# ---------------------------------------------------------------------------

DEFAULT_API_HOST = os.environ.get("VLLM_API_HOST", "localhost")
try:
    DEFAULT_API_PORT = int(os.environ.get("VLLM_API_PORT", "8000"))
except ValueError:
    DEFAULT_API_PORT = 8000
DEFAULT_API_BASE = os.environ.get(
    "VLLM_API_BASE", f"http://{DEFAULT_API_HOST}:{DEFAULT_API_PORT}/v1"
)
LOCAL_VLLM_HOSTS = {"localhost", "127.0.0.1", "::1"}


# ---------------------------------------------------------------------------
# Protocol for vision-language model backends
# ---------------------------------------------------------------------------


@runtime_checkable
class VisionLanguageModel(Protocol):
    """Protocol for vision-language model backends."""

    def generate(
        self,
        prompt: str,
        image_paths: list[str],
        **kwargs: Any,
    ) -> str:
        """
        Generate a response for the given prompt and images.

        Args:
            prompt: Text prompt.
            image_paths: List of paths to images.
            **kwargs: Additional generation parameters.

        Returns:
            Generated text response.
        """
        ...


# ---------------------------------------------------------------------------
# Model configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Configuration for a registered model."""

    family: str
    backend: str  # "vllm", "openai", "gemini"
    model: str  # Model identifier for the API
    api_base: str | None = None
    max_tokens: int = 1024
    model_path: str | None = None  # HuggingFace model path for vLLM
    hf_overrides: dict[str, Any] | None = None  # HuggingFace config overrides
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        """Create ModelConfig from dictionary."""
        return cls(
            family=data["family"],
            backend=data["backend"],
            model=data["model"],
            api_base=data.get("api_base"),
            max_tokens=data.get("max_tokens", 1024),
            model_path=data.get("model_path"),
            hf_overrides=data.get("hf_overrides"),
            extra={
                k: v
                for k, v in data.items()
                if k
                not in {
                    "family",
                    "backend",
                    "model",
                    "api_base",
                    "max_tokens",
                    "model_path",
                    "hf_overrides",
                }
            },
        )


# ---------------------------------------------------------------------------
# Unified model registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    # =========================================================================
    # Qwen3-VL models (vLLM)
    # =========================================================================
    "qwen3-vl-32b-thinking-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-32b-thinking-fp8",
        "model_path": "Qwen/Qwen3-VL-32B-Thinking-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-30b-a3b-thinking-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-30b-a3b-thinking-fp8",
        "model_path": "Qwen/Qwen3-VL-30B-A3B-Thinking-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-30b-a3b-instruct-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-30b-a3b-instruct-fp8",
        "model_path": "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-8b-thinking-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-8b-thinking-fp8",
        "model_path": "Qwen/Qwen3-VL-8B-Thinking-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-8b-instruct-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-8b-instruct-fp8",
        "model_path": "Qwen/Qwen3-VL-8B-Instruct-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-8b-instruct": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-8b-instruct",
        "model_path": "Qwen/Qwen3-VL-8B-Instruct",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-4b-thinking-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-4b-thinking-fp8",
        "model_path": "Qwen/Qwen3-VL-4B-Thinking-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    "qwen3-vl-4b-instruct-fp8": {
        "family": "qwen3-vl",
        "backend": "vllm",
        "model": "qwen3-vl-4b-instruct-fp8",
        "model_path": "Qwen/Qwen3-VL-4B-Instruct-FP8",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    # =========================================================================
    # Gemma-4 models (vLLM, vision-language)
    # =========================================================================
    "gemma-4-e4b-it": {
        "family": "gemma-4",
        "backend": "vllm",
        "model": "google/gemma-4-E4B-it",
        "model_path": "google/gemma-4-E4B-it",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 512,
    },
    # =========================================================================
    # OpenAI models
    # =========================================================================
    "gpt-5": {
        "family": "openai",
        "backend": "openai",
        "model": "gpt-5",
        "max_tokens": 1024,
    },
    "gpt-5-nano": {
        "family": "openai",
        "backend": "openai",
        "model": "gpt-5-nano",
        "max_tokens": 1024,
    },
    # =========================================================================
    # Google Gemini models
    # =========================================================================
    "gemini-3-pro-image-preview": {
        "family": "gemini",
        "backend": "gemini",
        "model": "gemini-3-pro-image-preview",
        "max_tokens": 64,
    },
    "gemini-3-pro": {
        "family": "gemini",
        "backend": "gemini",
        "model": "gemini-3-pro-preview",
        "max_tokens": 128,
    },
    "gemini-2.5-pro": {
        "family": "gemini",
        "backend": "gemini",
        "model": "gemini-2.5-pro",
        "max_tokens": 128,
    },
    "gemini-2.5-flash-lite": {
        "family": "gemini",
        "backend": "gemini",
        "model": "gemini-2.5-flash-lite",
        "max_tokens": 128,
    },
    "gemini-2.5-flash-lite-preview-06-17": {
        "family": "gemini",
        "backend": "gemini",
        "model": "gemini-2.5-flash-lite-preview-06-17",
        "max_tokens": 128,
    },
    # =========================================================================
    # Ministral models (vLLM)
    # =========================================================================
    "ministral-3-14b-reasoning-2512": {
        "family": "ministral",
        "backend": "vllm",
        "model": "ministral-3-14b-reasoning-2512",
        "model_path": "mistralai/Ministral-3-14B-Reasoning-2512",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "ministral-3-14b-instruct-2512": {
        "family": "ministral",
        "backend": "vllm",
        "model": "ministral-3-14b-instruct-2512",
        "model_path": "mistralai/Ministral-3-14B-Instruct-2512",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "ministral-3-8b-reasoning-2512": {
        "family": "ministral",
        "backend": "vllm",
        "model": "ministral-3-8b-reasoning-2512",
        "model_path": "mistralai/Ministral-3-8B-Reasoning-2512",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "ministral-3-8b-instruct-2512": {
        "family": "ministral",
        "backend": "vllm",
        "model": "ministral-3-8b-instruct-2512",
        "model_path": "mistralai/Ministral-3-8B-Instruct-2512",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "ministral-3-3b-reasoning-2512": {
        "family": "ministral",
        "backend": "vllm",
        "model": "ministral-3-3b-reasoning-2512",
        "model_path": "mistralai/Ministral-3-3B-Reasoning-2512",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "ministral-3-3b-instruct-2512": {
        "family": "ministral",
        "backend": "vllm",
        "model": "ministral-3-3b-instruct-2512",
        "model_path": "mistralai/Ministral-3-3B-Instruct-2512",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    # =========================================================================
    # DeepSeek-VL2 models (vLLM)
    # =========================================================================
    "deepseek-vl2": {
        "family": "deepseek-vl2",
        "backend": "vllm",
        "model": "deepseek-vl2",
        "model_path": "deepseek-ai/deepseek-vl2",
        "hf_overrides": {
            "trust_remote_code": True,
            "architectures": ["DeepseekVLV2ForCausalLM"],
            "language_config": {"vocab_size": 102400},
        },
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "deepseek-vl2-small": {
        "family": "deepseek-vl2",
        "backend": "vllm",
        "model": "deepseek-vl2-small",
        "model_path": "deepseek-ai/deepseek-vl2-small",
        "hf_overrides": {
            "trust_remote_code": True,
            "architectures": ["DeepseekVLV2ForCausalLM"],
            "language_config": {"vocab_size": 102400},
        },
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "deepseek-vl2-tiny": {
        "family": "deepseek-vl2",
        "backend": "vllm",
        "model": "deepseek-vl2-tiny",
        "model_path": "deepseek-ai/deepseek-vl2-tiny",
        "hf_overrides": {
            "trust_remote_code": True,
            "architectures": ["DeepseekVLV2ForCausalLM"],
            "language_config": {"vocab_size": 102400},
        },
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    # =========================================================================
    # Gemma-3 models (vLLM)
    # =========================================================================
    "gemma-3-27b-it": {
        "family": "gemma-3",
        "backend": "vllm",
        "model": "gemma-3-27b-it",
        "model_path": "google/gemma-3-27b-it",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "gemma-3-27b-pt": {
        "family": "gemma-3",
        "backend": "vllm",
        "model": "gemma-3-27b-pt",
        "model_path": "google/gemma-3-27b-pt",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "gemma-3-12b-it": {
        "family": "gemma-3",
        "backend": "vllm",
        "model": "gemma-3-12b-it",
        "model_path": "google/gemma-3-12b-it",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "gemma-3-12b-pt": {
        "family": "gemma-3",
        "backend": "vllm",
        "model": "gemma-3-12b-pt",
        "model_path": "google/gemma-3-12b-pt",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "gemma-3-4b-it": {
        "family": "gemma-3",
        "backend": "vllm",
        "model": "gemma-3-4b-it",
        "model_path": "google/gemma-3-4b-it",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
    "gemma-3-4b-pt": {
        "family": "gemma-3",
        "backend": "vllm",
        "model": "gemma-3-4b-pt",
        "model_path": "google/gemma-3-4b-pt",
        "api_base": DEFAULT_API_BASE,
        "max_tokens": 1024,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_model_config(model_name: str) -> dict[str, Any]:
    """
    Get a model configuration by name.

    Args:
        model_name: Registered model name.

    Returns:
        Shallow copy of the model configuration dictionary.

    Raises:
        KeyError: If model_name is not registered.

    Example:
        >>> cfg = get_model_config("qwen3-vl-8b-instruct-fp8")
        >>> print(cfg["backend"])
        'vllm'
    """
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(
            f"Unknown model_name: '{model_name}'. Available models: {available}"
        )
    return dict(MODEL_REGISTRY[model_name])


def _apply_port(base_url: str, port: int, force: bool = False) -> str:
    """
    Ensure the base URL includes an explicit port.

    Args:
        base_url: Original URL.
        port: Port to apply.
        force: Whether to replace existing port.

    Returns:
        URL with port applied.
    """
    if not base_url:
        return base_url
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        return base_url
    if parsed.port and not force:
        return base_url
    netloc = f"{parsed.hostname}:{port}"
    if parsed.username:
        auth = parsed.username
        if parsed.password:
            auth = f"{auth}:{parsed.password}"
        netloc = f"{auth}@{netloc}"
    return urlunparse(
        (
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        )
    )


def _validate_local_vllm_base_url(base_url: str) -> str:
    """Reject remote vLLM API bases in the public reviewer artifact."""

    parsed = urlparse(base_url)
    if parsed.scheme and parsed.hostname not in LOCAL_VLLM_HOSTS:
        raise ValueError(
            "VLLM_API_BASE must point to local vLLM serving "
            "(localhost, 127.0.0.1, or ::1)."
        )
    return base_url


def resolve_base_url(
    base_url: str | None,
    model_cfg: dict[str, Any],
    port: int = DEFAULT_API_PORT,
    force_port: bool = False,
) -> str:
    """
    Resolve API base URL, injecting vLLM port when applicable.

    Args:
        base_url: Explicit base URL (overrides model config).
        model_cfg: Model configuration dictionary.
        port: Port number for vLLM backends.
        force_port: Whether to force port replacement.

    Returns:
        Resolved API base URL.

    Example:
        >>> cfg = get_model_config("qwen3-vl-8b-instruct-fp8")
        >>> url = resolve_base_url(None, cfg, port=8080)
    """
    resolved = base_url or model_cfg.get("api_base") or DEFAULT_API_BASE
    if model_cfg.get("backend") == "vllm":
        return _validate_local_vllm_base_url(
            _apply_port(resolved, port, force=force_port)
        )
    return resolved


def list_models_by_backend(backend: str) -> list[str]:
    """
    List all registered models for a given backend.

    Args:
        backend: Backend type ("vllm", "openai", "gemini").

    Returns:
        List of model names.

    Example:
        >>> vllm_models = list_models_by_backend("vllm")
    """
    return [
        name for name, cfg in MODEL_REGISTRY.items() if cfg.get("backend") == backend
    ]


def list_models_by_family(family: str) -> list[str]:
    """
    List all registered models for a given family.

    Args:
        family: Model family ("qwen3-vl", "openai", "gemini", etc.).

    Returns:
        List of model names.

    Example:
        >>> qwen_models = list_models_by_family("qwen3-vl")
    """
    return [name for name, cfg in MODEL_REGISTRY.items() if cfg.get("family") == family]


# ---------------------------------------------------------------------------
# Build functions (for backward compatibility with inference_runner)
# ---------------------------------------------------------------------------


def build_model_config(model_name: str, prefer_local: bool = False) -> dict[str, Any]:
    """
    Build a model configuration dictionary.

    Args:
        model_name: Registered model name.
        prefer_local: Whether to prefer local vLLM mode (affects api_base).

    Returns:
        Model configuration dictionary.

    Raises:
        KeyError: If model is not registered.

    Example:
        >>> cfg = build_model_config("qwen3-vl-8b-instruct-fp8")
        >>> print(cfg["backend"])
        'vllm'
    """
    cfg = get_model_config(model_name)
    result = dict(cfg)
    result["name"] = model_name

    # Apply local mode adjustments
    if prefer_local and cfg.get("backend") == "vllm":
        result["mode"] = "local"
    else:
        result["mode"] = "server"

    return result


def build_model_from_config(config: dict[str, Any], **kwargs: Any) -> VisionLanguageModel:
    """
    Create a model instance from a configuration dictionary.

    Args:
        config: Model configuration dictionary (from build_model_config).
        **kwargs: Additional backend configuration.

    Returns:
        VisionLanguageModel instance.

    Example:
        >>> cfg = build_model_config("gpt-4o")
        >>> model = build_model_from_config(cfg)
        >>> result = model.generate("image.jpg", "Describe this.")
    """
    from .backends import create_backend

    model_name = config.get("name", config.get("model", "unknown"))
    mode = config.get("mode", "server")

    # Merge config with overrides
    merged_kwargs = {k: v for k, v in config.items() if k not in {"name", "mode"}}
    merged_kwargs.update(kwargs)

    return create_backend(model_name, mode=mode, **merged_kwargs)


def build_model(
    model_name: str,
    mode: str = "server",
    **kwargs: Any,
) -> VisionLanguageModel:
    """
    Build a vision-language model instance directly.

    This is a convenience function that combines build_model_config
    and build_model_from_config.

    Args:
        model_name: Registered model name.
        mode: Backend mode ("server" or "local").
        **kwargs: Additional backend configuration.

    Returns:
        VisionLanguageModel instance.

    Example:
        >>> model = build_model("qwen3-vl-8b-instruct-fp8")
        >>> result = model.generate("image.jpg", "Describe this image.")
    """
    from .backends import create_backend

    return create_backend(model_name, mode=mode, **kwargs)
