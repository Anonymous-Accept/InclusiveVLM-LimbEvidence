"""Base classes and utilities for vision-language model backends."""

from __future__ import annotations

import base64
import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

__all__ = [
    "BaseBackend",
    "BackendError",
    "encode_image_base64",
    "load_and_resize_image",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "DEFAULT_MAX_TOKENS",
    "STOP_STRINGS",
]

LOGGER = logging.getLogger(__name__)

# Default generation parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0
DEFAULT_MAX_TOKENS = 1024
STOP_STRINGS = ["<|im_end|>", "<|endoftext|>", "</s>", "<|eot_id|>"]


class BackendError(Exception):
    """Exception raised by backend operations."""

    pass


def load_and_resize_image(
    image_path: str | Path,
    max_size: int | None = None,
    convert_rgb: bool = True,
) -> Image.Image:
    """
    Load an image and optionally resize it.

    Args:
        image_path: Path to the image file.
        max_size: Maximum size for the longest side. None disables resizing.
        convert_rgb: Whether to convert to RGB mode.

    Returns:
        PIL Image object.

    Raises:
        FileNotFoundError: If the image file does not exist.
        BackendError: If the image cannot be loaded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    try:
        image = Image.open(path)
        if convert_rgb:
            image = image.convert("RGB")

        if max_size and max_size > 0:
            w, h = image.size
            longest = max(w, h)
            if longest > max_size:
                scale = max_size / float(longest)
                new_size = (int(w * scale), int(h * scale))
                image = image.resize(new_size, Image.LANCZOS)

        return image
    except Exception as e:
        raise BackendError(f"Failed to load image {path}: {e}") from e


def encode_image_base64(
    image: str | Path | Image.Image,
    format: str = "JPEG",
    max_size: int | None = None,
) -> str:
    """
    Encode an image as a base64 data URL.

    Args:
        image: Image path or PIL Image object.
        format: Output format (JPEG, PNG, etc.).
        max_size: Maximum size for resizing.

    Returns:
        Base64-encoded data URL string.

    Example:
        >>> url = encode_image_base64("image.jpg")
        >>> # Returns: "data:image/jpeg;base64,/9j/4AAQ..."
    """
    if isinstance(image, (str, Path)):
        image = load_and_resize_image(image, max_size=max_size)

    buffer = io.BytesIO()
    image.save(buffer, format=format)
    b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{b64}"


class BaseBackend(ABC):
    """
    Abstract base class for vision-language model backends.

    All backends must implement the `generate` method.

    Attributes:
        model_name: Human-readable model name.
        model: Model identifier for API calls.
        max_tokens: Default maximum tokens for generation.
        temperature: Generation temperature.
        top_p: Nucleus sampling probability.
    """

    def __init__(
        self,
        model_name: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the backend.

        Args:
            model_name: Human-readable model name.
            model: Model identifier for API calls.
            max_tokens: Default maximum tokens.
            temperature: Generation temperature.
            top_p: Nucleus sampling probability.
            **kwargs: Additional configuration (ignored by base class).
        """
        self.model_name = model_name
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a response for the given image and prompt.

        Args:
            image_path: Path to the input image.
            prompt: Text prompt.
            max_tokens: Override default max tokens.
            **kwargs: Additional generation parameters.

        Returns:
            Dictionary containing:
                - raw_text: Generated text response.
                - parsed_answer: Optional parsed structured answer.
                - usage: Token usage statistics.
                - success: Whether generation succeeded.
        """
        ...

    def batch_generate(
        self,
        requests: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for multiple requests.

        Default implementation processes sequentially.
        Override for parallel/batched processing.

        Args:
            requests: List of dicts with 'image_path' and 'prompt' keys.
            **kwargs: Additional generation parameters.

        Returns:
            List of generation results.
        """
        results = []
        for req in requests:
            result = self.generate(
                image_path=req["image_path"],
                prompt=req["prompt"],
                **kwargs,
            )
            results.append(result)
        return results
