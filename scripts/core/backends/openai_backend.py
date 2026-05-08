"""OpenAI backend for GPT vision models."""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    BackendError,
    BaseBackend,
    encode_image_base64,
)

__all__ = ["OpenAIBackend"]

LOGGER = logging.getLogger(__name__)


class OpenAIBackend(BaseBackend):
    """
    Backend for OpenAI GPT vision models.

    Uses the OpenAI Python client to communicate with the OpenAI API.

    Attributes:
        api_key: OpenAI API key.
        client: OpenAI client instance.

    Example:
        >>> backend = OpenAIBackend(
        ...     model_name="gpt-4o-mini",
        ...     model="gpt-4o-mini",
        ... )
        >>> result = backend.generate("image.jpg", "Describe this image.")
    """

    def __init__(
        self,
        model_name: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the OpenAI backend.

        Args:
            model_name: Human-readable model name.
            model: Model identifier (e.g., "gpt-4o-mini").
            max_tokens: Default maximum tokens.
            temperature: Generation temperature.
            **kwargs: Additional configuration.
        """
        super().__init__(
            model_name=model_name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        from openai import OpenAI

        resolved_key = os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise BackendError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable."
            )

        self.client = OpenAI(api_key=resolved_key)
        LOGGER.info("OpenAIBackend initialized: model=%s", self.model)

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a response using OpenAI vision chat completion.

        Args:
            image_path: Path to input image.
            prompt: Text prompt.
            max_tokens: Override default max tokens.
            task_name: Optional task identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with raw_text, parsed_answer, usage, success, and task_name.
        """
        img_b64 = encode_image_base64(image_path)

        params: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": img_b64}},
                    ],
                }
            ],
        }

        # Some newer models expect max_completion_tokens instead of max_tokens
        max_tok = max_tokens or self.max_tokens
        params["max_completion_tokens"] = max_tok

        try:
            completion = self.client.chat.completions.create(**params)
            raw = completion.choices[0].message.content if completion.choices else ""

            usage_dict: dict[str, Any] = {}
            if completion.usage:
                if hasattr(completion.usage, "model_dump"):
                    usage_dict = completion.usage.model_dump()
                elif hasattr(completion.usage, "to_dict"):
                    usage_dict = completion.usage.to_dict()
                else:
                    usage_dict = dict(completion.usage)

            result = {
                "raw_text": raw or "",
                "parsed_answer": None,
                "usage": usage_dict,
                "success": bool(raw),
            }

        except Exception as e:
            LOGGER.error("OpenAI generation failed: %s", e)
            result = {
                "raw_text": "",
                "parsed_answer": None,
                "usage": {},
                "success": False,
                "error": str(e),
            }

        if task_name:
            result["task_name"] = task_name

        return result
