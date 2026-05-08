"""Google Gemini backend for multimodal models."""

from __future__ import annotations

import logging
import os
import time
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from .base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    BackendError,
    BaseBackend,
)

__all__ = ["GeminiBackend"]

LOGGER = logging.getLogger(__name__)


class GeminiBackend(BaseBackend):
    """
    Backend for Google Gemini multimodal models.

    Uses the Google GenAI Python SDK to communicate with the Gemini API.

    Attributes:
        api_key: Gemini API key loaded from GEMINI_API_KEY.
        client: Google GenAI client instance.

    Example:
        >>> backend = GeminiBackend(
        ...     model_name="gemini-1.5-flash",
        ...     model="gemini-1.5-flash",
        ... )
        >>> result = backend.generate("image.jpg", "Describe this image.")
    """

    def __init__(
        self,
        model_name: str,
        model: str,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        max_retries: int = 5,
        base_delay: float = 2.0,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Gemini backend.

        Args:
            model_name: Human-readable model name.
            model: Model identifier (e.g., "gemini-1.5-flash").
            max_tokens: Default maximum output tokens.
            temperature: Generation temperature.
            max_retries: Maximum retry attempts for rate limiting.
            base_delay: Base delay for exponential backoff.
            **kwargs: Additional configuration.
        """
        super().__init__(
            model_name=model_name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        from google import genai

        resolved_key = os.environ.get("GEMINI_API_KEY")
        if not resolved_key:
            raise BackendError(
                "Gemini API key is required. Set GEMINI_API_KEY."
            )

        self.api_key = resolved_key
        self.max_retries = max_retries
        self.base_delay = base_delay

        self.client = genai.Client(api_key=resolved_key)
        LOGGER.info("GeminiBackend initialized: model=%s", self.model)

    def _mask_key(self, value: str | None) -> str:
        """Mask API key for logging."""
        if not value:
            return ""
        if len(value) <= 8:
            return value[:2] + "***" + value[-2:]
        return value[:4] + "***" + value[-4:]

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a response using Gemini vision-language API.

        Args:
            image_path: Path to input image.
            prompt: Text prompt.
            max_tokens: Override default max output tokens.
            task_name: Optional task identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with raw_text, parsed_answer, usage, finish_reason, success.
        """
        from google.api_core import exceptions
        from google.genai.types import Part

        # Load and prepare image
        img = Image.open(Path(image_path))
        buf = BytesIO()
        fmt = img.format or "PNG"
        img.save(buf, format=fmt)
        img_bytes = buf.getvalue()
        mime_type = Image.MIME.get(fmt, "image/png")
        image_part = Part.from_bytes(data=img_bytes, mime_type=mime_type)

        effective_max_tokens = max_tokens or self.max_tokens

        LOGGER.debug(
            "Gemini generate: model=%s max_tokens=%s image=%s api_key=%s",
            self.model,
            effective_max_tokens,
            image_path,
            self._mask_key(self.api_key),
        )

        response = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=[image_part, prompt],
                    config={
                        "temperature": self.temperature,
                        "max_output_tokens": effective_max_tokens,
                    },
                )
                break
            except exceptions.ResourceExhausted:
                if attempt == self.max_retries - 1:
                    raise
                delay = self.base_delay * (2**attempt)
                LOGGER.warning(
                    "Gemini 429 ResourceExhausted. attempt=%d retry_in=%.1fs model=%s",
                    attempt + 1,
                    delay,
                    self.model_name,
                )
                time.sleep(delay)
            except Exception:
                raise

        # Extract response text
        raw = ""
        finish_reason = None

        if getattr(response, "candidates", None):
            cand = response.candidates[0]
            finish_reason = getattr(cand, "finish_reason", None)
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) if content else None

            if parts:
                texts = []
                for p in parts:
                    if hasattr(p, "text"):
                        texts.append(p.text)
                    else:
                        LOGGER.debug("Gemini non-text part: %s", p)
                raw = "".join(t for t in texts if t)
            elif content and hasattr(content, "text"):
                raw = content.text or ""

        # Fallback: try response.text
        if not raw:
            try:
                raw = getattr(response, "text", None) or ""
            except Exception:
                raw = ""

        if not raw:
            LOGGER.debug("Gemini empty response for image=%s", image_path)
            if hasattr(response, "prompt_feedback"):
                LOGGER.debug("Gemini prompt_feedback: %s", response.prompt_feedback)
            if getattr(response, "candidates", None):
                cand = response.candidates[0]
                LOGGER.debug("Gemini candidate: %s", cand)
                LOGGER.debug(
                    "Gemini finish_reason: %s",
                    getattr(cand, "finish_reason", "Unknown"),
                )
                LOGGER.debug(
                    "Gemini safety_ratings: %s",
                    getattr(cand, "safety_ratings", "Unknown"),
                )

        # Extract usage metadata
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is None:
            usage: dict[str, Any] = {}
        elif hasattr(usage_meta, "to_dict"):
            usage = usage_meta.to_dict()
        else:
            try:
                usage = dict(usage_meta)
            except Exception:
                usage = {"usage_metadata": str(usage_meta)}

        result = {
            "raw_text": raw,
            "parsed_answer": None,
            "usage": usage,
            "finish_reason": finish_reason,
            "success": bool(raw),
        }

        if task_name:
            result["task_name"] = task_name

        return result
