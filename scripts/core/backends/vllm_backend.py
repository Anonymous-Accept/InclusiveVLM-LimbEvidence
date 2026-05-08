"""vLLM backend for vision-language models (server and local modes)."""

from __future__ import annotations

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from PIL import Image

from .base import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    STOP_STRINGS,
    BaseBackend,
    load_and_resize_image,
)

__all__ = ["VLLMBackend"]

LOGGER = logging.getLogger(__name__)
LOCAL_VLLM_HOSTS = {"localhost", "127.0.0.1", "::1"}


class VLLMBackend(BaseBackend):
    """
    vLLM backend supporting both server (OpenAI-compatible API) and local modes.

    Server mode uses the OpenAI Python client to communicate with a vLLM server.
    Local mode loads the model directly using vLLM's LLM class.

    Attributes:
        mode: Either "server" (default) or "local".
        api_base: Base URL for the vLLM server (server mode).
        model_path: HuggingFace model path (local mode).
        max_image_size: Maximum image dimension for resizing.

    Example:
        >>> backend = VLLMBackend(
        ...     model_name="qwen3-vl-8b-instruct-fp8",
        ...     model="qwen3-vl-8b-instruct-fp8",
        ...     api_base="http://localhost:8000/v1",
        ...     mode="server",
        ... )
        >>> result = backend.generate("image.jpg", "Describe this image.")
    """

    def __init__(
        self,
        model_name: str,
        model: str,
        api_base: str,
        mode: str = "server",
        model_path: str | None = None,
        family: str = "qwen3-vl",
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        top_p: float = DEFAULT_TOP_P,
        max_image_size: int = 1280,
        tensor_parallel_size: int = 1,
        max_model_len: int = 32768,
        hf_overrides: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the vLLM backend.

        Args:
            model_name: Human-readable model name.
            model: Model identifier for API calls.
            api_base: Base URL for vLLM server.
            mode: "server" for API or "local" for direct loading.
            model_path: HuggingFace model path for local mode.
            family: Model family (qwen3-vl, deepseek-vl2, gemma-3, etc.).
            max_tokens: Default maximum tokens.
            temperature: Generation temperature.
            top_p: Nucleus sampling probability.
            max_image_size: Maximum image dimension.
            tensor_parallel_size: GPU tensor parallelism (local mode).
            max_model_len: Maximum context length (local mode).
            hf_overrides: HuggingFace config overrides.
            **kwargs: Additional configuration.
        """
        super().__init__(
            model_name=model_name,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        self.mode = mode
        self.api_base = api_base
        self.model_path = model_path or model
        self.family = family
        self.max_image_size = max_image_size
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.hf_overrides = hf_overrides or {}

        # Initialize mode-specific components
        self.client = None
        self.llm = None
        self.processor = None
        self.sampling_params = None

        if mode == "server":
            self._init_server_mode()
        else:
            self._init_local_mode()

    def _init_server_mode(self) -> None:
        """Initialize OpenAI client for server mode."""
        from openai import OpenAI

        parsed = urlparse(self.api_base)
        if parsed.scheme and parsed.hostname not in LOCAL_VLLM_HOSTS:
            raise ValueError(
                "VLLM_API_BASE must point to local vLLM serving "
                "(localhost, 127.0.0.1, or ::1)."
            )
        self.client = OpenAI(
            base_url=self.api_base,
            api_key=os.environ.get("VLLM_API_KEY", "EMPTY"),
        )
        LOGGER.info(
            "VLLMBackend [server] initialized: model=%s api_base=%s",
            self.model,
            self.api_base,
        )

    def _init_local_mode(self) -> None:
        """Initialize local vLLM LLM instance."""
        from transformers import AutoProcessor
        from vllm import LLM, SamplingParams

        LOGGER.info(
            "VLLMBackend [local] initializing: model_path=%s",
            self.model_path,
        )

        # Handle DeepSeek VL2 special requirements
        trust_remote_code = self.hf_overrides.get("trust_remote_code", False)
        llm_overrides = {
            k: v for k, v in self.hf_overrides.items() if k != "trust_remote_code"
        }

        if self.family == "deepseek-vl2":
            trust_remote_code = True
            # Add external DeepSeek-VL2 path to Python path
            deepseek_path = Path("external/DeepSeek-VL2")
            if deepseek_path.exists():
                import sys

                sys.path.insert(0, str(deepseek_path))
                os.environ["PYTHONPATH"] = (
                    f"{deepseek_path}:{os.environ.get('PYTHONPATH', '')}"
                )

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=trust_remote_code,
        )

        # Initialize LLM
        self.llm = LLM(
            model=self.model_path,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            hf_overrides=llm_overrides if llm_overrides else None,
        )

        self.sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        LOGGER.info("VLLMBackend [local] initialized successfully")

    def _encode_image(self, image_path: str) -> str:
        """Encode image as base64 data URL for server mode."""
        image = load_and_resize_image(image_path, max_size=self.max_image_size)
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"

    def _prepare_prompt(
        self, image_path: str, prompt: str
    ) -> tuple[str, list[Image.Image]]:
        """Construct chat template prompt for local mode."""
        image = load_and_resize_image(image_path, max_size=self.max_image_size)

        # DeepSeek VL2 uses different message format
        if self.family == "deepseek-vl2":
            messages = [
                {"role": "<|User|>", "content": f"<image>\n{prompt}"},
                {"role": "<|Assistant|>", "content": ""},
            ]
            prompt_text = self.processor.format_messages(
                conversations=messages,
                sft_format="deepseek",
                system_prompt="",
            )
            return prompt_text, [image]

        # Standard format (Qwen, Gemma, etc.)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        prompt_text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt_text, [image]

    def _generate_server(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        assistant_prefill: str | None = None,
        guided_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate via OpenAI-compatible server."""
        img_b64 = self._encode_image(image_path)

        extra_body: dict[str, Any] = {}
        response_format = None

        # Handle guided decoding
        if guided_info:
            g_type = guided_info.get("type")
            if g_type == "choice":
                extra_body["guided_choice"] = guided_info.get("choices", [])
            elif g_type == "json":
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "json_response",
                        "schema": guided_info.get("schema"),
                    },
                }
            elif g_type == "regex":
                extra_body["guided_regex"] = guided_info.get("regex")
            elif g_type == "grammar":
                extra_body["guided_grammar"] = guided_info.get("grammar")

        # Handle assistant prefill
        if assistant_prefill:
            extra_body["assistant_prefill"] = [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": assistant_prefill}],
                }
            ]

        content_items = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": img_b64}},
        ]
        if self.family == "deepseek-vl2":
            # DeepSeek-VL2 expects the image token before the text prompt.
            content_items = [
                {"type": "image_url", "image_url": {"url": img_b64}},
                {"type": "text", "text": prompt},
            ]

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=max_tokens or self.max_tokens,
            stop=STOP_STRINGS,
            extra_body=extra_body if extra_body else None,
            response_format=response_format,
            messages=[
                {
                    "role": "user",
                    "content": content_items,
                }
            ],
        )

        raw = completion.choices[0].message.content if completion.choices else ""
        usage_dict: dict[str, Any] = {}
        if completion.usage:
            if hasattr(completion.usage, "model_dump"):
                usage_dict = completion.usage.model_dump()
            elif hasattr(completion.usage, "to_dict"):
                usage_dict = completion.usage.to_dict()
            else:
                usage_dict = dict(completion.usage)

        return {
            "raw_text": raw or "",
            "parsed_answer": None,
            "usage": usage_dict,
            "success": bool(raw),
        }

    def _generate_local(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        guided_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Generate via local vLLM instance."""
        from vllm import SamplingParams

        guided_kwargs = self._build_guided_decoding_params(guided_info)

        sampling_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": STOP_STRINGS,
        }
        if guided_kwargs:
            sampling_kwargs.update(guided_kwargs)

        sampling_params = SamplingParams(**sampling_kwargs)
        prompt_text, images = self._prepare_prompt(image_path, prompt)

        request = {
            "prompt": prompt_text,
            "multi_modal_data": {"image": images},
        }

        outputs = self.llm.generate(
            request,
            sampling_params=sampling_params,
            use_tqdm=False,
        )

        raw_text = ""
        generated_tokens = None
        prompt_tokens = None

        if outputs:
            first = outputs[0]
            if first.outputs:
                raw_text = first.outputs[0].text
                generated_tokens = len(first.outputs[0].token_ids)
            prompt_tokens = len(first.prompt_token_ids)

        return {
            "raw_text": raw_text,
            "parsed_answer": None,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
            },
            "success": bool(raw_text),
        }

    @staticmethod
    def _build_guided_decoding_params(
        guided_info: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Build guided decoding parameters for vLLM."""
        if not guided_info:
            return None

        # Try new API (vLLM >= 0.12.0)
        try:
            from vllm.sampling_params import StructuredOutputsParams

            if guided_info.get("type") == "choice":
                return {
                    "structured_outputs": StructuredOutputsParams(
                        choice=guided_info.get("choices", [])
                    )
                }
            if guided_info.get("type") == "json":
                return {
                    "structured_outputs": StructuredOutputsParams(
                        json=guided_info.get("schema")
                    )
                }
        except ImportError:
            pass

        # Fallback to old API
        try:
            from vllm.sampling_params import GuidedDecodingParams

            if guided_info.get("type") == "choice":
                return {
                    "guided_decoding": GuidedDecodingParams(
                        choice=guided_info.get("choices", [])
                    )
                }
            if guided_info.get("type") == "json":
                return {
                    "guided_decoding": GuidedDecodingParams(
                        json=guided_info.get("schema")
                    )
                }
        except ImportError:
            pass

        return None

    def generate(
        self,
        image_path: str,
        prompt: str,
        max_tokens: int | None = None,
        assistant_prefill: str | None = None,
        guided_info: dict[str, Any] | None = None,
        task_name: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Generate a response using the configured vLLM mode.

        Args:
            image_path: Path to input image.
            prompt: Text prompt.
            max_tokens: Override default max tokens.
            assistant_prefill: Optional assistant prefill text (server mode only).
            guided_info: Guided decoding configuration.
            task_name: Optional task identifier.
            **kwargs: Additional parameters.

        Returns:
            Dictionary with raw_text, parsed_answer, usage, and success.
        """
        if self.mode == "local":
            result = self._generate_local(
                image_path, prompt, max_tokens, guided_info=guided_info
            )
        else:
            result = self._generate_server(
                image_path,
                prompt,
                max_tokens,
                assistant_prefill=assistant_prefill,
                guided_info=guided_info,
            )

        if task_name:
            result["task_name"] = task_name
        return result

    def generate_batch(
        self,
        requests: list[dict[str, Any]],
        max_tokens: int | None = None,
        assistant_prefill: str | None = None,
        guided_info: dict[str, Any] | None = None,
        max_concurrent: int = 8,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Generate responses for a batch of requests.

        批量推理接口 | Batch Inference Interface

        This method processes multiple requests efficiently:
        - Server mode: Uses async HTTP requests with concurrency control
        - Local mode: Uses vLLM's native batch processing

        Args:
            requests: List of request dicts, each containing:
                - image_path: Path to input image
                - prompt: Text prompt
                - item_id (optional): Identifier for tracking
            max_tokens: Override default max tokens for all requests.
            assistant_prefill: Optional assistant prefill text (server mode).
            guided_info: Guided decoding configuration.
            max_concurrent: Maximum concurrent requests (server mode).
            **kwargs: Additional parameters.

        Returns:
            List of result dicts in same order as requests, each containing:
                - raw_text: Generated text
                - parsed_answer: Parsed result (if applicable)
                - usage: Token usage statistics
                - success: Whether generation succeeded
                - item_id: Original item_id (if provided)
                - error: Error message (if failed)

        Example:
            >>> requests = [
            ...     {"image_path": "img1.jpg", "prompt": "Describe", "item_id": "001"},
            ...     {"image_path": "img2.jpg", "prompt": "Describe", "item_id": "002"},
            ... ]
            >>> results = backend.generate_batch(requests, max_concurrent=4)
        """
        if not requests:
            return []

        if self.mode == "local":
            return self._generate_batch_local(
                requests, max_tokens, guided_info=guided_info
            )
        else:
            return self._generate_batch_server(
                requests,
                max_tokens,
                assistant_prefill=assistant_prefill,
                guided_info=guided_info,
                max_concurrent=max_concurrent,
            )

    def _generate_batch_local(
        self,
        requests: list[dict[str, Any]],
        max_tokens: int | None = None,
        guided_info: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Batch generate via local vLLM instance (native batching)."""
        from vllm import SamplingParams

        guided_kwargs = self._build_guided_decoding_params(guided_info)

        sampling_kwargs = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_tokens or self.max_tokens,
            "stop": STOP_STRINGS,
        }
        if guided_kwargs:
            sampling_kwargs.update(guided_kwargs)

        sampling_params = SamplingParams(**sampling_kwargs)

        # Prepare all prompts and images
        vllm_requests = []
        for req in requests:
            image_path = req["image_path"]
            prompt = req["prompt"]
            prompt_text, images = self._prepare_prompt(image_path, prompt)
            vllm_requests.append({
                "prompt": prompt_text,
                "multi_modal_data": {"image": images},
            })

        # Batch generate
        outputs = self.llm.generate(
            vllm_requests,
            sampling_params=sampling_params,
            use_tqdm=True,
        )

        # Process results
        results = []
        for i, output in enumerate(outputs):
            item_id = requests[i].get("item_id")
            raw_text = ""
            generated_tokens = None
            prompt_tokens = None

            if output.outputs:
                raw_text = output.outputs[0].text
                generated_tokens = len(output.outputs[0].token_ids)
            prompt_tokens = len(output.prompt_token_ids)

            result = {
                "raw_text": raw_text,
                "parsed_answer": None,
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                },
                "success": bool(raw_text),
            }
            if item_id:
                result["item_id"] = item_id
            results.append(result)

        return results

    def _generate_batch_server(
        self,
        requests: list[dict[str, Any]],
        max_tokens: int | None = None,
        assistant_prefill: str | None = None,
        guided_info: dict[str, Any] | None = None,
        max_concurrent: int = 8,
    ) -> list[dict[str, Any]]:
        """Batch generate via OpenAI-compatible server with async concurrency."""
        from concurrent.futures import ThreadPoolExecutor

        results = [None] * len(requests)

        def process_single(idx: int, req: dict[str, Any]) -> tuple[int, dict[str, Any]]:
            """Process a single request and return (index, result)."""
            try:
                result = self._generate_server(
                    image_path=req["image_path"],
                    prompt=req["prompt"],
                    max_tokens=max_tokens,
                    assistant_prefill=assistant_prefill,
                    guided_info=guided_info,
                )
                if req.get("item_id"):
                    result["item_id"] = req["item_id"]
                return idx, result
            except Exception as e:
                LOGGER.error(f"Batch request {idx} failed: {e}")
                return idx, {
                    "raw_text": "",
                    "parsed_answer": None,
                    "usage": {},
                    "success": False,
                    "error": str(e),
                    "item_id": req.get("item_id"),
                }

        # Use ThreadPoolExecutor for concurrent HTTP requests
        with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            futures = [
                executor.submit(process_single, i, req)
                for i, req in enumerate(requests)
            ]
            for future in futures:
                idx, result = future.result()
                results[idx] = result

        return results
