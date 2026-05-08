"""Run VLM inference for Prosthesis Matching.

Output constraint strategy (per docs/guides/llm_output_constraints.md):
- L1: Prompt Engineering - JSON format instruction in prompt
- L2: Assistant Prefill - Pre-fill '{"choices":' for server mode
- L3: Guided Decoding - Choice constraint for A-H options
- L4: Robust Parsing - Regex fallback for non-standard outputs

Usage:
    conda activate inclusivevlm-lep
    python -m scripts.inclusive_vlm_lep.prosthesis_match.run_inference \
        --items-path data/inclusive_vlm_lep/prosthesis_match/items/prosthesis_match.jsonl \
        --model-name qwen2-vl \
        --output-dir outputs/inclusive_vlm_lep/prosthesis_matching/runs/qwen2-vl/compatibility_diversity \
        --assistant-prefill \
        --guided-decoding
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import re
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image

from .config import (
    ALLOWED_CHOICES,
    DEFAULT_DATA_GRIDS_DIR,
    DEFAULT_DATA_ITEMS_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    PROJECT_ROOT,
)

__all__ = ["run_inference", "InferenceConfig", "PredictionRecord"]

logger = logging.getLogger(__name__)


@dataclass
class InferenceConfig:
    """Configuration for inference run."""

    model_name: str
    items_path: Path
    output_dir: Path
    max_samples: Optional[int] = None
    seed: int = DEFAULT_SEED
    use_grid: bool = True
    grids_dir: Path = DEFAULT_DATA_GRIDS_DIR
    max_tokens: int = 1024
    temperature: float = 0.0
    vllm_mode: str = "server"
    vllm_port: int | None = None
    resume: bool = False
    overwrite: bool = False  # New: overwrite existing results
    log_level: str = "INFO"
    # L2: Assistant prefill
    assistant_prefill: bool = False
    # L3: Guided decoding
    guided_decoding: bool = False
    # DeepSeek-specific adjustments
    deepseek_fixes: bool = False
    # Batch processing
    batch_size: int = 1  # Number of items to process in parallel
    max_concurrent: int = 8  # Max concurrent requests (server mode)

    def __post_init__(self) -> None:
        """Normalize relative path fields to absolute paths."""
        self.items_path = _resolve_config_path(self.items_path)
        self.output_dir = _resolve_config_path(self.output_dir)
        self.grids_dir = _resolve_config_path(self.grids_dir)


@dataclass
class PredictionRecord:
    """Single prediction record."""

    item_id: str
    model_id: str
    choices: List[str]  # Changed from single choice to list of choices
    raw_text: str
    parse_error: int
    latency_ms: int
    timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def _resolve_config_path(path: Path) -> Path:
    """Resolve config paths relative to project root when needed.

    Args:
        path: Path from CLI or config.

    Returns:
        Absolute path resolved from project root if input is relative.
    """
    resolved = Path(path).expanduser()
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


def _resolve_item_path(path_value: str) -> Path:
    """Resolve item path entries relative to project root when needed.

    Args:
        path_value: Path string from items JSONL.

    Returns:
        Absolute path resolved from project root if input is relative.
    """
    resolved = Path(path_value).expanduser()
    if resolved.is_absolute():
        return resolved
    return (PROJECT_ROOT / resolved).resolve()


# ---------------------------------------------------------------------------
# L1: Prompt Engineering - System prompt with clear JSON format instruction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a visual analysis assistant. Your task is to analyze images and select all appropriate options based on visual evidence.

Rules:
- Only use visible cues from the images.
- Do not infer health status, identity, or private attributes.
- Select all matching options (zero, one, or multiple).
- If no option matches, return an empty list.
- Output your answer as strict JSON in a single line.
- Do not include any markdown formatting or extra text."""

# L2: Assistant prefill string
ASSISTANT_PREFILL = '{"choices":'
DEEPSEEK_ASSISTANT_PREFILL = '{"choices":['

# L3: JSON schema for guided decoding (choices array from A-H)
CHOICES_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": list(ALLOWED_CHOICES),
            },
            "minItems": 0,
            "maxItems": len(ALLOWED_CHOICES),
            "uniqueItems": True,
        }
    },
    "required": ["choices"],
    "additionalProperties": False,
}
CHOICES_JSON_SCHEMA_COMPAT = {
    "type": "object",
    "properties": {
        "choices": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": list(ALLOWED_CHOICES),
            },
            "minItems": 0,
            "maxItems": len(ALLOWED_CHOICES),
        }
    },
    "required": ["choices"],
    "additionalProperties": False,
}


def _build_choices_schema(
    allowed_ids: List[str],
    *,
    strict_unique: bool,
    max_items: Optional[int] = None,
) -> Dict[str, Any]:
    """Build a JSON schema for the choices array scoped to the given option IDs."""
    effective_max = max_items if max_items is not None else len(allowed_ids)
    schema: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "choices": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": list(allowed_ids),
                },
                "minItems": 0,
                "maxItems": effective_max,
            }
        },
        "required": ["choices"],
        "additionalProperties": False,
    }
    if strict_unique:
        schema["properties"]["choices"]["uniqueItems"] = True
    return schema


def get_guided_decoding_params(
    use_guided: bool,
    *,
    is_deepseek: bool,
    allowed_ids: Optional[List[str]] = None,
    max_items: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Build guided decoding params.

    Args:
        use_guided: Whether to enable guided decoding.
        is_deepseek: Whether the model is DeepSeek-VL2.
        allowed_ids: Option labels to restrict output to (e.g. ["A","B","C","D"]).
            Falls back to the full A-H pool when omitted.

    Returns:
        Guided decoding info dict or None.
    """
    if not use_guided:
        return None

    if allowed_ids is None:
        allowed_ids = list(ALLOWED_CHOICES)

    schema = _build_choices_schema(allowed_ids, strict_unique=True, max_items=max_items)
    schema_compat = _build_choices_schema(allowed_ids, strict_unique=False, max_items=max_items)

    if is_deepseek:
        return {"type": "json", "schema": schema_compat}

    try:
        # Try vLLM >= 0.12.0 API first
        from vllm import StructuredOutputsParams

        return {"structured_outputs": StructuredOutputsParams(json=schema)}
    except ImportError:
        pass

    try:
        # Fallback to vLLM < 0.12.0 API
        from vllm import GuidedDecodingParams

        return {"guided_decoding": GuidedDecodingParams(json=schema)}
    except ImportError:
        logger.warning(
            "Guided decoding requested but vLLM not available. "
            "Install vllm or disable --guided-decoding."
        )
        return None


def build_user_prompt(
    item: Dict[str, Any],
    use_grid: bool = True,
    *,
    is_deepseek: bool = False,
) -> str:
    """Build user prompt for a benchmark item.

    L1: Includes explicit JSON format instruction.

    Args:
        item: Benchmark item dictionary.
        use_grid: Whether to use grid mode.

    Returns:
        User prompt string.
    """
    query_text = item["query"]["text"]
    if is_deepseek:
        query_text = _strip_json_instructions(query_text)

    # Build options description
    options_text = "Options:\n"
    for opt in item["options"]:
        opt_id = opt["option_id"]
        if is_deepseek:
            part8 = opt.get("part8", "")
            label = part8 if part8 else "prosthesis"
            options_text += f"  {opt_id}: {label}\n"
        else:
            options_text += f"  {opt_id}: [Image of prosthesis]\n"

    # Category variant = single-pick hit@1; do not invite multi-select
    is_category = (item.get("answer") or {}).get("mode") == "category"

    # L1: Explicit JSON format instruction for multiple choices
    if is_deepseek:
        if is_category:
            prompt = (
                f"{query_text}\n"
                "Use the person image to identify the affected limb region.\n"
                f"{options_text}\n"
                "Pick exactly one option whose category best matches. "
                'Return {"choices": ["X"]} where X is a single letter.\n'
                "Do not list more than one letter. Output only the JSON object."
            )
        else:
            prompt = (
                f"{query_text}\n"
                "Use the person image to identify the affected limb region.\n"
                f"{options_text}\n"
                "Select all options whose label matches that segment.\n"
                'Return a JSON object with key "choices" whose value is a list of '
                "option letters (A-H).\n"
                'If none match, return {"choices":[]}.\n'
                "Do not repeat the options or labels. Only output the JSON object "
                "(no extra text, no numbers)."
            )
    else:
        if is_category:
            prompt = f"""{query_text}

{options_text}

Output a single-letter answer as JSON: {{"choices": ["A"]}}. Pick exactly one letter; never list more than one."""
        else:
            prompt = f"""{query_text}

{options_text}

Respond with JSON only: {{"choices": ["A", "B"]}} or {{"choices": []}} if none match."""

    return prompt


def _strip_json_instructions(text: str) -> str:
    """Remove embedded JSON instruction sentences to avoid answer priming.

    Args:
        text: Raw query text from benchmark items.

    Returns:
        Query text with JSON instruction sentences removed.
    """
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    filtered = [sentence for sentence in sentences if "json" not in sentence.lower()]
    cleaned = " ".join(filtered).strip()
    return cleaned or text


def create_composite_image(
    person_image_path: str,
    grid_image_path: str,
    max_height: int = 1024,
) -> str:
    """Create a composite image combining person and options grid.

    Args:
        person_image_path: Path to person image.
        grid_image_path: Path to options grid image.
        max_height: Maximum height for the composite image.

    Returns:
        Path to the temporary composite image file.
    """
    # Load images
    person_img = Image.open(person_image_path).convert("RGB")
    grid_img = Image.open(grid_image_path).convert("RGB")

    # Resize both images to have the same height
    target_height = min(max_height, max(person_img.height, grid_img.height))

    # Calculate new dimensions maintaining aspect ratio
    person_scale = target_height / person_img.height
    person_new_width = int(person_img.width * person_scale)
    person_resized = person_img.resize((person_new_width, target_height), Image.LANCZOS)

    grid_scale = target_height / grid_img.height
    grid_new_width = int(grid_img.width * grid_scale)
    grid_resized = grid_img.resize((grid_new_width, target_height), Image.LANCZOS)

    # Create composite (person on left, grid on right)
    gap = 20  # Gap between images
    composite_width = person_new_width + gap + grid_new_width
    composite = Image.new("RGB", (composite_width, target_height), (255, 255, 255))

    composite.paste(person_resized, (0, 0))
    composite.paste(grid_resized, (person_new_width + gap, 0))

    # Save to a temporary file
    temp_file = tempfile.NamedTemporaryFile(
        suffix=".jpg", delete=False, prefix="prosthesis_match_composite_"
    )
    composite.save(temp_file.name, "JPEG", quality=95)

    return temp_file.name


def get_grid_path_for_item(
    item: Dict[str, Any],
    grids_dir: Path = DEFAULT_DATA_GRIDS_DIR,
) -> Optional[Path]:
    """Get the options grid image path for an item.

    Args:
        item: Benchmark item dictionary.
        grids_dir: Directory containing grid images.

    Returns:
        Path to grid image if exists, None otherwise.
    """
    item_id = item["item_id"]
    # Generic: "prosthesis_match:00000000" -> "grid_prosthesis_match_00000000.png"
    #          "compatibility_category:00000000" -> "grid_compatibility_category_00000000.png"
    grid_filename = f"grid_{item_id.replace(':', '_')}.png"
    grid_path = grids_dir / grid_filename

    if grid_path.exists():
        return grid_path

    # Fallback: legacy format "grid_prosthesis_match_{numeric_id}.png"
    numeric_id = item_id.split(":")[-1]
    legacy_path = grids_dir / f"grid_prosthesis_match_{numeric_id}.png"
    if legacy_path.exists():
        return legacy_path

    return None


def parse_model_output(
    raw_text: str,
    prefill_used: bool = False,
    *,
    allow_numeric: bool = False,
    prefill_text: str = ASSISTANT_PREFILL,
) -> Tuple[List[str], int]:
    """Parse model output to extract choices.

    L4: Robust parsing with multiple fallback strategies.

    Args:
        raw_text: Raw model output text.
        prefill_used: Whether assistant prefill was used.
        allow_numeric: Whether to map numeric choices (1-8) to letters.
        prefill_text: Prefill text used when reconstructing JSON.

    Returns:
        Tuple of (list of choices, parse_error flag).
    """
    # Clean up the text
    text = raw_text.strip()

    # If prefill was used, prepend it for complete JSON
    if prefill_used and not text.startswith("{"):
        text = prefill_text + text

    # Remove markdown code fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)

    # Try to find JSON object with choices array
    json_match = re.search(r"\{[^}]*\}", text)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Try "choices" key (new format)
            choices = data.get("choices")
            if choices is not None:
                if isinstance(choices, list):
                    valid_choices = []
                    seen = set()
                    for choice in choices:
                        normalized = _normalize_choice_token(
                            choice, allow_numeric=allow_numeric
                        )
                        if normalized and normalized not in seen:
                            valid_choices.append(normalized)
                            seen.add(normalized)
                    return valid_choices, 0
            # Fallback to "choice" key (old format, single choice)
            choice = data.get("choice")
            if choice and isinstance(choice, str) and choice.upper() in ALLOWED_CHOICES:
                return [choice.upper()], 0
        except json.JSONDecodeError:
            pass

    # Try to extract array pattern
    array_match = re.search(r'"choices"\s*:\s*\[(.*?)\]', text)
    if array_match:
        array_content = array_match.group(1)
        choices = re.findall(r'"([A-H])"', array_content)
        if allow_numeric and not choices:
            choices = re.findall(r"\b([1-8])\b", array_content)
        if choices or array_content.strip() == "":
            deduped = []
            seen = set()
            for choice in choices:
                normalized = _normalize_choice_token(choice, allow_numeric=allow_numeric)
                if normalized and normalized not in seen:
                    deduped.append(normalized)
                    seen.add(normalized)
            return deduped, 0

    # Try simple pattern matching for single choice (fallback)
    for choice in ALLOWED_CHOICES:
        patterns = [
            rf'"choice"\s*:\s*"{choice}"',
            rf"'choice'\s*:\s*'{choice}'",
            rf'"choice"\s*:\s*"{choice.lower()}"',
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return [choice], 0

    # Check if any single letter choice appears prominently
    found_choices = []
    for choice in ALLOWED_CHOICES:
        if re.search(rf"\b{choice}\b", text):
            found_choices.append(choice)
    if found_choices:
        return found_choices, 0

    # Fallback: map numeric outputs like "1, 2" to A-H
    if allow_numeric:
        digit_matches = re.findall(r"\b([1-8])\b", text)
        if digit_matches:
            deduped = []
            seen = set()
            for digit in digit_matches:
                normalized = _normalize_choice_token(
                    digit, allow_numeric=allow_numeric
                )
                if normalized and normalized not in seen:
                    deduped.append(normalized)
                    seen.add(normalized)
            if deduped:
                return deduped, 0

    return [], 1


def _normalize_choice_token(token: Any, *, allow_numeric: bool) -> Optional[str]:
    """Normalize a choice token into an allowed option letter.

    Args:
        token: Raw token from model output.

    Returns:
        Uppercase choice letter or None if invalid.
    """
    if isinstance(token, int):
        if allow_numeric and 1 <= token <= len(ALLOWED_CHOICES):
            return ALLOWED_CHOICES[token - 1]
        return None
    if isinstance(token, str):
        stripped = token.strip()
        if not stripped:
            return None
        upper = stripped.upper()
        if upper in ALLOWED_CHOICES:
            return upper
        if allow_numeric and stripped.isdigit():
            digit = int(stripped)
            if 1 <= digit <= len(ALLOWED_CHOICES):
                return ALLOWED_CHOICES[digit - 1]
    return None




def run_inference_single(
    model: Any,
    item: Dict[str, Any],
    config: InferenceConfig,
    guided_params: Optional[Dict[str, Any]] = None,
) -> PredictionRecord:
    """Run inference on a single item.

    Applies output constraints per docs/guides/llm_output_constraints.md:
    - L1: Prompt engineering (built into user prompt)
    - L2: Assistant prefill (if enabled and server mode)
    - L3: Guided decoding (if enabled)
    - L4: Robust parsing (in parse_model_output)

    Args:
        model: VisionLanguageModel instance.
        item: Benchmark item dictionary.
        config: Inference configuration.
        guided_params: Guided decoding parameters (L3).

    Returns:
        PredictionRecord.
    """
    item_id = item["item_id"]
    composite_path: Optional[str] = None

    # Build prompt (L1: includes JSON format instruction)
    user_prompt = build_user_prompt(
        item, config.use_grid, is_deepseek=config.deepseek_fixes
    )

    # Prepare image path for model
    person_path = _resolve_item_path(item["person_image"]["path"])

    if config.use_grid:
        # Try to find grid image and create composite
        grid_path = get_grid_path_for_item(item, config.grids_dir)
        if grid_path and person_path.exists():
            try:
                composite_path = create_composite_image(
                    str(person_path), str(grid_path)
                )
                image_path = composite_path
            except Exception as e:
                logger.warning(f"Failed to create composite for {item_id}: {e}")
                image_path = str(person_path)
        else:
            image_path = str(person_path)
    else:
        image_path = str(person_path)

    # Prepare generation kwargs for VLLMBackend
    gen_kwargs: Dict[str, Any] = {
        "image_path": image_path,
        "prompt": user_prompt,
        "max_tokens": config.max_tokens,
    }

    # L2: Assistant prefill (server mode only)
    prefill_used = False
    prefill_text = ASSISTANT_PREFILL
    if config.deepseek_fixes:
        prefill_text = DEEPSEEK_ASSISTANT_PREFILL
    if config.assistant_prefill and config.vllm_mode == "server":
        gen_kwargs["assistant_prefill"] = prefill_text
        prefill_used = True

    # L3: Guided decoding
    if guided_params:
        gen_kwargs["guided_info"] = guided_params

    # Run inference
    start_time = time.time()
    try:
        result = model.generate(**gen_kwargs)
        # VLLMBackend returns a dict with 'raw_text' key
        if isinstance(result, dict):
            raw_text = result.get("raw_text", "")
        else:
            raw_text = str(result)
    except Exception as e:
        logger.error(f"Inference error for {item_id}: {e}")
        raw_text = f"ERROR: {str(e)}"
    finally:
        # Clean up temporary composite image
        if composite_path:
            try:
                Path(composite_path).unlink(missing_ok=True)
            except Exception:
                pass

    latency_ms = int((time.time() - start_time) * 1000)

    # L4: Parse output with robust fallback
    choices, parse_error = parse_model_output(
        raw_text,
        prefill_used=prefill_used,
        allow_numeric=not config.deepseek_fixes,
        prefill_text=prefill_text,
    )

    return PredictionRecord(
        item_id=item_id,
        model_id=config.model_name,
        choices=choices,
        raw_text=raw_text,
        parse_error=parse_error,
        latency_ms=latency_ms,
        timestamp=datetime.now().isoformat(),
    )


def prepare_batch_request(
    item: Dict[str, Any],
    config: InferenceConfig,
) -> Tuple[Dict[str, Any], Optional[str]]:
    """Prepare a single item for batch inference.

    Args:
        item: Benchmark item dictionary.
        config: Inference configuration.

    Returns:
        Tuple of (request dict for batch, temp_file_path or None).
    """
    item_id = item["item_id"]
    user_prompt = build_user_prompt(
        item, config.use_grid, is_deepseek=config.deepseek_fixes
    )
    person_path = _resolve_item_path(item["person_image"]["path"])
    composite_path: Optional[str] = None

    if config.use_grid:
        grid_path = get_grid_path_for_item(item, config.grids_dir)
        if grid_path and person_path.exists():
            try:
                composite_path = create_composite_image(
                    str(person_path), str(grid_path)
                )
                image_path = composite_path
            except Exception as e:
                logger.warning(f"Failed to create composite for {item_id}: {e}")
                image_path = str(person_path)
        else:
            image_path = str(person_path)
    else:
        image_path = str(person_path)

    request = {
        "item_id": item_id,
        "image_path": image_path,
        "prompt": user_prompt,
    }

    return request, composite_path


def run_inference_batch(
    model: Any,
    items: List[Dict[str, Any]],
    config: InferenceConfig,
    guided_params: Optional[Dict[str, Any]] = None,
) -> List[PredictionRecord]:
    """Run batch inference on multiple items.

    批量推理 | Batch Inference

    Args:
        model: VisionLanguageModel instance with generate_batch method.
        items: List of benchmark items.
        config: Inference configuration.
        guided_params: Guided decoding parameters.

    Returns:
        List of PredictionRecord objects.
    """
    if not items:
        return []

    # Prepare all requests
    requests = []
    temp_files = []
    for item in items:
        req, temp_path = prepare_batch_request(item, config)
        requests.append(req)
        temp_files.append(temp_path)

    # Prepare generation kwargs
    gen_kwargs: Dict[str, Any] = {
        "requests": requests,
        "max_tokens": config.max_tokens,
        "max_concurrent": config.max_concurrent,
    }

    # L2: Assistant prefill
    prefill_used = False
    prefill_text = ASSISTANT_PREFILL
    if config.deepseek_fixes:
        prefill_text = DEEPSEEK_ASSISTANT_PREFILL
    if config.assistant_prefill and config.vllm_mode == "server":
        gen_kwargs["assistant_prefill"] = prefill_text
        prefill_used = True

    # L3: Guided decoding
    if guided_params:
        gen_kwargs["guided_info"] = guided_params

    # Run batch inference
    start_time = time.time()
    try:
        results = model.generate_batch(**gen_kwargs)
    except Exception as e:
        logger.error(f"Batch inference error: {e}")
        # Fallback: return error records for all items
        return [
            PredictionRecord(
                item_id=item["item_id"],
                model_id=config.model_name,
                choices=[],
                raw_text=f"ERROR: {str(e)}",
                parse_error=1,
                latency_ms=0,
                timestamp=datetime.now().isoformat(),
            )
            for item in items
        ]
    finally:
        # Clean up all temporary files
        for temp_path in temp_files:
            if temp_path:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception:
                    pass

    total_latency_ms = int((time.time() - start_time) * 1000)
    avg_latency_ms = total_latency_ms // len(items) if items else 0

    # Process results
    predictions = []
    for i, (item, result) in enumerate(zip(items, results)):
        raw_text = result.get("raw_text", "")
        choices, parse_error = parse_model_output(
            raw_text,
            prefill_used=prefill_used,
            allow_numeric=not config.deepseek_fixes,
            prefill_text=prefill_text,
        )

        predictions.append(
            PredictionRecord(
                item_id=item["item_id"],
                model_id=config.model_name,
                choices=choices,
                raw_text=raw_text,
                parse_error=parse_error,
                latency_ms=avg_latency_ms,
                timestamp=datetime.now().isoformat(),
            )
        )

    return predictions


def run_inference(
    config: InferenceConfig,
) -> List[PredictionRecord]:
    """Run inference on all benchmark items.

    Args:
        config: Inference configuration.

    Returns:
        List of PredictionRecord objects.
    """
    # Import model registry
    try:
        from scripts.core.backends import create_backend
        from scripts.core.model_registry import get_model_config, resolve_base_url
    except ImportError:
        logger.error(
            "Could not import model registry. Please check scripts.core module."
        )
        raise

    model_cfg = get_model_config(config.model_name)
    config.deepseek_fixes = model_cfg.get("family") == "deepseek-vl2"
    if config.deepseek_fixes and config.vllm_mode == "server":
        config.assistant_prefill = True

    # Determine run directory
    # Default: output_dir is already model-specific (set by evaluate.py or CLI)
    run_dir = config.output_dir
    predictions_path = run_dir / "predictions.jsonl"

    if config.overwrite and predictions_path.exists():
        # Overwrite mode: remove existing predictions
        logger.info(f"Overwrite mode: removing existing {predictions_path}")
        predictions_path.unlink()

    run_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = run_dir / "predictions.jsonl"

    # Load items
    items = []
    with open(config.items_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    logger.info(f"Loaded {len(items)} benchmark items")

    # Apply max_samples limit
    if config.max_samples and config.max_samples < len(items):
        items = items[: config.max_samples]
        logger.info(f"Limited to {config.max_samples} items")

    # Load completed predictions if resuming
    completed_ids: Set[str] = set()
    if (config.resume or not config.overwrite) and predictions_path.exists():
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    pred = json.loads(line)
                    completed_ids.add(pred["item_id"])
        if completed_ids:
            logger.info(
                f"Found {len(completed_ids)} completed predictions, resuming..."
            )

    # Calculate remaining items
    remaining_items = [item for item in items if item["item_id"] not in completed_ids]
    total_remaining = len(remaining_items)
    logger.info(f"Items to process: {total_remaining} / {len(items)}")

    # L3: Initialize guided decoding params (scope enum to the actual option IDs in use)
    observed_ids = sorted({
        str(opt.get("option_id"))
        for item in items
        for opt in item.get("options", [])
        if opt.get("option_id") is not None
    })
    # Category variant is single-pick; cap maxItems so thinking models can't dump every letter
    all_category = bool(items) and all(
        (it.get("answer") or {}).get("mode") == "category" for it in items
    )
    max_items_cap = 1 if all_category else None
    guided_params = get_guided_decoding_params(
        config.guided_decoding,
        is_deepseek=config.deepseek_fixes,
        allowed_ids=observed_ids or None,
        max_items=max_items_cap,
    )
    if observed_ids:
        logger.info(
            f"L3: guided-decoding option space: {observed_ids}"
            + (f" (single-pick, maxItems=1)" if max_items_cap == 1 else "")
        )
    if config.guided_decoding:
        if guided_params:
            logger.info("L3: Guided decoding enabled")
        else:
            logger.warning("L3: Guided decoding requested but not available")

    # Log constraint settings
    logger.info("Output constraint settings:")
    logger.info(f"  L1: Prompt engineering = enabled (always)")
    logger.info(f"  L2: Assistant prefill = {config.assistant_prefill}")
    logger.info(f"  L3: Guided decoding = {guided_params is not None}")
    logger.info(f"  L4: Robust parsing = enabled (always)")
    logger.info(f"  DeepSeek fixes = {config.deepseek_fixes}")
    logger.info(f"  Batch size: {config.batch_size}")
    logger.info(f"  Max concurrent: {config.max_concurrent}")

    # Initialize model
    logger.info(f"Initializing model: {config.model_name}")
    backend_kwargs: Dict[str, Any] = {}
    if config.vllm_port is not None:
        if model_cfg.get("backend") != "vllm":
            logger.warning(
                "Ignoring --vllm-port for non-vllm backend: %s", config.model_name
            )
        elif config.vllm_mode != "server":
            logger.warning(
                "Ignoring --vllm-port because vllm_mode=%s", config.vllm_mode
            )
        else:
            backend_kwargs["api_base"] = resolve_base_url(
                base_url=None,
                model_cfg=model_cfg,
                port=config.vllm_port,
                force_port=True,
            )
            logger.info("Using vLLM server port override: %s", config.vllm_port)
    model = create_backend(config.model_name, mode=config.vllm_mode, **backend_kwargs)

    # Check if batch processing is supported
    use_batch = config.batch_size > 1 and hasattr(model, "generate_batch")
    if config.batch_size > 1 and not use_batch:
        logger.warning(
            f"Batch size {config.batch_size} requested but model does not support "
            "generate_batch. Falling back to sequential processing."
        )

    # Run inference
    predictions: List[PredictionRecord] = []
    failures: List[Dict[str, Any]] = []

    # ETA tracking
    start_time = time.time()
    processed_count = 0

    with open(predictions_path, "a", encoding="utf-8") as pred_file:
        if use_batch:
            # Batch processing mode
            logger.info(f"Using batch processing (batch_size={config.batch_size})")
            for batch_start in range(0, len(remaining_items), config.batch_size):
                batch_end = min(batch_start + config.batch_size, len(remaining_items))
                batch_items = remaining_items[batch_start:batch_end]

                # Run batch inference
                batch_preds = run_inference_batch(
                    model, batch_items, config, guided_params
                )

                for pred in batch_preds:
                    predictions.append(pred)
                    processed_count += 1

                    # Write prediction immediately
                    pred_file.write(
                        json.dumps(pred.to_dict(), ensure_ascii=False) + "\n"
                    )
                    pred_file.flush()

                    # Track failures
                    if pred.parse_error:
                        failures.append({
                            "item_id": pred.item_id,
                            "raw_text": pred.raw_text,
                            "error": "parse_error",
                        })

                # Progress logging
                elapsed = time.time() - start_time
                avg_time = elapsed / processed_count if processed_count > 0 else 0
                remaining = total_remaining - processed_count
                eta_seconds = avg_time * remaining

                if eta_seconds > 3600:
                    eta_str = f"{eta_seconds/3600:.1f}h"
                elif eta_seconds > 60:
                    eta_str = f"{eta_seconds/60:.1f}m"
                else:
                    eta_str = f"{eta_seconds:.0f}s"

                logger.info(
                    f"Progress: {processed_count}/{total_remaining} "
                    f"({processed_count/total_remaining*100:.1f}%) | "
                    f"Avg: {avg_time*1000:.0f}ms/item | "
                    f"ETA: {eta_str} | "
                    f"Batch: {len(batch_items)} items"
                )
        else:
            # Sequential processing mode
            for i, item in enumerate(items):
                item_id = item["item_id"]

                if item_id in completed_ids:
                    continue

                # Run inference with constraint params
                pred = run_inference_single(model, item, config, guided_params)
                predictions.append(pred)
                processed_count += 1

                # Write prediction immediately (real-time recording)
                pred_file.write(json.dumps(pred.to_dict(), ensure_ascii=False) + "\n")
                pred_file.flush()

                # Track failures
                if pred.parse_error:
                    failures.append({
                        "item_id": item_id,
                        "raw_text": pred.raw_text,
                        "error": "parse_error",
                    })

                # Progress logging with ETA
                if processed_count % 10 == 0 or processed_count == total_remaining:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / processed_count
                    remaining = total_remaining - processed_count
                    eta_seconds = avg_time * remaining

                    # Format ETA
                    if eta_seconds > 3600:
                        eta_str = f"{eta_seconds/3600:.1f}h"
                    elif eta_seconds > 60:
                        eta_str = f"{eta_seconds/60:.1f}m"
                    else:
                        eta_str = f"{eta_seconds:.0f}s"

                    logger.info(
                        f"Progress: {processed_count}/{total_remaining} "
                        f"({processed_count/total_remaining*100:.1f}%) | "
                        f"Avg: {avg_time*1000:.0f}ms/item | "
                        f"ETA: {eta_str}"
                    )

    # Save failures
    failures_path = run_dir / "failures.jsonl"
    with open(failures_path, "w", encoding="utf-8") as f:
        for failure in failures:
            f.write(json.dumps(failure, ensure_ascii=False) + "\n")

    # Save run manifest
    manifest = {
        "model_name": config.model_name,
        "items_path": str(config.items_path),
        "total_items": len(items),
        "total_predictions": len(predictions) + len(completed_ids),
        "new_predictions": len(predictions),
        "resumed_from": len(completed_ids),
        "total_failures": len(failures),
        "timestamp": datetime.now().isoformat(),
        "config": {
            "max_samples": config.max_samples,
            "seed": config.seed,
            "use_grid": config.use_grid,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "resume": config.resume,
            "overwrite": config.overwrite,
            "batch_size": config.batch_size,
            "max_concurrent": config.max_concurrent,
            "deepseek_fixes": config.deepseek_fixes,
        },
        "output_constraints": {
            "L1_prompt_engineering": True,
            "L2_assistant_prefill": config.assistant_prefill,
            "L3_guided_decoding": guided_params is not None,
            "L4_robust_parsing": True,
        },
    }
    manifest_path = run_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    logger.info(f"Completed {len(predictions)} predictions, {len(failures)} failures")
    logger.info(f"Results saved to {run_dir}")

    return predictions


def configure_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _infer_variant_name(items_path: Path) -> str:
    """Infer the public Prosthesis Matching variant from the items file name."""

    name = items_path.name.lower()
    if "compatibility_category" in name:
        return "compatibility_category"
    return "compatibility_diversity"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run VLM inference for Prosthesis Matching."
    )
    parser.add_argument(
        "--items-path",
        type=Path,
        default=DEFAULT_DATA_ITEMS_PATH,
        help=f"Path to items JSONL file. Default: {DEFAULT_DATA_ITEMS_PATH}",
    )
    parser.add_argument(
        "--grids-dir",
        type=Path,
        default=DEFAULT_DATA_GRIDS_DIR,
        help=f"Directory containing grid images. Default: {DEFAULT_DATA_GRIDS_DIR}",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Name of the model to use.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Final output directory for predictions. Default: "
            f"{DEFAULT_RUNS_DIR}/<model>/<variant>"
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to process.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--use-grid",
        action="store_true",
        default=True,
        help="Use grid images for options (default: True).",
    )
    parser.add_argument(
        "--no-grid",
        action="store_false",
        dest="use_grid",
        help="Use individual option images instead of grid.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens for model output. Default: 1024",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--vllm-mode",
        type=str,
        default="server",
        choices=["server", "local"],
        help="VLLM mode. Default: server",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="Override vLLM server port (server mode only).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from previous run (use with --run-dir to specify directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions in output directory.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Specific run directory (for resume/overwrite). If not set, creates"
            " timestamped dir."
        ),
    )
    parser.add_argument(
        "--assistant-prefill",
        action="store_true",
        help="L2: Enable assistant prefill (server mode only).",
    )
    parser.add_argument(
        "--guided-decoding",
        action="store_true",
        help="L3: Enable guided decoding for choice constraint.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for parallel inference. Default: 1 (sequential).",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=8,
        help="Max concurrent requests for server mode batch. Default: 8",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    configure_logging(args.log_level)

    logger.info("=" * 60)
    logger.info("Prosthesis Matching: Run Inference")
    logger.info("=" * 60)

    # Determine output directory. Explicit --output-dir is treated as the final
    # run directory so public reproduce commands and orchestrator roots align.
    if args.run_dir:
        output_dir = args.run_dir
    elif args.output_dir:
        output_dir = args.output_dir
    else:
        variant = _infer_variant_name(args.items_path)
        output_dir = DEFAULT_RUNS_DIR / args.model_name / variant

    # Create config
    config = InferenceConfig(
        model_name=args.model_name,
        items_path=args.items_path,
        output_dir=output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
        use_grid=args.use_grid,
        grids_dir=args.grids_dir,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        vllm_mode=args.vllm_mode,
        vllm_port=args.vllm_port,
        resume=args.resume,
        overwrite=args.overwrite,
        log_level=args.log_level,
        assistant_prefill=args.assistant_prefill,
        guided_decoding=args.guided_decoding,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )

    # Run inference
    predictions = run_inference(config)

    logger.info("=" * 60)
    logger.info(f"Done! Generated {len(predictions)} predictions.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
