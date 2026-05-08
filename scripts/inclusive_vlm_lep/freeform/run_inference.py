"""Run Attribution-FreeForm inference with an official or local backend."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from scripts.core.backends import create_backend
from scripts.core.model_registry import get_model_config, resolve_base_url

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUBSET_JSONL = (
    PROJECT_ROOT / "data" / "inclusive_vlm_lep" / "freeform" / "freeform_queries.jsonl"
)
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "outputs" / "inclusive_vlm_lep" / "freeform" / "runs"
DEFAULT_MAX_TOKENS = 256


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file."""

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _load_existing_ids(predictions_path: Path) -> set[str]:
    """Load completed record IDs for resume mode."""

    if not predictions_path.exists():
        return set()
    record_ids: set[str] = set()
    for record in _load_jsonl(predictions_path):
        record_id = str(record.get("record_id") or "").strip()
        if record_id:
            record_ids.add(record_id)
    return record_ids


def _resolve_image_path(raw_path: str | Path) -> Path:
    """Resolve image paths relative to the repository root."""

    path = Path(raw_path)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate


def _default_output_dir(model_name: str) -> Path:
    """Build the default run directory for a model."""

    return DEFAULT_RUNS_ROOT / model_name


def _build_backend(
    model_name: str,
    *,
    max_tokens: int,
    temperature: float,
    max_image_size: int | None,
) -> Any:
    """Build a backend from scripts.core.backends."""

    kwargs: dict[str, Any] = {"max_tokens": max_tokens, "temperature": temperature}
    if max_image_size:
        kwargs["max_image_size"] = max_image_size
    return create_backend(model_name, mode="server", **kwargs)


def run_inference(
    *,
    model_name: str,
    subset_jsonl: Path,
    output_dir: Path | None = None,
    resume: bool = False,
    overwrite: bool = False,
    max_samples: int = -1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = 0.0,
    max_image_size: int | None = None,
    env_file: Path | None = None,
) -> Path:
    """Run Attribution-FreeForm inference and write predictions.jsonl.

    Args:
        model_name: Model registry key.
        subset_jsonl: Path to the Attribution-FreeForm subset JSONL.
        output_dir: Optional output report directory.
        resume: Skip record_ids already written to predictions.jsonl.
        overwrite: Remove an existing predictions.jsonl before writing.
        max_samples: Optional sample cap. `-1` keeps all.
        max_tokens: Backend max token limit.
        temperature: Backend sampling temperature.
        max_image_size: Optional longer-side resize for images.
        env_file: Optional `.env` file for backend credentials.

    Returns:
        Path to the written predictions JSONL.
    """

    if env_file:
        load_dotenv(env_file)
    else:
        load_dotenv(PROJECT_ROOT / ".env")

    output_dir = output_dir or _default_output_dir(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    if overwrite and predictions_path.exists():
        predictions_path.unlink()

    records = _load_jsonl(subset_jsonl)
    if max_samples >= 0:
        records = records[:max_samples]
    existing_ids = _load_existing_ids(predictions_path) if resume else set()
    pending = [
        record
        for record in records
        if str(record.get("record_id") or "").strip() not in existing_ids
    ]
    if not pending:
        LOGGER.info("No pending records for %s", predictions_path)
        return predictions_path

    backend = _build_backend(
        model_name,
        max_tokens=max_tokens,
        temperature=temperature,
        max_image_size=max_image_size,
    )

    with predictions_path.open("a", encoding="utf-8") as handle:
        for index, item in enumerate(pending):
            record_id = str(item.get("record_id") or "")
            image_path = _resolve_image_path(item.get("image_path") or "")
            prompt = str(item.get("prompt") or "")
            start_time = time.perf_counter()
            raw_output = ""

            img_for_call = str(image_path) if image_path.exists() else ""
            result = backend.generate(img_for_call, prompt)
            raw_output = str(
                result.get("raw_text")
                or result.get("parsed_answer")
                or result.get("answer_text")
                or ""
            ).strip()

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            prediction = {
                "record_id": record_id,
                "image_id": item.get("image_id"),
                "raw_output": raw_output,
                "latency_ms": latency_ms,
                "model_name": model_name,
            }
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()

            if (index + 1) % 10 == 0 or index + 1 == len(pending):
                LOGGER.info(
                    "Processed %d/%d records for %s",
                    index + 1,
                    len(pending),
                    model_name,
                )

    LOGGER.info("Predictions written to %s", predictions_path)
    return predictions_path


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run Attribution-FreeForm inference."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        required=True,
        help="Model name from the registry.",
    )
    parser.add_argument(
        "--subset-jsonl",
        type=Path,
        default=DEFAULT_SUBSET_JSONL,
        help="Path to the Attribution-FreeForm subset JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output run directory (default: outputs/inclusive_vlm_lep/freeform/runs/<model>/).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing predictions.jsonl.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite an existing predictions.jsonl before writing.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Maximum number of subset records to run (-1 keeps all).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Backend max token limit.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Backend sampling temperature.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=0,
        help="Optional max image size; 0 disables resizing.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional .env file for backend credentials.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    configure_logging(args.log_level)
    run_inference(
        model_name=args.model_name,
        subset_jsonl=args.subset_jsonl,
        output_dir=args.output_dir,
        resume=args.resume,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_image_size=args.max_image_size if args.max_image_size > 0 else None,
        env_file=args.env_file,
    )


if __name__ == "__main__":
    main()
