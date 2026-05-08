"""Run model inference for Limb-Evidence Grounding (spec_009)."""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

from . import config as benchmark_config
from .eval_config import (
    BENCHMARK_NAME,
    BENCHMARK_PART,
    DEFAULT_MAX_TOKENS,
    DEFAULT_QUERIES_PATH,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    DEFAULT_TASKS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    PROJECT_ROOT,
    get_model_predictions_path,
    get_model_run_dir,
)
from ..model_registry import (
    VisionLanguageModel,
    build_model,
    build_model_config,
    build_model_from_config,
    resolve_base_url,
)

OPTION_TO_LABEL = {v: k for k, v in benchmark_config.PRESENCE_LABEL_TO_OPTION.items()}
PRESENCE_OPTIONS = ["a", "b", "c", "d"]


@dataclass
class InferenceConfig:
    """Configuration for inference run."""

    model_name: str
    queries_path: Path
    output_path: Path
    tasks: Set[str]
    max_samples: int
    seed: int
    shuffle: bool
    overwrite: bool
    max_tokens: int | None = None
    log_level: str = "INFO"
    vllm_mode: str = "server"
    vllm_port: int | None = None
    assistant_prefill: bool = False
    max_model_len: int = 4096
    max_image_size: int | None = None
    resume: bool = False
    resume_failed: bool = False
    api_batch_size: int = 1
    api_batch_delay: float = 0.0
    progress_interval: int = 100


def configure_logging(level: str = "INFO") -> None:
    """Configure basic console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run model inference for Limb-Evidence Grounding."
    )
    parser.add_argument("--model-name", required=True, help="Model name in registry.")
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to queries.jsonl.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output JSONL path for predictions.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Comma separated tasks to run: recognition,attribution,all.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of samples for debugging (-1 for all).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override max_tokens when generating (optional).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Max model len for local vLLM mode.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=0,
        help="Resize images to this max longer side (px) with aspect ratio preserved; 0 to disable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed (used when shuffling).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle evaluation order.",
    )
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )
    parser.add_argument(
        "--vllm-mode",
        type=str,
        choices=["server", "local"],
        default="server",
        help="vLLM invocation mode: server (OpenAI-compatible) or local LLM.",
    )
    parser.add_argument(
        "--vllm-port",
        type=int,
        default=None,
        help="Override vLLM server port (server mode only).",
    )
    parser.add_argument(
        "--assistant-prefill",
        action="store_true",
        help="Use vLLM assistant_prefill to constrain JSON-only answers (server mode).",
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="If output exists, reuse successful records and only re-run missing/failed ones.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output (skip all cached records).",
    )
    parser.add_argument(
        "--api-batch-size",
        type=int,
        default=1,
        help="Only for GPT/Gemini backends: concurrent API calls to improve throughput.",
    )
    parser.add_argument(
        "--api-batch-delay",
        type=float,
        default=0.0,
        help="Only for GPT/Gemini backends: delay seconds between API batches to ease rate limits.",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=100,
        help="Log progress/ETA every N processed samples.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR).",
    )
    return parser.parse_args()


def parse_tasks(raw: str) -> Set[str]:
    """Parse task list string to a set."""

    if not raw or raw.lower() == "all":
        return set(DEFAULT_TASKS)
    tasks = {t.strip().lower() for t in raw.split(",") if t.strip()}
    unsupported = tasks - set(DEFAULT_TASKS)
    if unsupported:
        raise ValueError(f"Unsupported tasks: {unsupported}")
    return tasks


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    """Load queries JSONL into memory."""

    records: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _resolve_image_path(raw_path: str | Path) -> Path:
    """Resolve image path relative to project root when needed."""

    path = Path(raw_path)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    return candidate


def _parse_presence_answer(raw_text: str) -> str | None:
    """Parse model raw output into a/b/c/d option."""

    text = raw_text.strip()
    # Try JSON first
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            ans = data.get("answer")
            if isinstance(ans, str) and ans.lower() in {"a", "b", "c", "d"}:
                return ans.lower()
    except Exception:
        pass

    lowered = text.lower()
    match = re.search(r"\b([abcd])\b", lowered)
    if match:
        return match.group(1)
    return None


def _clean_token(token: str) -> str:
    """Remove punctuation/backticks around a token."""

    return token.strip(" `\"'").strip()


def _parse_segment_labels(raw_text: str, label_vocab: Sequence[str]) -> List[str]:
    """Parse comma separated labels into a deduplicated list."""

    # JSON path first
    try:
        data = json.loads(raw_text)
        if isinstance(data, dict) and isinstance(data.get("labels"), list):
            labels = []
            for item in data["labels"]:
                if isinstance(item, str):
                    labels.append(item)
            return [label for label in labels if label and label != "none"]
    except Exception:
        pass

    vocab_set = {label.lower() for label in label_vocab}
    tokens = [tok.strip() for tok in raw_text.split(",")]
    parsed: List[str] = []
    seen = set()
    for token in tokens:
        cleaned = _clean_token(token)
        if not cleaned:
            continue
        cleaned_lower = cleaned.lower()
        if cleaned_lower not in vocab_set:
            continue
        if cleaned_lower == "none":
            # Only keep none when it is the only token.
            continue
        if cleaned_lower not in seen:
            parsed.append(cleaned_lower)
            seen.add(cleaned_lower)
    if not parsed and any(tok.lower().strip() == "none" for tok in tokens):
        return []
    return parsed


def _format_prompt(query: Dict[str, Any]) -> str:
    """Append JSON-only instruction to the base prompt."""

    base = query.get("prompt", "")
    task_name = query.get("task_name")
    if task_name == "recognition":
        return (
            f"{base}\n"
            'Respond only with JSON: {"answer": "a|b|c|d"} (lowercase, no extra text).'
        )
    if task_name == "attribution":
        labels = query.get("label_vocab", [])
        labels_str = ", ".join(labels)
        return (
            f"{base}\n"
            f'Respond only with JSON: {{"labels": [<labels from: {labels_str}>]}}; '
            "use an empty list when none."
        )
    return base


def _format_assistant_prefill(query: Dict[str, Any]) -> str:
    """Build assistant prefill text to enforce JSON outputs."""

    task_name = query.get("task_name")
    if task_name == "recognition":
        return '{"answer": '
    if task_name == "attribution":
        return '{"labels": ['
    return ""


def _build_guided_info(query: Dict[str, Any]) -> Dict[str, Any] | None:
    """Construct guided decoding schema/info for vLLM."""

    task_name = query.get("task_name")
    if task_name == "recognition":
        # Use JSON schema for presence to ensure consistent JSON output
        # matching the prompt requirement: {"answer": "..."}
        return {
            "type": "json",
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {"type": "string", "enum": ["a", "b", "c", "d"]}
                },
                "required": ["answer"],
                "additionalProperties": False,
            },
        }
    if task_name == "attribution":
        vocab = query.get("label_vocab") or []
        schema = {
            "type": "object",
            "properties": {
                "labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": vocab},
                }
            },
            "required": ["labels"],
            "additionalProperties": False,
        }
        return {"type": "json", "schema": schema}
    return None


def _build_result_record(
    model: VisionLanguageModel,
    query: Dict[str, Any],
    raw_text: str,
    parsed_answer: str | None,
    prediction_labels: List[str],
    latency_ms: int,
    usage: Dict[str, Any] | None,
    max_tokens: int | None,
    success: bool,
) -> Dict[str, Any]:
    """Assemble a JSONL record for a single prediction."""

    record = {
        "benchmark_name": query.get("benchmark_name", BENCHMARK_NAME),
        "benchmark_part": query.get("benchmark_part", BENCHMARK_PART),
        "model_name": getattr(model, "model_name", None),
        "backend": getattr(model, "backend", None),
        "task_name": query.get("task_name"),
        "image_id": query.get("image_id"),
        "group_id": query.get("group_id"),
        "prompt_id": query.get("prompt_id"),
        "target_answer": query.get("target_answer"),
        "target_labels": query.get("target_labels") or [],
        "prediction_raw": raw_text,
        "prediction_parsed": parsed_answer,
        "prediction_labels": prediction_labels,
        "latency_ms": latency_ms,
        "meta": {
            "temperature": getattr(model, "temperature", DEFAULT_TEMPERATURE),
            "top_p": getattr(model, "top_p", DEFAULT_TOP_P),
            "max_tokens": max_tokens or getattr(model, "max_tokens", None),
            "usage": usage or {},
        },
        "success": success,
    }
    if "label_vocab" in query:
        record["label_vocab"] = query["label_vocab"]
    return record


def _run_single_inference(
    model: VisionLanguageModel,
    query: Dict[str, Any],
    max_tokens: int | None,
    assistant_prefill: bool,
) -> Dict[str, Any]:
    """Run inference for a single query."""

    task_name = query["task_name"]
    prompt = _format_prompt(query)
    image_path = _resolve_image_path(query["image_path"])
    raw_text = ""
    usage: Dict[str, Any] | None = None
    success = True
    start = time.time()
    backend_name = getattr(model, "backend", "")
    generate_kwargs: Dict[str, Any] = {
        "image_path": str(image_path),
        "prompt": prompt,
        "task_name": task_name,
        "max_tokens": max_tokens,
    }
    if backend_name == "vllm":
        generate_kwargs["guided_info"] = _build_guided_info(query)
        if assistant_prefill:
            generate_kwargs["assistant_prefill"] = _format_assistant_prefill(query)
    try:
        result = model.generate(**generate_kwargs)
        raw_text = result.get("raw_text", "") or ""
        usage = result.get("usage")
        success = result.get("success", True) and bool(raw_text.strip())
    except Exception as exc:  # noqa: BLE001
        success = False
        logging.exception(
            "Generation failed for image_id=%s prompt_id=%s: %s",
            query.get("image_id"),
            query.get("prompt_id"),
            exc,
        )
    latency_ms = int((time.time() - start) * 1000)

    if task_name == "recognition":
        parsed_answer = _parse_presence_answer(raw_text)
        prediction_labels: List[str] = []
    else:
        parsed_answer = raw_text or None
        prediction_labels = _parse_segment_labels(raw_text, query.get("label_vocab", []))

    return _build_result_record(
        model=model,
        query=query,
        raw_text=raw_text,
        parsed_answer=parsed_answer,
        prediction_labels=prediction_labels,
        latency_ms=latency_ms,
        usage=usage,
        max_tokens=max_tokens,
        success=success,
    )


def _format_eta(seconds: float) -> str:
    """Format seconds into xdy zh ym."""

    total = int(max(0, seconds))
    days, rem = divmod(total, 86_400)
    hours, rem = divmod(rem, 3600)
    minutes, _ = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m"


def _load_existing_results(path: Path) -> Dict[Tuple[str, int, str, int], Dict[str, Any]]:
    """Load existing results indexed by unique key."""

    data: Dict[Tuple[str, int, str, int], Dict[str, Any]] = {}
    if not path.exists():
        return data
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            key = (
                rec.get("task_name"),
                rec.get("image_id"),
                rec.get("group_id"),
                rec.get("prompt_id"),
            )
            data[key] = rec
    return data


def run_inference(config: InferenceConfig) -> None:
    """Execute inference pipeline."""

    configure_logging(config.log_level)
    random.seed(config.seed)
    tasks = config.tasks or set(DEFAULT_TASKS)

    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not (
        config.overwrite or config.resume_failed or config.resume
    ):
        raise FileExistsError(
            f"Output file {output_path} exists. Use --overwrite-output to replace or "
            "--resume/--resume-failed."
        )

    logging.info("Loading queries from %s", config.queries_path)
    queries = _load_queries(config.queries_path)
    filtered = [q for q in queries if q.get("task_name") in tasks]
    if config.shuffle:
        random.shuffle(filtered)
    if config.max_samples >= 0:
        filtered = filtered[: config.max_samples]
    total = len(filtered)
    if total == 0:
        logging.info("No queries to process for tasks=%s", ",".join(sorted(tasks)))
        return

    prefer_local = config.vllm_mode == "local"
    model_cfg = build_model_config(config.model_name, prefer_local)
    if prefer_local:
        model_cfg["max_model_len"] = config.max_model_len
    if config.max_image_size and config.max_image_size > 0:
        model_cfg["max_image_size"] = config.max_image_size
    if config.vllm_port is not None:
        if model_cfg.get("backend") != "vllm":
            logging.warning(
                "Ignoring --vllm-port for non-vllm backend: %s", config.model_name
            )
        elif config.vllm_mode != "server":
            logging.warning(
                "Ignoring --vllm-port because vllm_mode=%s", config.vllm_mode
            )
        else:
            model_cfg["api_base"] = resolve_base_url(
                base_url=None,
                model_cfg=model_cfg,
                port=config.vllm_port,
                force_port=True,
            )
            logging.info("Using vLLM server port override: %s", config.vllm_port)
    backend_name = model_cfg.get("backend", "")
    is_api_backend = backend_name in {"openai", "gemini"}
    batch_size = config.api_batch_size if is_api_backend else 1
    batch_delay = config.api_batch_delay if is_api_backend else 0.0

    def _write_records(records: List[Dict[str, Any]]) -> None:
        with output_path.open("w", encoding="utf-8") as writer:
            for rec in records:
                writer.write(json.dumps(rec, ensure_ascii=False))
                writer.write("\n")

    logging.info("Starting inference: %d samples for tasks=%s", total, ",".join(sorted(tasks)))
    start_time = time.time()

    processed = 0
    if is_api_backend and batch_size > 1:
        existing = (
            _load_existing_results(output_path)
            if (config.resume_failed or config.resume)
            else {}
        )
        records_map: Dict[int, Dict[str, Any]] = {}
        to_run: List[Tuple[int, Dict[str, Any]]] = []
        processed_done = 0
        for idx, query in enumerate(filtered):
            key = (
                query.get("task_name"),
                query.get("image_id"),
                query.get("group_id"),
                query.get("prompt_id"),
            )
            cached = existing.get(key)
            if cached and (config.resume or cached.get("success")):
                records_map[idx] = cached
                processed_done += 1
            else:
                to_run.append((idx, query))

        def _worker(task: Tuple[int, Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
            idx, q = task
            local_model = build_model_from_config(model_cfg)
            rec = _run_single_inference(
                local_model, q, config.max_tokens, config.assistant_prefill
            )
            return idx, rec

        chunks = [to_run[i : i + batch_size] for i in range(0, len(to_run), batch_size)]
        interval = max(config.progress_interval, 1)
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            for ci, chunk in enumerate(chunks):
                futures = [executor.submit(_worker, t) for t in chunk]
                for fut in as_completed(futures):
                    idx, rec = fut.result()
                    records_map[idx] = rec
                    processed_done += 1
                    if processed_done % interval == 0 or processed_done == total:
                        elapsed = time.time() - start_time
                        rate = processed_done / elapsed if elapsed > 0 else 0
                        remaining = (total - processed_done) / rate if rate > 0 else 0
                        logging.info(
                            "Processed %d/%d | ETA %s",
                            processed_done,
                            total,
                            _format_eta(remaining),
                        )
                if batch_delay > 0 and ci < len(chunks) - 1:
                    time.sleep(batch_delay)
                # incremental save snapshot
                ordered_records = [records_map[i] for i in sorted(records_map.keys())]
                _write_records(ordered_records)
        processed = processed_done
    else:
        model = build_model_from_config(model_cfg)
        if config.resume_failed or config.resume:
            existing = _load_existing_results(output_path)
        else:
            existing = {}
        interval = max(config.progress_interval, 1)
        with output_path.open("w", encoding="utf-8") as writer:
            for idx, query in enumerate(filtered, start=1):
                processed = idx
                key = (
                    query.get("task_name"),
                    query.get("image_id"),
                    query.get("group_id"),
                    query.get("prompt_id"),
                )
                cached = existing.get(key)
                if cached and (config.resume or cached.get("success")):
                    writer.write(json.dumps(cached, ensure_ascii=False) + "\n")
                    writer.flush()  # Real-time flush
                else:
                    record = _run_single_inference(
                        model, query, config.max_tokens, config.assistant_prefill
                    )
                    writer.write(json.dumps(record, ensure_ascii=False) + "\n")
                    writer.flush()  # Real-time flush
                if processed % interval == 0 or processed == total:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    remaining = (total - processed) / rate if rate > 0 else 0
                    logging.info(
                        "Processed %d/%d | ETA %s",
                        processed,
                        total,
                        _format_eta(remaining),
                    )

    logging.info("Finished inference for %d samples -> %s", processed, output_path)


def main() -> None:
    """CLI entry point."""

    args = parse_args()
    tasks = parse_tasks(args.tasks)
    # Use unified directory structure: runs/{model}/predictions.jsonl
    output_path = (
        args.output_path
        if args.output_path is not None
        else get_model_predictions_path(args.model_name)
    )
    config = InferenceConfig(
        model_name=args.model_name,
        queries_path=args.queries_path,
        output_path=output_path,
        tasks=tasks,
        max_samples=args.max_samples,
        seed=args.seed,
        shuffle=args.shuffle,
        overwrite=args.overwrite_output,
        max_tokens=args.max_tokens,
        log_level=args.log_level,
        vllm_mode=args.vllm_mode,
        vllm_port=args.vllm_port,
        assistant_prefill=args.assistant_prefill,
        max_model_len=args.max_model_len,
        max_image_size=args.max_image_size if args.max_image_size > 0 else None,
        resume=args.resume,
        resume_failed=args.resume_failed,
        api_batch_size=args.api_batch_size,
        api_batch_delay=args.api_batch_delay,
        progress_interval=args.progress_interval,
    )
    run_inference(config)


if __name__ == "__main__":
    main()
STOP_STRINGS = ["</s>", "```", "<|end_of_text|>"]
