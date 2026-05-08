"""High-level evaluation entrypoint for Limb-Evidence Grounding."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

from .eval_config import (
    DEFAULT_LOG_DIR,
    DEFAULT_METRICS_DIR,
    DEFAULT_QUERIES_PATH,
    DEFAULT_SEED,
    DEFAULT_TASKS,
    PROJECT_ROOT,
    get_model_log_path,
    get_model_metrics_path,
    get_model_predictions_path,
)
from .inference_runner import InferenceConfig, parse_tasks, run_inference
from .metrics_runner import run_metrics
from ..prosthesis_match.config import (
    DEFAULT_DATA_ITEMS_PATH as DEFAULT_PROSTHESIS_ITEMS_PATH,
)


def configure_logging(log_path: Path | None, level: str = "INFO") -> None:
    """Configure console and optional file logging."""

    handlers = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
        force=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Run inference + metrics for Limb-Evidence Grounding."
    )
    parser.add_argument("--model-name", required=True, help="Model key in registry.")
    parser.add_argument(
        "--queries-path",
        type=Path,
        default=DEFAULT_QUERIES_PATH,
        help="Path to queries.jsonl.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help="Optional results JSONL path. Defaults to work_dir/results/{model}.jsonl",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help="Optional metrics JSON path. Defaults to work_dir/metrics/{model}.metrics.json",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="all",
        help="Tasks to run (recognition,attribution,all).",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=-1,
        help="Limit number of samples for debug (-1 for all).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override model max tokens for generation.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Only for vLLM local mode: set max_model_len when loading the model.",
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
        "--run-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to run inference step.",
    )
    parser.add_argument(
        "--run-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to run metrics step.",
    )
    parser.add_argument(
        "--overwrite-results",
        action="store_true",
        help="Overwrite existing results JSONL when running inference.",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Optional log file path (default: work_dir/logs/eval_{model}.log).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level.",
    )
    parser.add_argument(
        "--bootstrap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=1000,
        help="Number of bootstrap samples (default: 1000).",
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        help="Confidence level for bootstrap CI (default: 0.95).",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=42,
        help="Random seed for bootstrap (default: 42).",
    )
    parser.add_argument(
        "--vllm-mode",
        type=str,
        choices=["server", "local"],
        default="server",
        help="vLLM mode: server (OpenAI-compatible) or local LLM.",
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
        help="Use vLLM assistant_prefill to constrain JSON answers (server mode).",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Optional .env file to load API keys (default: project .env).",
    )
    parser.add_argument(
        "--resume-failed",
        action="store_true",
        help="Reuse successful records and re-run only missing/failed ones.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing results (skip cached records).",
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
        "--prosthesis-results-path",
        type=Path,
        default=None,
        help="Optional Prosthesis Matching predictions JSONL for cross-benchmark analysis.",
    )
    parser.add_argument(
        "--prosthesis-items-path",
        type=Path,
        default=DEFAULT_PROSTHESIS_ITEMS_PATH,
        help="Path to Prosthesis Matching items JSONL (default: dataset items).",
    )
    parser.add_argument(
        "--correctness-strategy",
        type=str,
        default="majority",
        choices=["majority", "all", "any"],
        help="Image-level correctness aggregation strategy.",
    )
    return parser.parse_args()


def _resolve_flags(run_inference: bool | None, run_metrics: bool | None) -> tuple[bool, bool]:
    """Resolve booleans where None means unspecified."""

    if run_inference is None and run_metrics is None:
        return True, True
    return bool(run_inference), bool(run_metrics)


def _default_log_path(model_name: str) -> Path:
    return get_model_log_path(model_name)


def _default_output_layout(
    model_name: str, tasks: Set[str]
) -> tuple[Path, Path, Path, list[Path], list[Path]]:
    """Resolve the primary output paths and any secondary mirror targets."""

    if tasks == {"attribution"}:
        primary_task = "attribution"
        mirror_tasks: list[str] = []
    else:
        primary_task = "recognition"
        mirror_tasks = ["attribution"] if tasks == set(DEFAULT_TASKS) else []
    return (
        get_model_predictions_path(model_name, primary_task),
        get_model_metrics_path(model_name, primary_task),
        get_model_log_path(model_name, primary_task),
        [get_model_predictions_path(model_name, task_name) for task_name in mirror_tasks],
        [get_model_metrics_path(model_name, task_name) for task_name in mirror_tasks],
    )


def _sync_output_artifact(src: Path, dst: Path) -> None:
    """Mirror one output file into a secondary task root."""

    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        try:
            if os.path.samefile(src, dst):
                return
        except FileNotFoundError:
            pass
        dst.unlink()
    src_mount = Path(os.path.dirname(src)).stat().st_dev
    dst_mount = Path(os.path.dirname(dst)).stat().st_dev
    if src_mount == dst_mount:
        os.link(src, dst)
    else:
        shutil.copy2(src, dst)


def main() -> None:
    """CLI entry point for one-click evaluation."""

    args = parse_args()
    load_dotenv(args.env_file)
    run_inference_flag, run_metrics_flag = _resolve_flags(
        args.run_inference, args.run_metrics
    )
    if not run_inference_flag and not run_metrics_flag:
        raise ValueError("At least one of --run-inference/--run-metrics must be true.")

    tasks: Set[str] = parse_tasks(args.tasks)
    (
        default_results_path,
        default_metrics_path,
        default_log_path,
        mirror_results_paths,
        mirror_metrics_paths,
    ) = _default_output_layout(args.model_name, tasks)
    results_path = (
        args.results_path
        if args.results_path is not None
        else default_results_path
    )
    metrics_path = (
        args.metrics_path
        if args.metrics_path is not None
        else default_metrics_path
    )
    log_path = args.log_path or default_log_path

    configure_logging(log_path, args.log_level)
    logging.info(
        "Evaluation start (model=%s tasks=%s run_inference=%s run_metrics=%s)",
        args.model_name,
        ",".join(sorted(tasks)) if tasks else "all",
        run_inference_flag,
        run_metrics_flag,
    )

    if run_inference_flag:
        if results_path.exists() and not (
            args.overwrite_results or args.resume_failed or args.resume
        ):
            logging.warning(
                "Results file %s exists; skipping inference (use --overwrite-results to rerun or --resume/--resume-failed to append missing).",
                results_path,
            )
        else:
            inference_config = InferenceConfig(
                model_name=args.model_name,
                queries_path=args.queries_path,
                output_path=results_path,
                tasks=tasks or set(DEFAULT_TASKS),
                max_samples=args.max_samples,
                seed=args.seed,
                shuffle=args.shuffle,
                overwrite=args.overwrite_results,
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
            if results_path.exists() and (
                args.resume or args.resume_failed
            ) and not args.overwrite_results:
                if args.resume_failed and not args.resume:
                    logging.info(
                        "Results file %s exists; running in resume-failed mode (reuse successes, fill missing).",
                        results_path,
                    )
                else:
                    logging.info(
                        "Results file %s exists; resuming inference (reuse cached records).",
                        results_path,
                    )
            run_inference(inference_config)
    if args.results_path is None and results_path.exists():
        for mirror_path in mirror_results_paths:
            _sync_output_artifact(results_path, mirror_path)

    if run_metrics_flag:
        if not results_path.exists():
            raise FileNotFoundError(
                f"Results file {results_path} not found. Run inference first."
            )
        run_metrics(
            results_path=results_path,
            output_path=metrics_path,
            queries_path=args.queries_path,
            log_level=args.log_level,
            compute_bootstrap=args.bootstrap,
            bootstrap_samples=args.bootstrap_samples,
            confidence_level=args.confidence_level,
            bootstrap_seed=args.bootstrap_seed,
            prosthesis_results_path=args.prosthesis_results_path,
            prosthesis_items_path=args.prosthesis_items_path,
            correctness_strategy=args.correctness_strategy,
        )
    if args.metrics_path is None and metrics_path.exists():
        for mirror_path in mirror_metrics_paths:
            _sync_output_artifact(metrics_path, mirror_path)

    logging.info("Evaluation complete.")


if __name__ == "__main__":
    main()
