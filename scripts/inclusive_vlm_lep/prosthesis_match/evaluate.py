"""High-level evaluation entrypoint for Prosthesis Matching."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .config import (
    DEFAULT_DATA_ITEMS_PATH,
    DEFAULT_SEED,
    DEFAULT_WORK_DIR,
    PROJECT_ROOT,
)
from .eval_prosthesis_match import evaluate_prosthesis_match
from .run_inference import InferenceConfig, run_inference

__all__ = ["main", "run_prosthesis_match_evaluation"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default paths (model-specific directories)
# ---------------------------------------------------------------------------

DEFAULT_RUNS_DIR = DEFAULT_WORK_DIR / "runs"
DEFAULT_METRICS_DIR = DEFAULT_WORK_DIR / "metrics"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"


def infer_variant_name(items_path: Path) -> str:
    """Infer the public Prosthesis Matching variant from the items file name."""

    name = items_path.name.lower()
    if "compatibility_category" in name:
        return "compatibility_category"
    return "compatibility_diversity"


def get_model_run_dir(model_name: str, variant: str) -> Path:
    """Get model- and variant-specific run directory."""
    return DEFAULT_RUNS_DIR / model_name / variant


def get_model_metrics_path(model_name: str, variant: str) -> Path:
    """Get model- and variant-specific metrics file path."""
    return DEFAULT_METRICS_DIR / model_name / f"{variant}.metrics.json"


def get_model_log_path(model_name: str, variant: str) -> Path:
    """Get model- and variant-specific log file path."""
    return DEFAULT_LOG_DIR / model_name / f"eval_{variant}.log"


def configure_logging(log_path: Optional[Path], level: str = "INFO") -> None:
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
        description="Run inference + metrics for Prosthesis Matching."
    )

    # Model configuration
    parser.add_argument(
        "--model-name",
        required=True,
        help="Model key in registry (e.g., qwen3-vl-30b-a3b-thinking-fp8).",
    )

    # Input paths
    parser.add_argument(
        "--items-path",
        type=Path,
        default=DEFAULT_DATA_ITEMS_PATH,
        help=f"Path to benchmark items JSONL. Default: {DEFAULT_DATA_ITEMS_PATH}",
    )

    # Output paths (auto-generated based on model name if not specified)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory for predictions. Default: outputs/.../runs/{model_name}/{variant}",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=None,
        help=(
            "Metrics JSON output path. Default:"
            " outputs/.../metrics/{model_name}/{variant}.metrics.json"
        ),
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=None,
        help="Log file path. Default: outputs/.../logs/{model_name}/eval_{variant}.log",
    )

    # Execution control
    parser.add_argument(
        "--run-inference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to run inference step. Default: True unless --no-run-inference.",
    )
    parser.add_argument(
        "--run-metrics",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Whether to run metrics step. Default: True unless --no-run-metrics.",
    )

    # Resume / Overwrite
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing predictions (skip completed items).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing predictions.",
    )

    # Inference settings
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples for debug.",
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
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )

    # Model backend settings
    parser.add_argument(
        "--vllm-mode",
        type=str,
        choices=["server", "local"],
        default="server",
        help="vLLM mode: server (OpenAI-compatible) or local. Default: server",
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
        help="L2: Enable assistant prefill for JSON output constraint.",
    )
    parser.add_argument(
        "--guided-decoding",
        action="store_true",
        help="L3: Enable guided decoding for choice constraint.",
    )

    # Batch processing
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

    # Grid mode
    parser.add_argument(
        "--use-grid",
        action="store_true",
        default=True,
        help="Use grid images for options. Default: True",
    )
    parser.add_argument(
        "--no-grid",
        action="store_false",
        dest="use_grid",
        help="Use individual option images instead of grid.",
    )

    # Logging
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level. Default: INFO",
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

    # Environment
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="Optional .env file to load API keys.",
    )

    return parser.parse_args()


def _resolve_flags(
    run_inference: Optional[bool],
    run_metrics: Optional[bool],
) -> tuple[bool, bool]:
    """Resolve execution flags where None means unspecified."""
    if run_inference is None and run_metrics is None:
        return True, True
    return bool(run_inference), bool(run_metrics)


def run_prosthesis_match_evaluation(
    model_name: str,
    items_path: Path = DEFAULT_DATA_ITEMS_PATH,
    run_dir: Optional[Path] = None,
    metrics_path: Optional[Path] = None,
    log_path: Optional[Path] = None,
    run_inference_flag: bool = True,
    run_metrics_flag: bool = True,
    resume: bool = False,
    overwrite: bool = False,
    max_samples: Optional[int] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: int = DEFAULT_SEED,
    vllm_mode: str = "server",
    vllm_port: int | None = None,
    assistant_prefill: bool = False,
    guided_decoding: bool = False,
    use_grid: bool = True,
    log_level: str = "INFO",
    compute_bootstrap: bool = True,
    bootstrap_samples: int = 1000,
    confidence_level: float = 0.95,
    bootstrap_seed: int = 42,
    batch_size: int = 1,
    max_concurrent: int = 8,
) -> Path:
    """Run Prosthesis Matching evaluation pipeline.

    Args:
        model_name: Model key in registry.
        items_path: Path to benchmark items JSONL.
        run_dir: Run directory for predictions.
        metrics_path: Metrics JSON output path.
        log_path: Log file path.
        run_inference_flag: Whether to run inference.
        run_metrics_flag: Whether to run metrics.
        resume: Resume from existing predictions.
        overwrite: Overwrite existing predictions.
        max_samples: Limit number of samples.
        max_tokens: Maximum tokens for output.
        temperature: Sampling temperature.
        seed: Random seed.
        vllm_mode: vLLM mode (server/local).
        vllm_port: Override vLLM server port (server mode only).
        assistant_prefill: Enable L2 assistant prefill.
        guided_decoding: Enable L3 guided decoding.
        use_grid: Use grid images for options.
        log_level: Logging level.
        compute_bootstrap: Whether to compute bootstrap confidence intervals.
        bootstrap_samples: Number of bootstrap samples.
        confidence_level: Confidence level for bootstrap CI.
        bootstrap_seed: Random seed for bootstrap.
        batch_size: Batch size for parallel inference.
        max_concurrent: Max concurrent requests for batch.

    Returns:
        Path to metrics file.
    """
    # Resolve paths based on model name and public benchmark variant.
    variant = infer_variant_name(items_path)
    if run_dir is None:
        run_dir = get_model_run_dir(model_name, variant)
    if metrics_path is None:
        metrics_path = get_model_metrics_path(model_name, variant)
    if log_path is None:
        log_path = get_model_log_path(model_name, variant)

    # Ensure directories exist
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Predictions path
    predictions_path = run_dir / "predictions.jsonl"

    # Configure logging
    configure_logging(log_path, log_level)

    logger.info("=" * 70)
    logger.info("Prosthesis Matching Evaluation Pipeline")
    logger.info("=" * 70)
    logger.info(f"Model: {model_name}")
    logger.info(f"Items path: {items_path}")
    logger.info(f"Run directory: {run_dir}")
    logger.info(f"Metrics path: {metrics_path}")
    logger.info(f"Run inference: {run_inference_flag}")
    logger.info(f"Run metrics: {run_metrics_flag}")
    logger.info(f"Resume: {resume}")
    logger.info(f"Overwrite: {overwrite}")
    logger.info("=" * 70)

    # Step 1: Inference
    if run_inference_flag:
        if predictions_path.exists() and not (resume or overwrite):
            logger.warning(
                f"Predictions file {predictions_path} exists. "
                "Use --resume to continue or --overwrite to replace."
            )
        else:
            logger.info("Starting inference...")

            inference_config = InferenceConfig(
                model_name=model_name,
                items_path=items_path,
                output_dir=run_dir,
                max_samples=max_samples,
                seed=seed,
                use_grid=use_grid,
                max_tokens=max_tokens,
                temperature=temperature,
                vllm_mode=vllm_mode,
                vllm_port=vllm_port,
                resume=resume,
                overwrite=overwrite,
                log_level=log_level,
                assistant_prefill=assistant_prefill,
                guided_decoding=guided_decoding,
                batch_size=batch_size,
                max_concurrent=max_concurrent,
            )

            run_inference(inference_config)
            logger.info("Inference completed.")

    # Step 2: Metrics
    if run_metrics_flag:
        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Predictions file {predictions_path} not found. "
                "Run inference first or check --run-dir path."
            )

        logger.info("Computing metrics...")
        evaluate_prosthesis_match(
            predictions_path=predictions_path,
            items_path=items_path,
            output_path=metrics_path,
            compute_bootstrap=compute_bootstrap,
            bootstrap_samples=bootstrap_samples,
            confidence_level=confidence_level,
            bootstrap_seed=bootstrap_seed,
        )
        logger.info(f"Metrics saved to {metrics_path}")

    logger.info("=" * 70)
    logger.info("Evaluation complete!")
    logger.info("=" * 70)

    return metrics_path


def main() -> None:
    """CLI entry point for one-click Prosthesis Matching evaluation."""
    args = parse_args()

    # Load environment variables
    if args.env_file.exists():
        load_dotenv(args.env_file)

    # Resolve execution flags
    run_inference_flag, run_metrics_flag = _resolve_flags(
        args.run_inference, args.run_metrics
    )

    if not run_inference_flag and not run_metrics_flag:
        raise ValueError(
            "At least one of --run-inference or --run-metrics must be enabled."
        )

    # Run evaluation
    run_prosthesis_match_evaluation(
        model_name=args.model_name,
        items_path=args.items_path,
        run_dir=args.run_dir,
        metrics_path=args.metrics_path,
        log_path=args.log_path,
        run_inference_flag=run_inference_flag,
        run_metrics_flag=run_metrics_flag,
        resume=args.resume,
        overwrite=args.overwrite,
        max_samples=args.max_samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=args.seed,
        vllm_mode=args.vllm_mode,
        vllm_port=args.vllm_port,
        assistant_prefill=args.assistant_prefill,
        guided_decoding=args.guided_decoding,
        use_grid=args.use_grid,
        log_level=args.log_level,
        compute_bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples,
        confidence_level=args.confidence_level,
        bootstrap_seed=args.bootstrap_seed,
        batch_size=args.batch_size,
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
