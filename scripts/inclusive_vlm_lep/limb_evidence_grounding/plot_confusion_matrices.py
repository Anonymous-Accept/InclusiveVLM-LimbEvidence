"""Plot Limb-Evidence Grounding confusion matrices from metrics JSON files."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from .config import PRESENCE_OPTION_DESCRIPTIONS

logger = logging.getLogger(__name__)

DEFAULT_COMBINE_EXCLUDES = {
    "gemini-2.5-flash-lite-preview",
    "gemini-2.5-pro",
    "gemini-3-pro",
}
FAMILY_ORDER = {
    "gpt": 0,
    "gemini": 1,
    "qwen3": 2,
    "gemma": 3,
    "deepseek": 4,
    "ministral": 5,
}
QWEN_SIZE_ORDER = {"30b": 0, "8b": 1, "4b": 2}
QWEN_VARIANT_ORDER = {"instruct": 0, "thinking": 1}
GEMMA_SIZE_ORDER = {"27b": 0, "12b": 1, "4b": 2}
DEEPSEEK_ORDER = {"vl2": 0, "vl2-small": 1, "vl2-tiny": 2}
MISTRAL_SIZE_ORDER = {"14b": 0, "8b": 1, "3b": 2}
MISTRAL_VARIANT_ORDER = {"reasoning": 0, "instruct": 1}


def configure_logging(level: str) -> None:
    """Configure console logging.

    Args:
        level: Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def resolve_default_metrics_dir() -> Path:
    """Select a default metrics directory based on existing outputs.

    Returns:
        Path to metrics directory.
    """
    return (
        Path("outputs")
        / "inclusive_vlm_lep"
        / "limb_evidence_grounding"
        / "metrics"
    )


def load_presence_confusion(
    metrics_path: Path,
) -> Tuple[np.ndarray, List[str], List[str]]:
    """Load presence confusion matrix from a metrics JSON file.

    Args:
        metrics_path: Path to metrics JSON.

    Returns:
        Tuple of (matrix, row label codes, column label codes).
    """
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    presence = data.get("presence", {})
    confusion = presence.get("confusion_matrix", {})
    labels = confusion.get("labels", [])
    matrix = confusion.get("matrix", [])
    if not labels or not matrix:
        return np.zeros((0, 0), dtype=float), [], []
    matrix_array = np.array(matrix, dtype=float)
    row_labels = list(labels)
    col_labels = list(labels)

    # Drop the "none" target row if it has no target samples.
    if "d" in row_labels:
        none_idx = row_labels.index("d")
        if matrix_array.shape[0] > none_idx and matrix_array[none_idx].sum() == 0:
            row_labels = [
                label for idx, label in enumerate(row_labels) if idx != none_idx
            ]
            matrix_array = np.delete(matrix_array, none_idx, axis=0)

    return matrix_array, row_labels, col_labels


def normalize_matrix(matrix: np.ndarray, mode: str) -> np.ndarray:
    """Normalize a matrix by row, column, or globally.

    Args:
        matrix: Raw confusion matrix.
        mode: One of "none", "row", "col".

    Returns:
        Normalized matrix (float).
    """
    if matrix.size == 0:
        return matrix
    if mode == "none":
        denom = matrix.sum()
        if denom == 0:
            return matrix
        return matrix / denom

    normalized = matrix.astype(float)
    if mode == "row":
        denom = normalized.sum(axis=1, keepdims=True)
    elif mode == "col":
        denom = normalized.sum(axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown normalization mode: {mode}")

    denom[denom == 0] = 1.0
    return normalized / denom


def apply_nature_style() -> None:
    """Apply a Nature-like plotting style."""
    preferred_fonts = ["Arial", "Liberation Sans", "DejaVu Sans"]
    font_family = _pick_font_family(preferred_fonts)
    plt.style.use("seaborn-v0_8-white")
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#333333",
            "xtick.major.size": 3,
            "xtick.major.width": 0.8,
            "ytick.major.size": 3,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.grid": False,
            "legend.frameon": False,
        }
    )


def _pick_font_family(candidates: List[str]) -> str:
    """Pick the first available font family."""
    for name in candidates:
        try:
            font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=False,
            )
        except Exception:
            continue
        return name
    return "DejaVu Sans"


def _presence_label_map(label_code: str) -> str:
    """Map label codes to display labels."""
    mapping = {
        "a": "Residual",
        "b": "Prosthetic",
        "c": "Residual+Prosthetic",
        "d": "None",
    }
    if label_code in mapping:
        return mapping[label_code]
    desc = PRESENCE_OPTION_DESCRIPTIONS.get(label_code)
    return desc if desc else label_code


def annotate_cells(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    font_size: int,
) -> None:
    """Annotate each cell with values.

    Args:
        ax: Matplotlib axes.
        matrix: Matrix values to annotate.
        font_size: Font size for annotations.
    """
    max_value = matrix.max() if matrix.size else 0.0
    threshold = max_value * 0.6 if max_value > 0 else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            label = f"{value:.2%}"
            color = "white" if value >= threshold else "black"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color=color,
                fontsize=font_size,
            )


def format_model_title(model_name: str) -> str:
    """Format model name for plot title."""
    if not model_name:
        return model_name
    parts = model_name.split("-")
    if parts and parts[-1].isdigit():
        parts = parts[:-1]
    formatted: List[str] = []
    for part in parts:
        if not part:
            formatted.append(part)
            continue
        formatted.append(part[:1].upper() + part[1:])
    name = "-".join(formatted)
    if name.startswith("Qwen3-Vl"):
        name = "Qwen3-VL" + name[len("Qwen3-Vl") :]
    elif name.startswith("Qwen3-VL"):
        name = "Qwen3-VL" + name[len("Qwen3-VL") :]
    if name.startswith("Deepseek-Vl2"):
        name = "Deepseek-VL2" + name[len("Deepseek-Vl2") :]
    if name.lower().startswith("qwen3-") and name.lower().endswith("-fp8"):
        name = name[:-4]
    name = name.replace("Gpt-", "GPT-")
    return name


def plot_confusion_matrix(
    matrix: np.ndarray,
    row_labels: List[str],
    col_labels: List[str],
    model_name: str,
    output_path: Path,
    *,
    normalize: str,
) -> None:
    """Plot a confusion matrix and save to disk."""
    apply_nature_style()
    fig, ax = plt.subplots(figsize=(7.8, 6.2))

    cmap = plt.get_cmap("YlGnBu")
    im = ax.imshow(matrix, cmap=cmap)

    ax.set_xticks(range(len(col_labels)), col_labels, rotation=50, ha="right")
    ax.set_yticks(
        range(len(row_labels)), row_labels, rotation=50, ha="right", va="center"
    )
    ax.set_xlabel("Predicted", fontsize=13)
    ax.set_ylabel("Target", fontsize=13)
    ax.set_title(format_model_title(model_name), pad=10, fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.set_aspect("equal")

    annotate_cells(ax, matrix, font_size=13)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Percent")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_combined_matrices(
    matrices: List[np.ndarray],
    row_labels: List[str],
    col_labels: List[str],
    model_names: List[str],
    output_path: Path,
    *,
    normalize: str,
) -> None:
    """Plot a combined grid of confusion matrices with a shared colorbar."""
    apply_nature_style()
    n_models = len(matrices)
    ncols = 4 if n_models >= 4 else max(1, n_models)
    nrows = int(np.ceil(n_models / ncols))

    fig_width = 4.2 * ncols
    fig_height = 3.6 * nrows
    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height), squeeze=False
    )

    cmap = plt.get_cmap("YlGnBu")
    im = None
    for idx, (matrix, model_name) in enumerate(zip(matrices, model_names)):
        row = idx // ncols
        col = idx % ncols
        ax = axes[row][col]
        im = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1.0)

        ax.set_xticks(range(len(col_labels)), col_labels, rotation=50, ha="right")
        ax.set_yticks(
            range(len(row_labels)),
            row_labels,
            rotation=50,
            ha="right",
            va="center",
        )
        ax.set_title(format_model_title(model_name))
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(axis="both", labelsize=9)
        ax.set_aspect("equal")
        annotate_cells(ax, matrix, font_size=9)

        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    for idx in range(n_models, nrows * ncols):
        row = idx // ncols
        col = idx % ncols
        axes[row][col].axis("off")

    fig.subplots_adjust(
        left=0.08, right=0.85, bottom=0.08, top=0.92, wspace=0.35, hspace=0.4
    )
    fig.supxlabel("Predicted", fontsize=11, y=0.03)
    fig.supylabel("Target", fontsize=11, x=0.02)
    if im is not None:
        cbar_ax = fig.add_axes([0.86, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Percent")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def build_output_name(model_name: str, normalize: str, suffix: str) -> str:
    """Build output filename for a model."""
    parts = []
    if normalize != "none":
        parts.append(normalize)
    if suffix:
        parts.append(suffix)
    joined = "_".join(parts)
    if joined:
        return f"{model_name}_presence_confusion_{joined}"
    return f"{model_name}_presence_confusion"


def iter_metrics_files(metrics_dir: Path) -> Iterable[Path]:
    """Yield metrics JSON files sorted by name."""
    return sorted(metrics_dir.glob("*.metrics.json"))


def _combined_sort_key(model_name: str) -> Tuple:
    """Sort key for combined plots to enforce family + size ordering."""
    parts = model_name.split("-") if model_name else []
    family = parts[0] if parts else ""
    family_rank = FAMILY_ORDER.get(family, 99)

    if model_name.startswith("qwen3-vl-"):
        size = parts[2] if len(parts) > 2 else ""
        variant = parts[3] if len(parts) > 3 else ""
        return (
            family_rank,
            QWEN_SIZE_ORDER.get(size, 99),
            QWEN_VARIANT_ORDER.get(variant, 99),
            model_name,
        )
    if model_name.startswith("gemma-3-"):
        size = parts[2] if len(parts) > 2 else ""
        return (family_rank, GEMMA_SIZE_ORDER.get(size, 99), model_name)
    if model_name.startswith("deepseek-vl2"):
        return (family_rank, DEEPSEEK_ORDER.get(model_name.replace("deepseek-", ""), 99), model_name)
    if model_name.startswith("ministral-3-"):
        size = parts[2] if len(parts) > 2 else ""
        variant = parts[3] if len(parts) > 3 else ""
        return (
            family_rank,
            MISTRAL_SIZE_ORDER.get(size, 99),
            MISTRAL_VARIANT_ORDER.get(variant, 99),
            model_name,
        )
    return (family_rank, model_name)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Plot Limb-Evidence Grounding presence confusion matrices."
    )
    parser.add_argument(
        "--metrics-dir",
        type=Path,
        default=resolve_default_metrics_dir(),
        help="Directory containing *.metrics.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write plots (default: metrics_dir/confusion_matrices).",
    )
    parser.add_argument(
        "--normalize",
        choices=["none", "row", "col"],
        default="none",
        help="Normalization mode for the confusion matrix (none=global).",
    )
    parser.add_argument(
        "--file-suffix",
        type=str,
        default="nature",
        help="Suffix appended to output filenames.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg", "pdf"],
        choices=["svg", "pdf", "png"],
        help="Output formats for plots (default: svg pdf).",
    )
    parser.add_argument(
        "--combine",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to write a combined multi-model plot.",
    )
    parser.add_argument(
        "--exclude-models",
        nargs="+",
        default=[],
        help="Model names to exclude from plotting (exact match).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the confusion matrix plotting pipeline."""
    args = parse_args()
    configure_logging(args.log_level)

    metrics_dir: Path = args.metrics_dir
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")

    output_dir = args.output_dir or (metrics_dir / "confusion_matrices")

    metrics_files = list(iter_metrics_files(metrics_dir))
    if not metrics_files:
        raise FileNotFoundError(f"No metrics files found in {metrics_dir}")

    excluded = {name.lower() for name in args.exclude_models}
    if excluded:
        filtered = []
        for path in metrics_files:
            model_name = path.stem.replace(".metrics", "")
            if model_name.lower() in excluded:
                logger.info("Skipping excluded model: %s", model_name)
                continue
            filtered.append(path)
        metrics_files = filtered

    formats: List[str] = []
    for fmt in args.formats:
        if fmt not in formats:
            formats.append(fmt)

    combined_entries: List[Tuple[str, np.ndarray]] = []
    display_row_labels: Optional[List[str]] = None
    display_col_labels: Optional[List[str]] = None

    for metrics_path in metrics_files:
        matrix, row_label_codes, col_label_codes = load_presence_confusion(metrics_path)
        if matrix.size == 0:
            logger.warning("Skipping %s (missing presence confusion)", metrics_path.name)
            continue

        matrix = normalize_matrix(matrix, args.normalize)
        row_labels = [_presence_label_map(code) for code in row_label_codes]
        col_labels = [_presence_label_map(code) for code in col_label_codes]

        model_name = metrics_path.stem.replace(".metrics", "")
        base_name = build_output_name(model_name, args.normalize, args.file_suffix)

        for ext in formats:
            out_path = output_dir / f"{base_name}.{ext}"
            plot_confusion_matrix(
                matrix,
                row_labels,
                col_labels,
                model_name,
                out_path,
                normalize=args.normalize,
            )
            logger.info("Wrote %s", out_path)

        combined_entries.append((model_name, matrix))
        display_row_labels = row_labels
        display_col_labels = col_labels

    if (
        args.combine
        and combined_entries
        and display_row_labels is not None
        and display_col_labels is not None
    ):
        excluded_combined = {name.lower() for name in DEFAULT_COMBINE_EXCLUDES}
        if excluded:
            excluded_combined.update(excluded)
        filtered_entries = [
            (name, matrix)
            for name, matrix in combined_entries
            if name.lower() not in excluded_combined
        ]
        filtered_entries = sorted(
            filtered_entries, key=lambda item: _combined_sort_key(item[0])
        )
        matrices = [matrix for _, matrix in filtered_entries]
        model_names = [name for name, _ in filtered_entries]
        if not matrices:
            logger.warning("No models left for combined plot after exclusions.")
            return
        combine_parts = [args.file_suffix] if args.file_suffix else []
        if args.normalize != "none":
            combine_parts.insert(0, args.normalize)
        suffix = "_".join(combine_parts)
        name = (
            f"presence_confusion_all_models_{suffix}"
            if suffix
            else "presence_confusion_all_models"
        )

        for ext in formats:
            combined_path = output_dir / f"{name}.{ext}"
            plot_combined_matrices(
                matrices,
                display_row_labels,
                display_col_labels,
                model_names,
                combined_path,
                normalize=args.normalize,
            )
            logger.info("Wrote %s", combined_path)


if __name__ == "__main__":
    main()
