"""Configuration for Prosthesis Matching."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_DATASET_ROOT",
    "DEFAULT_DATA_PROSTHESIS_MATCH_DIR",
    "DEFAULT_ANNO_PATH",
    "DEFAULT_IMAGE_ROOT",
    "DEFAULT_DATA_ITEMS_PATH",
    "DEFAULT_DATA_GRIDS_DIR",
    "DEFAULT_DATA_OPTIONS_DIR",
    "DEFAULT_WORK_DIR",
    "DEFAULT_OPTIONS_DIR",
    "DEFAULT_ITEMS_PATH",
    "DEFAULT_GRIDS_DIR",
    "DEFAULT_RUNS_DIR",
    "DEFAULT_METRICS_DIR",
    "DEFAULT_LOG_DIR",
    "DEFAULT_SEED",
    "BENCHMARK_NAME",
    "BENCHMARK_PART",
    "PART8_LABELS",
    "COARSE_PARTS",
    "PART8_TO_COARSE",
    "PROSTHESIS_CATEGORY_IDS",
    "PROSTHESIS_CATEGORY_TO_PART8",
    "LIMB_CATEGORY_IDS",
    "LIMB_CATEGORY_TO_PART8",
    "RESIDUAL_CATEGORY_IDS",
    "RESIDUAL_CATEGORY_TO_PART8",
    "ALLOWED_CHOICES",
    "DEFAULT_K_OPTIONS",
    "OPTION_POOL_MIN_AREA",
    "OPTION_POOL_PAD_PX",
    "OPTION_POOL_OUT_SIZE",
    "GRID_COLS",
    "HARD_NEGATIVE_RATIO",
]

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data paths: default to InclusiveVLM-LEP for runtime
# (Generated data still goes to outputs)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "inclusive_vlm_lep"
DEFAULT_DATA_PROSTHESIS_MATCH_DIR = DEFAULT_DATASET_ROOT / "prosthesis_match"
DEFAULT_ANNO_PATH = DEFAULT_DATASET_ROOT / "annotations" / "full_dataset.json"
DEFAULT_IMAGE_ROOT = DEFAULT_DATASET_ROOT / "images"
DEFAULT_DATA_ITEMS_PATH = (
    DEFAULT_DATA_PROSTHESIS_MATCH_DIR / "items" / "prosthesis_match.jsonl"
)
DEFAULT_DATA_GRIDS_DIR = DEFAULT_DATA_PROSTHESIS_MATCH_DIR / "grids"
DEFAULT_DATA_OPTIONS_DIR = DEFAULT_DATA_PROSTHESIS_MATCH_DIR / "options"

# Work directories for generated artifacts
DEFAULT_WORK_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "prosthesis_matching"
)
DEFAULT_OPTIONS_DIR = DEFAULT_WORK_DIR / "options"
DEFAULT_ITEMS_PATH = DEFAULT_WORK_DIR / "items" / "prosthesis_match.jsonl"
DEFAULT_GRIDS_DIR = DEFAULT_WORK_DIR / "grids"
DEFAULT_RUNS_DIR = DEFAULT_WORK_DIR / "runs"
DEFAULT_METRICS_DIR = DEFAULT_WORK_DIR / "metrics"
DEFAULT_LOG_DIR = DEFAULT_WORK_DIR / "logs"

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_SEED = 42
BENCHMARK_NAME = "Prosthesis Matching"
BENCHMARK_PART = "prosthesis_match"

# ---------------------------------------------------------------------------
# Label definitions (8 canonical parts)
# ---------------------------------------------------------------------------

PART8_LABELS: List[str] = [
    "left_arm_upper",
    "left_arm_lower",
    "right_arm_upper",
    "right_arm_lower",
    "left_leg_upper",
    "left_leg_lower",
    "right_leg_upper",
    "right_leg_lower",
]

# Coarse parts (ignore laterality)
COARSE_PARTS: List[str] = [
    "arm_upper",
    "arm_lower",
    "leg_upper",
    "leg_lower",
]

# Map part8 to coarse parts
PART8_TO_COARSE: Dict[str, str] = {
    "left_arm_upper": "arm_upper",
    "right_arm_upper": "arm_upper",
    "left_arm_lower": "arm_lower",
    "right_arm_lower": "arm_lower",
    "left_leg_upper": "leg_upper",
    "right_leg_upper": "leg_upper",
    "left_leg_lower": "leg_lower",
    "right_leg_lower": "leg_lower",
}

# ---------------------------------------------------------------------------
# Category ID mappings (from InclusiveVLM-LEP dataset)
# ---------------------------------------------------------------------------

# Prosthesis category IDs (all types: articulated, functional, cosmetic)
# Format: {category_id: category_name}
PROSTHESIS_CATEGORY_IDS: Dict[int, str] = {
    # Left upper arm prosthesis
    30: "left-upper-arm-articulated-pro",
    32: "left-upper-arm-functional-pro",
    34: "left-upper-arm-cosmetic-pro",
    # Right upper arm prosthesis
    31: "right-upper-arm-articulated-pro",
    33: "right-upper-arm-functional-pro",
    35: "right-upper-arm-cosmetic-pro",
    # Left forearm prosthesis
    36: "left-forearm-articulated-pro",
    38: "left-forearm-functional-pro",
    40: "left-forearm-cosmetic-pro",
    # Right forearm prosthesis
    37: "right-forearm-articulated-pro",
    39: "right-forearm-functional-pro",
    41: "right-forearm-cosmetic-pro",
    # Left thigh prosthesis
    42: "left-thigh-articulated-pro",
    44: "left-thigh-functional-pro",
    46: "left-thigh-cosmetic-pro",
    # Right thigh prosthesis
    43: "right-thigh-articulated-pro",
    45: "right-thigh-functional-pro",
    47: "right-thigh-cosmetic-pro",
    # Left calf prosthesis
    48: "left-calf-articulated-pro",
    50: "left-calf-functional-pro",
    52: "left-calf-cosmetic-pro",
    # Right calf prosthesis
    49: "right-calf-articulated-pro",
    51: "right-calf-functional-pro",
    53: "right-calf-cosmetic-pro",
}

# Map prosthesis category ID to part8 label
PROSTHESIS_CATEGORY_TO_PART8: Dict[int, str] = {
    # Left upper arm -> left_arm_upper
    30: "left_arm_upper",
    32: "left_arm_upper",
    34: "left_arm_upper",
    # Right upper arm -> right_arm_upper
    31: "right_arm_upper",
    33: "right_arm_upper",
    35: "right_arm_upper",
    # Left forearm -> left_arm_lower
    36: "left_arm_lower",
    38: "left_arm_lower",
    40: "left_arm_lower",
    # Right forearm -> right_arm_lower
    37: "right_arm_lower",
    39: "right_arm_lower",
    41: "right_arm_lower",
    # Left thigh -> left_leg_upper
    42: "left_leg_upper",
    44: "left_leg_upper",
    46: "left_leg_upper",
    # Right thigh -> right_leg_upper
    43: "right_leg_upper",
    45: "right_leg_upper",
    47: "right_leg_upper",
    # Left calf -> left_leg_lower
    48: "left_leg_lower",
    50: "left_leg_lower",
    52: "left_leg_lower",
    # Right calf -> right_leg_lower
    49: "right_leg_lower",
    51: "right_leg_lower",
    53: "right_leg_lower",
}

# Limb category IDs (natural limb parts)
LIMB_CATEGORY_IDS: Dict[int, str] = {
    8: "left-upper-arm",
    9: "right-upper-arm",
    10: "left-forearm",
    11: "right-forearm",
    19: "left-thigh",
    20: "right-thigh",
    21: "left-calf",
    22: "right-calf",
}

LIMB_CATEGORY_TO_PART8: Dict[int, str] = {
    8: "left_arm_upper",
    9: "right_arm_upper",
    10: "left_arm_lower",
    11: "right_arm_lower",
    19: "left_leg_upper",
    20: "right_leg_upper",
    21: "left_leg_lower",
    22: "right_leg_lower",
}

# Residual limb category IDs
RESIDUAL_CATEGORY_IDS: Dict[int, str] = {
    54: "left-upper-arm-residual",
    55: "right-upper-arm-residual",
    56: "left-forearm-residual",
    57: "right-forearm-residual",
    58: "left-thigh-residual",
    59: "right-thigh-residual",
    60: "left-calf-residual",
    61: "right-calf-residual",
}

RESIDUAL_CATEGORY_TO_PART8: Dict[int, str] = {
    54: "left_arm_upper",
    55: "right_arm_upper",
    56: "left_arm_lower",
    57: "right_arm_lower",
    58: "left_leg_upper",
    59: "right_leg_upper",
    60: "left_leg_lower",
    61: "right_leg_lower",
}

# ---------------------------------------------------------------------------
# Benchmark options
# ---------------------------------------------------------------------------

ALLOWED_CHOICES: List[str] = ["A", "B", "C", "D", "E", "F", "G", "H"]
DEFAULT_K_OPTIONS = 8

# Option pool construction parameters
OPTION_POOL_MIN_AREA = 400
OPTION_POOL_PAD_PX = 10
OPTION_POOL_OUT_SIZE = 336

# Grid rendering
GRID_COLS = 4

# Negative sampling
HARD_NEGATIVE_RATIO = 0.5


def get_all_prosthesis_category_ids() -> Set[int]:
    """Return all prosthesis category IDs."""
    return set(PROSTHESIS_CATEGORY_IDS.keys())


def get_all_residual_category_ids() -> Set[int]:
    """Return all residual limb category IDs."""
    return set(RESIDUAL_CATEGORY_IDS.keys())


def get_coarse_part(part8: str) -> str:
    """Map part8 label to coarse part.

    Args:
        part8: One of the 8 canonical part labels.

    Returns:
        Coarse part label (ignoring laterality).

    Raises:
        KeyError: If part8 is not a valid label.
    """
    return PART8_TO_COARSE[part8]


def get_hard_negative_parts(coarse_part: str) -> List[str]:
    """Get hard negative coarse parts (same limb, different segment).

    Args:
        coarse_part: A coarse part label.

    Returns:
        List of hard negative coarse parts.
    """
    hard_negatives: Dict[str, List[str]] = {
        "arm_upper": ["arm_lower"],
        "arm_lower": ["arm_upper"],
        "leg_upper": ["leg_lower"],
        "leg_lower": ["leg_upper"],
    }
    return hard_negatives.get(coarse_part, [])


def get_cross_limb_parts(coarse_part: str) -> List[str]:
    """Get cross-limb negative coarse parts.

    Args:
        coarse_part: A coarse part label.

    Returns:
        List of cross-limb coarse parts.
    """
    if coarse_part.startswith("arm"):
        return ["leg_upper", "leg_lower"]
    else:
        return ["arm_upper", "arm_lower"]
