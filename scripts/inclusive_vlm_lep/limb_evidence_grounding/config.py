"""Configuration and shared constants for Limb-Evidence Grounding."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

__all__ = [
    "PROJECT_ROOT",
    "DEFAULT_ANNO_PATH",
    "DEFAULT_IMAGE_ROOT",
    "DEFAULT_WORK_DIR",
    "DEFAULT_QUERIES_PATH",
    "DEFAULT_PRESENCE_PROMPTS_PATH",
    "DEFAULT_SEGMENT_PROMPTS_PATH",
    "DEFAULT_NUM_PRESENCE_PROMPTS",
    "DEFAULT_NUM_SEGMENT_PROMPTS",
    "DEFAULT_INCLUDE_NONE_CLASS",
    "DEFAULT_INCLUDE_EMPTY_SEGMENT_IMAGES",
    "DEFAULT_SEED",
    "BENCHMARK_NAME",
    "BENCHMARK_PART",
    "ATTRIBUTION",
    "SEGMENT_MAP",
    "LABEL_VOCAB",
    "PRESENCE_LABEL_TO_OPTION",
    "PRESENCE_OPTION_DESCRIPTIONS",
]

# ---------------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Data paths: default to InclusiveVLM-LEP for runtime
# (Generated data still goes to outputs)
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "inclusive_vlm_lep"
DEFAULT_ANNO_PATH = DEFAULT_DATASET_ROOT / "annotations" / "full_dataset.json"
DEFAULT_IMAGE_ROOT = DEFAULT_DATASET_ROOT / "images"

# Work directories for generated artifacts
DEFAULT_WORK_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
)
DEFAULT_QUERIES_PATH = (
    DEFAULT_DATASET_ROOT / "limb_evidence_grounding" / "queries_v2.jsonl"
)
DEFAULT_PRESENCE_PROMPTS_PATH = PROJECT_ROOT / "configs" / "presence_paraphrases.json"
DEFAULT_SEGMENT_PROMPTS_PATH = PROJECT_ROOT / "configs" / "segment_paraphrases.json"

# ---------------------------------------------------------------------------
# Default settings
# ---------------------------------------------------------------------------

DEFAULT_NUM_PRESENCE_PROMPTS = 3
DEFAULT_NUM_SEGMENT_PROMPTS = 3
DEFAULT_INCLUDE_NONE_CLASS = False
DEFAULT_INCLUDE_EMPTY_SEGMENT_IMAGES = False
DEFAULT_SEED = 42

BENCHMARK_NAME = "InclusiveVLM_LEP"
BENCHMARK_PART = "limb_evidence_grounding"

# ---------------------------------------------------------------------------
# Body segment definitions
# ---------------------------------------------------------------------------

ATTRIBUTION: List[str] = [
    "upper_arm_l",
    "upper_arm_r",
    "forearm_l",
    "forearm_r",
    "thigh_l",
    "thigh_r",
    "calf_l",
    "calf_r",
]

SEGMENT_MAP: Dict[tuple[str, str], str] = {
    ("left", "upper-arm"): "upper_arm_l",
    ("right", "upper-arm"): "upper_arm_r",
    ("left", "forearm"): "forearm_l",
    ("right", "forearm"): "forearm_r",
    ("left", "thigh"): "thigh_l",
    ("right", "thigh"): "thigh_r",
    ("left", "calf"): "calf_l",
    ("right", "calf"): "calf_r",
}

LABEL_VOCAB: List[str] = [
    "upper_arm_l_residual",
    "upper_arm_r_residual",
    "forearm_l_residual",
    "forearm_r_residual",
    "thigh_l_residual",
    "thigh_r_residual",
    "calf_l_residual",
    "calf_r_residual",
    "upper_arm_l_prosthesis",
    "upper_arm_r_prosthesis",
    "forearm_l_prosthesis",
    "forearm_r_prosthesis",
    "thigh_l_prosthesis",
    "thigh_r_prosthesis",
    "calf_l_prosthesis",
    "calf_r_prosthesis",
    "none",
]

# ---------------------------------------------------------------------------
# Presence detection options
# ---------------------------------------------------------------------------

PRESENCE_LABEL_TO_OPTION: Dict[str, str] = {
    "residual_only": "a",
    "prosthesis_only": "b",
    "residual_and_prosthesis": "c",
    "none": "d",
}

PRESENCE_OPTION_DESCRIPTIONS: Dict[str, str] = {
    "a": "Residual limb only",
    "b": "Prosthetic limb only",
    "c": "Both residual and prosthetic limbs",
    "d": "No visible residual limb or prosthetic limb",
}
