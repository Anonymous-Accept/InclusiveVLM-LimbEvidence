"""Prompt loading helpers for Limb-Evidence Grounding."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List


def _load_prompts(json_path: Path, key: str) -> List[str]:
    """Load prompt list from a JSON file under the given key."""

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    prompts = data.get(key, [])
    if not isinstance(prompts, list):
        raise ValueError(f"Expected list under key '{key}' in {json_path}")
    if not prompts:
        raise ValueError(f"No prompts found under key '{key}' in {json_path}")
    return [str(p) for p in prompts]


def load_presence_prompts(json_path: Path) -> List[str]:
    """Load presence paraphrases."""

    return _load_prompts(json_path, "presence_paraphrases")


def load_segment_prompts(json_path: Path) -> List[str]:
    """Load segment paraphrases."""

    return _load_prompts(json_path, "segment_paraphrases")
