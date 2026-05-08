"""Build query records for Limb-Evidence Grounding tasks."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .config import (
    BENCHMARK_NAME,
    BENCHMARK_PART,
    LABEL_VOCAB,
    PRESENCE_LABEL_TO_OPTION,
    PRESENCE_OPTION_DESCRIPTIONS,
    PROJECT_ROOT,
)
from .label_extraction import ImageLabels


def _ensure_prompt_budget(
    prompts: Sequence[str], requested: int, task_name: str
) -> None:
    """Validate that we can sample without replacement."""

    if requested > len(prompts):
        raise ValueError(
            f"Requested {requested} {task_name} prompts but only {len(prompts)} "
            f"available."
        )


def _sample_prompts(
    prompts: Sequence[str], requested: int, task_name: str
) -> List[str]:
    """Sample prompts without replacement."""

    _ensure_prompt_budget(prompts, requested, task_name)
    return random.sample(list(prompts), requested)


def _build_image_path(image_root: Path, file_name: str) -> str:
    """Return image path string, relative to project root when possible."""

    full_path = image_root / file_name
    try:
        return full_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return full_path.as_posix()


def build_presence_records(
    image: ImageLabels,
    prompts: Sequence[str],
    num_prompts: int,
    image_root: Path,
    include_none_class: bool,
) -> List[Dict[str, Any]]:
    """Construct presence task records for an image."""

    if image.presence_label == "none" and not include_none_class:
        return []

    prompt_samples = _sample_prompts(prompts, num_prompts, "presence")
    target_answer = PRESENCE_LABEL_TO_OPTION[image.presence_label]
    image_path = _build_image_path(image_root, image.file_name)

    records: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompt_samples):
        records.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_part": BENCHMARK_PART,
                "task_name": "recognition",
                "image_id": image.image_id,
                "file_name": image.file_name,
                "image_path": image_path,
                "group_id": f"presence_img{image.image_id}",
                "prompt_id": idx,
                "prompt": prompt,
                "options": ["a", "b", "c", "d"],
                "option_descriptions": dict(PRESENCE_OPTION_DESCRIPTIONS),
                "target_answer": target_answer,
            }
        )
    return records


def build_segment_records(
    image: ImageLabels,
    prompts: Sequence[str],
    num_prompts: int,
    image_root: Path,
    keep_empty_segments: bool,
) -> List[Dict[str, Any]]:
    """Construct segment task records for an image."""

    target_labels = image.build_target_labels(keep_empty_segments)
    if not target_labels:
        return []

    prompt_samples = _sample_prompts(prompts, num_prompts, "segments")
    image_path = _build_image_path(image_root, image.file_name)

    records: List[Dict[str, Any]] = []
    for idx, prompt in enumerate(prompt_samples):
        records.append(
            {
                "benchmark_name": BENCHMARK_NAME,
                "benchmark_part": BENCHMARK_PART,
                "task_name": "attribution",
                "image_id": image.image_id,
                "file_name": image.file_name,
                "image_path": image_path,
                "group_id": f"segments_img{image.image_id}",
                "prompt_id": idx,
                "prompt": prompt,
                "label_vocab": list(LABEL_VOCAB),
                "target_labels": target_labels,
                "target_answer": None,
            }
        )
    return records


def build_queries(
    images: Iterable[ImageLabels],
    presence_prompts: Sequence[str],
    segment_prompts: Sequence[str],
    num_presence_prompts: int,
    num_segment_prompts: int,
    image_root: Path,
    include_none_class: bool,
    keep_empty_segments: bool,
) -> List[Dict[str, Any]]:
    """Generate all presence and segment queries."""

    queries: List[Dict[str, Any]] = []
    for image in images:
        queries.extend(
            build_presence_records(
                image,
                presence_prompts,
                num_presence_prompts,
                image_root,
                include_none_class,
            )
        )
        queries.extend(
            build_segment_records(
                image,
                segment_prompts,
                num_segment_prompts,
                image_root,
                keep_empty_segments,
            )
        )
    return queries
