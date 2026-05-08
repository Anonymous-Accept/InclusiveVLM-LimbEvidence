"""Parse COCO category names into canonical limb segments and kinds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from .config import SEGMENT_MAP

PROSTHESIS_SUBTYPE_TOKENS = {"articulated", "functional", "cosmetic"}


@dataclass(frozen=True)
class CategoryParseResult:
    """Structured result for a parsed category name."""

    segment: str
    kind: str  # "residual" or "prosthesis"


def _extract_side(tokens: List[str]) -> Optional[str]:
    """Return the limb side token if present."""

    for tok in tokens:
        if tok in {"left", "right"}:
            return tok
    return None


def _extract_segment_token(tokens: List[str]) -> Optional[str]:
    """Infer the segment token (upper-arm, forearm, thigh, calf) from tokens."""

    if "upper-arm" in tokens:
        return "upper-arm"
    for idx, tok in enumerate(tokens):
        if tok == "upper" and idx + 1 < len(tokens) and tokens[idx + 1] == "arm":
            return "upper-arm"
        if tok == "forearm":
            return "forearm"
        if tok == "fore" and idx + 1 < len(tokens) and tokens[idx + 1] == "arm":
            return "forearm"
        if tok == "thigh":
            return "thigh"
        if tok == "calf":
            return "calf"
    return None


def _extract_kind(tokens: List[str]) -> str:
    """Infer category kind from tokens following spec rules."""

    if tokens and tokens[-1] == "residual":
        return "residual"
    if (
        len(tokens) >= 2
        and tokens[-1] == "pro"
        and tokens[-2] in PROSTHESIS_SUBTYPE_TOKENS
    ):
        return "prosthesis"
    return "other"


def parse_category_name(category_name: str) -> Optional[CategoryParseResult]:
    """
    Parse a COCO category name into canonical segment and kind.

    Args:
        category_name: Raw category name from COCO categories.

    Returns:
        Parsed result if the name matches a residual or prosthetic limb; otherwise None.
    """

    tokens = [tok.strip().lower() for tok in category_name.split("-") if tok.strip()]
    if not tokens:
        return None

    side = _extract_side(tokens)
    segment_token = _extract_segment_token(tokens)
    kind = _extract_kind(tokens)
    if side is None or segment_token is None or kind == "other":
        return None

    segment = SEGMENT_MAP.get((side, segment_token))
    if segment is None:
        return None

    return CategoryParseResult(segment=segment, kind=kind)
