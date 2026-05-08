"""Normalize Attribution-FreeForm outputs under the frozen parser policy."""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.core.io_utils import load_yaml

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ALIAS_CONFIG = (
    PROJECT_ROOT / "configs" / "inclusive_vlm_lep" / "attribution_freeform_aliases.yaml"
)
POLICY_PATH = (
    PROJECT_ROOT
    / "docs"
    / "research"
    / "inclusive_vlm_lep"
    / "parser_normalization_policy.md"
)
POLICY_VERSION = "1.1.0"
SEPARATOR_PATTERN = re.compile(r"\s*(?:,|;|\bas well as\b|\band\b)\s*")
WHITESPACE_PATTERN = re.compile(r"\s+")


@dataclass(frozen=True)
class ParsedOutput:
    """Normalized free-form parse result."""

    raw_text: str
    normalized_text: str
    predicted_labels: list[str]
    unresolved_tokens: list[str]
    flags: list[str]
    parser_status: str

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return {
            "raw_text": self.raw_text,
            "normalized_text": self.normalized_text,
            "normalized_predictions": list(self.predicted_labels),
            "unresolved_tokens": list(self.unresolved_tokens),
            "flags": list(self.flags),
            "parser_status": self.parser_status,
        }


@dataclass(frozen=True)
class AliasEntry:
    """Normalized alias phrase for greedy stage-5 matching."""

    canonical: str
    normalized_alias: str
    tokens: tuple[str, ...]


def configure_logging(level: str = "INFO") -> None:
    """Configure console logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _extract_policy_version(path: Path) -> str:
    """Read the parser policy version pin."""

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("policy_version:"):
                return line.split(":", 1)[1].strip()
    raise ValueError(f"Missing policy_version in {path}")


def load_alias_table(path: Path | str = DEFAULT_ALIAS_CONFIG) -> tuple[dict[str, list[str]], set[str]]:
    """Load the version-pinned alias table.

    Args:
        path: YAML alias config path.

    Returns:
        Tuple of alias mapping and canonical label set.
    """

    path = Path(path)
    config = load_yaml(path)
    policy_version = str(config.get("policy_version", "")).strip()
    expected_version = _extract_policy_version(POLICY_PATH)
    if policy_version != expected_version or expected_version != POLICY_VERSION:
        raise ValueError(
            "Alias table policy version mismatch: "
            f"aliases={policy_version!r}, policy={expected_version!r}, expected={POLICY_VERSION!r}"
        )

    aliases = config.get("aliases") or {}
    if not isinstance(aliases, dict):
        raise ValueError(f"Invalid aliases mapping in {path}")

    merged: dict[str, list[str]] = {}
    for canonical, raw_aliases in aliases.items():
        merged[str(canonical)] = [str(alias) for alias in raw_aliases or []]
    return merged, set(merged)


def _stage_one(text: str) -> str:
    """Stage 1: strip, lowercase, collapse whitespace."""

    return WHITESPACE_PATTERN.sub(" ", text.strip().lower())


def _stage_two(text: str) -> str:
    """Stage 2: apply Unicode NFKC normalization."""

    return unicodedata.normalize("NFKC", text)


def _stage_three(text: str) -> str:
    """Stage 3: trim punctuation while preserving separators and token internals."""

    cleaned_chars: list[str] = []
    for char in text:
        if char.isalnum() or char in {"-", "/", ",", ";"} or char.isspace():
            cleaned_chars.append(char)
        else:
            cleaned_chars.append(" ")
    collapsed = WHITESPACE_PATTERN.sub(" ", "".join(cleaned_chars)).strip()
    collapsed = re.sub(r"\s*([,;])\s*", r"\1", collapsed)
    return collapsed


def _normalize_item_text(text: str) -> str:
    """Normalize an individual candidate item for exact alias matching."""

    return _stage_three(_stage_two(_stage_one(text)))


def _stage_four(text: str) -> list[str]:
    """Stage 4: split text into phrase-level candidate items."""

    if not text:
        return []
    if SEPARATOR_PATTERN.search(text):
        parts = [part for part in SEPARATOR_PATTERN.split(text) if part]
    else:
        parts = [text]
    items = [_normalize_item_text(part) for part in parts]
    return [item for item in items if item]


def _build_alias_lookup(alias_table: dict[str, list[str]]) -> dict[str, str]:
    """Build exact-match alias lookup after the cleanup stages."""

    lookup: dict[str, str] = {}
    for canonical, aliases in alias_table.items():
        normalized_canonical = _normalize_item_text(canonical)
        if normalized_canonical:
            lookup[normalized_canonical] = canonical
        for alias in aliases:
            normalized_alias = _normalize_item_text(alias)
            if normalized_alias:
                lookup[normalized_alias] = canonical
    return lookup


def _build_alias_phrase_index(alias_table: dict[str, list[str]]) -> dict[str, list[AliasEntry]]:
    """Build a first-token index for greedy longest-match alias parsing."""

    deduped_entries: dict[tuple[str, str], AliasEntry] = {}
    for canonical, aliases in alias_table.items():
        normalized_forms = [_normalize_item_text(canonical)]
        normalized_forms.extend(_normalize_item_text(alias) for alias in aliases)
        for normalized_alias in normalized_forms:
            tokens = tuple(token for token in normalized_alias.split(" ") if token)
            if not tokens:
                continue
            key = (canonical, normalized_alias)
            deduped_entries[key] = AliasEntry(
                canonical=canonical,
                normalized_alias=normalized_alias,
                tokens=tokens,
            )

    phrase_index: dict[str, list[AliasEntry]] = {}
    for entry in deduped_entries.values():
        phrase_index.setdefault(entry.tokens[0], []).append(entry)

    for entries in phrase_index.values():
        entries.sort(
            key=lambda entry: (-len(entry.tokens), -len(entry.normalized_alias), entry.normalized_alias)
        )
    return phrase_index


def _stage_five(candidate: str, alias_index: dict[str, list[AliasEntry]]) -> tuple[list[str], list[str]]:
    """Stage 5: greedily decompose a candidate item into canonical labels and leftovers."""

    tokens = [token for token in candidate.split(" ") if token]
    if not tokens:
        return [], []

    matched_labels: list[str] = []
    unresolved_tokens: list[str] = []
    unresolved_buffer: list[str] = []
    position = 0

    while position < len(tokens):
        current = tokens[position]
        entries = alias_index.get(current, [])
        matched_entry: AliasEntry | None = None
        for entry in entries:
            width = len(entry.tokens)
            if tuple(tokens[position : position + width]) == entry.tokens:
                matched_entry = entry
                break

        if matched_entry is None:
            unresolved_buffer.append(current)
            position += 1
            continue

        if unresolved_buffer:
            unresolved_tokens.append(" ".join(unresolved_buffer))
            unresolved_buffer = []
        matched_labels.append(matched_entry.canonical)
        position += len(matched_entry.tokens)

    if unresolved_buffer:
        unresolved_tokens.append(" ".join(unresolved_buffer))

    return matched_labels, unresolved_tokens


def parse_freeform_output(
    raw_text: str,
    canonical_target_set: set[str] | list[str],
    alias_table: dict[str, list[str]],
) -> ParsedOutput:
    """Parse a free-form attribution response using the frozen 7-stage pipeline.

    Args:
        raw_text: Raw model output.
        canonical_target_set: Valid canonical labels for the scored item.
        alias_table: Version-pinned alias table.

    Returns:
        ParsedOutput with normalized predictions, unresolved tokens, and flags.
    """

    stage_one = _stage_one(raw_text)
    stage_two = _stage_two(stage_one)
    normalized_text = _stage_three(stage_two)
    candidates = _stage_four(normalized_text)

    if not candidates:
        return ParsedOutput(
            raw_text=raw_text,
            normalized_text=normalized_text,
            predicted_labels=[],
            unresolved_tokens=[],
            flags=["unparsable"],
            parser_status="unparsable",
        )

    valid_targets = set(canonical_target_set)
    alias_lookup = _build_alias_lookup(alias_table)
    alias_index = _build_alias_phrase_index(alias_table)
    deduped_predictions: list[str] = []
    seen_predictions: set[str] = set()
    unresolved_tokens: list[str] = []

    for candidate in candidates:
        canonical = alias_lookup.get(candidate)
        candidate_matches: list[str]
        candidate_unresolved: list[str]
        if canonical is None:
            candidate_matches, candidate_unresolved = _stage_five(candidate, alias_index)
        else:
            candidate_matches, candidate_unresolved = [canonical], []

        unresolved_tokens.extend(candidate_unresolved)
        for matched_canonical in candidate_matches:
            if matched_canonical in valid_targets:
                if matched_canonical not in seen_predictions:
                    seen_predictions.add(matched_canonical)
                    deduped_predictions.append(matched_canonical)
            else:
                unresolved_tokens.append(matched_canonical)

    flags: list[str] = []
    if unresolved_tokens:
        flags.append("hallucinated")
    if "none_visible" in seen_predictions:
        flags.append("none_visible_marker")
    parser_status = "hallucinated" if unresolved_tokens else "ok"
    return ParsedOutput(
        raw_text=raw_text,
        normalized_text=normalized_text,
        predicted_labels=deduped_predictions,
        unresolved_tokens=unresolved_tokens,
        flags=flags,
        parser_status=parser_status,
    )


def _run_self_test(alias_table: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Run basic parser self-tests."""

    cases = [
        {
            "name": "left alias",
            "raw_text": "lhs",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["left"],
            "expected_status": "ok",
        },
        {
            "name": "right alias",
            "raw_text": "on the right",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["right"],
            "expected_status": "ok",
        },
        {
            "name": "upper arm alias",
            "raw_text": "above elbow",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["upper_arm"],
            "expected_status": "ok",
        },
        {
            "name": "forearm alias",
            "raw_text": "below-elbow",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["forearm"],
            "expected_status": "ok",
        },
        {
            "name": "thigh alias",
            "raw_text": "proximal leg",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["thigh"],
            "expected_status": "ok",
        },
        {
            "name": "calf alias",
            "raw_text": "shin area",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["calf"],
            "expected_status": "ok",
        },
        {
            "name": "residual limb alias",
            "raw_text": "amputated arm",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["residual_limb"],
            "expected_status": "ok",
        },
        {
            "name": "prosthesis alias",
            "raw_text": "prosthetic device",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["prosthesis"],
            "expected_status": "ok",
        },
        {
            "name": "none visible alias",
            "raw_text": "none visible",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["none_visible"],
            "expected_status": "ok",
        },
        {
            "name": "composite arm phrase decomposes",
            "raw_text": "right forearm prosthesis",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["right", "forearm", "prosthesis"],
            "expected_status": "ok",
        },
        {
            "name": "composite leg phrase decomposes in first-seen order",
            "raw_text": "below-knee prosthesis on the right",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["calf", "prosthesis", "right"],
            "expected_status": "ok",
        },
        {
            "name": "stage four split then stage five decompose",
            "raw_text": "amputated leg and left upper arm",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": ["residual_limb", "left", "upper_arm"],
            "expected_status": "ok",
        },
        {
            "name": "out-of-space canonical becomes unresolved",
            "raw_text": "left, prosthesis, calf",
            "target": {"right", "prosthesis", "calf"},
            "expected_predictions": ["prosthesis", "calf"],
            "expected_status": "hallucinated",
        },
        {
            "name": "removed abstention alias is unresolved",
            "raw_text": "unsure",
            "target": {"left", "right", "upper_arm", "forearm", "thigh", "calf", "residual_limb", "prosthesis", "none_visible"},
            "expected_predictions": [],
            "expected_status": "hallucinated",
        },
        {
            "name": "empty output is unparsable",
            "raw_text": "   ",
            "target": {"none_visible"},
            "expected_predictions": [],
            "expected_status": "unparsable",
        },
    ]

    results: list[dict[str, Any]] = []
    for case in cases:
        parsed = parse_freeform_output(
            raw_text=case["raw_text"],
            canonical_target_set=set(case["target"]),
            alias_table=alias_table,
        )
        passed = (
            parsed.predicted_labels == list(case["expected_predictions"])
            and parsed.parser_status == case["expected_status"]
        )
        results.append(
            {
                "name": case["name"],
                "passed": passed,
                "parsed": parsed.to_dict(),
            }
        )
    return results


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Normalize Attribution-FreeForm outputs with the frozen parser policy."
    )
    parser.add_argument(
        "--raw-text",
        type=str,
        default="",
        help="Raw free-form text to parse.",
    )
    parser.add_argument(
        "--canonical-target-set",
        type=str,
        default="",
        help="Comma separated canonical target set.",
    )
    parser.add_argument(
        "--alias-config",
        type=Path,
        default=DEFAULT_ALIAS_CONFIG,
        help="Path to the alias-table YAML.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run parser self-tests instead of parsing one string.",
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
    alias_table, _ = load_alias_table(args.alias_config)

    if args.self_test:
        results = _run_self_test(alias_table)
        failed = [item for item in results if not item["passed"]]
        LOGGER.info(
            "Parser self-test: %d passed, %d failed",
            len(results) - len(failed),
            len(failed),
        )
        print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
        if failed:
            raise SystemExit(1)
        return

    canonical_target_set = {
        token.strip()
        for token in args.canonical_target_set.split(",")
        if token.strip()
    }
    parsed = parse_freeform_output(
        raw_text=args.raw_text,
        canonical_target_set=canonical_target_set,
        alias_table=alias_table,
    )
    print(json.dumps(parsed.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
