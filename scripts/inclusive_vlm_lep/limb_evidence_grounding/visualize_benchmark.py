#!/usr/bin/env python3
"""
Visualize Limb-Evidence Grounding queries (Presence and Attribution) alongside the source image.

Usage:
  python -m scripts.inclusive_vlm_lep.limb_evidence_grounding.visualize_benchmark \
    --jsonl data/inclusive_vlm_lep/limb_evidence_grounding/queries_v2.jsonl \
    --output-dir outputs/inclusive_vlm_lep/limb_evidence_grounding/visuals_check \
    --limit 20
"""

from __future__ import annotations

import argparse
import json
import logging
import textwrap
from pathlib import Path
from typing import Any, Dict, Iterable, List

import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSONL = (
    PROJECT_ROOT
    / "data"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
    / "queries_v2.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "inclusive_vlm_lep"
    / "limb_evidence_grounding"
    / "visuals_check"
)
DATA_IMAGE_ROOTS = [
    PROJECT_ROOT / "data" / "inclusive_vlm_lep" / "images",
]

def load_records(jsonl_path: Path) -> Iterable[Dict[str, Any]]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def group_records(records: Iterable[Dict[str, Any]], limit: int | None) -> List[Dict[str, Any]]:
    """
    Group records by image_id.
    Returns list of dicts: {'image_id': ..., 'meta': ..., 'tasks': [record, ...]}`
    """
    grouped = {}
    for rec in records:
        img_id = rec["image_id"]
        if img_id not in grouped:
            grouped[img_id] = {
                "image_id": img_id,
                "file_name": rec.get("file_name"),
                "image_path": rec.get("image_path"),
                "tasks": []
            }
        grouped[img_id]["tasks"].append(rec)

    items = list(grouped.values())
    # Sort by image_id for stability
    items.sort(key=lambda x: str(x["image_id"]))

    if limit is not None:
        return items[:limit]
    return items

def _resolve_image_path(rec: Dict[str, Any]) -> Path | None:
    # Try explicit image_path first
    if rec.get("image_path"):
        p = Path(rec["image_path"])
        if p.exists(): return p
        if not p.is_absolute():
            alt = PROJECT_ROOT / p
            if alt.exists(): return alt

    # Try file_name
    fname = rec.get("file_name")
    if fname:
        p = Path(fname)
        if p.exists(): return p
        for root in DATA_IMAGE_ROOTS:
            alt = root / p.name
            if alt.exists(): return alt
            # Also try assuming fname is relative to root
            alt2 = root / fname
            if alt2.exists(): return alt2

    return None

def render_panel(group: Dict[str, Any], output_path: Path) -> None:
    image_path = _resolve_image_path(group)
    tasks = group["tasks"]

    # Sort tasks: Presence first, then Segments
    tasks.sort(key=lambda x: (x["task_name"], x.get("prompt_id", 0)))

    fig = plt.figure(figsize=(20, 12), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

    # Left: Image
    ax_img = fig.add_subplot(gs[0])
    ax_img.axis("off")
    if image_path and image_path.exists():
        try:
            img = Image.open(image_path).convert("RGB")
            ax_img.imshow(img)
            ax_img.set_title(f"Image ID: {group['image_id']}\n{image_path.name}", fontsize=12)
        except Exception as e:
            ax_img.text(0.5, 0.5, f"Error loading image:\n{e}", ha="center", va="center", color="red")
    else:
        ax_img.text(0.5, 0.5, "Image not found", ha="center", va="center", fontsize=14)
        ax_img.set_title(f"Image ID: {group['image_id']}", fontsize=12)

    # Right: Text info
    ax_text = fig.add_subplot(gs[1])
    ax_text.axis("off")
    ax_text.set_ylim(0, 1)

    y = 0.98
    line_height = 0.025

    def write_line(text, size=10, bold=False, color="black", indent=0.0):
        nonlocal y
        weight = "bold" if bold else "normal"
        # specific handling for wrapping
        lines = textwrap.wrap(text, width=90)
        for line in lines:
            if y < 0.02: break
            ax_text.text(indent, y, line, fontsize=size, fontweight=weight, color=color, va="top", fontfamily="monospace")
            y -= line_height

    write_line(f"Total Query Records: {len(tasks)}", size=14, bold=True)
    y -= line_height

    # We might have many tasks, so we limit how many we show or just show one of each type + summary
    # Let's show up to 2 presence and 2 segments fully, or list them compactly

    presence_tasks = [t for t in tasks if t["task_name"] == "recognition"]
    segments_tasks = [t for t in tasks if t["task_name"] == "attribution"]

    if presence_tasks:
        write_line(f"--- Task: PRESENCE ({len(presence_tasks)} prompts) ---", size=12, bold=True, color="#2980b9")
        # Show details of the first one
        pt = presence_tasks[0]
        write_line(f"Prompt (ID {pt.get('prompt_id')})", size=10, bold=True)
        write_line(pt["prompt"], indent=0.02)
        y -= 0.01
        write_line(f"Options: {pt.get('options')}", indent=0.02)
        write_line(f"Target Answer: {pt.get('target_answer')} ({pt.get('option_descriptions', {}).get(pt.get('target_answer'), 'N/A')})", indent=0.02, bold=True, color="green")
        y -= line_height

    if segments_tasks:
        write_line(f"--- Task: ATTRIBUTION ({len(segments_tasks)} prompts) ---", size=12, bold=True, color="#8e44ad")
        # Show details of the first one
        st = segments_tasks[0]
        write_line(f"Prompt (ID {st.get('prompt_id')})", size=10, bold=True)
        write_line(st["prompt"], indent=0.02)
        y -= 0.01
        write_line(f"Target Labels: {st.get('target_labels')}", indent=0.02, bold=True, color="green")
        y -= line_height

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    if not args.jsonl.exists():
        logging.error(f"Input file not found: {args.jsonl}")
        return

    records = load_records(args.jsonl)
    grouped = group_records(records, args.limit)

    logging.info(f"Visualizing {len(grouped)} images...")

    for i, group in enumerate(grouped):
        out_name = f"{group['image_id']}.png"
        render_panel(group, args.output_dir / out_name)
        if (i+1) % 10 == 0:
            logging.info(f"Processed {i+1}/{len(grouped)}")

    logging.info(f"Done. Outputs in {args.output_dir}")

if __name__ == "__main__":
    main()
