# InclusiveVLM-LEP

Anonymous reviewer artifact for **Seeing Is Not Enough: Evaluating Limb-Evidence
Preservation in Vision-Language Models on Images of Individuals with Limb
Deficiencies**.

InclusiveVLM-LEP is a test-only, human-calibrated benchmark on open-world
images of individuals with limb deficiencies. It links readouts through a shared
annotation basis, moving from coarse recognition to body-region attribution and
compatibility-style option matching. The benchmark contains two paper-facing
components:

- **Limb-Evidence Grounding**: Presence, Attribution-Constrained, and
  Attribution-FreeForm readouts over the same limb-evidence basis.
- **Prosthesis Matching**: Compatibility-Category and Compatibility-Diversity
  readouts over visible body-device configuration evidence.

This repository contains reviewer-facing code only. The full benchmark data is
provided separately through the review artifact channel. After upload, the
repository location is:

```text
https://github.com/Anonymous-Accept/InclusiveVLM-LEP
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For local open-weight VLM runs, install a vLLM build compatible with your CUDA
environment if it is not already available.

## Data Layout

Place the externally supplied full benchmark data at:

```text
data/inclusive_vlm_lep/
```

The evaluation code expects these full-dataset files:

```text
data/inclusive_vlm_lep/limb_evidence_grounding/queries_v2.jsonl
data/inclusive_vlm_lep/freeform/freeform_queries.jsonl
data/inclusive_vlm_lep/prosthesis_match/items/compatibility_category.jsonl
data/inclusive_vlm_lep/prosthesis_match/items/prosthesis_match.jsonl
```

Image paths inside those files are project-relative and should resolve under the
same `data/inclusive_vlm_lep/` root.

## API Configuration

Copy `env.example` to `.env` or export variables in your shell:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export VLLM_API_BASE="http://localhost:8000/v1"
export VLLM_API_KEY="EMPTY"
```

The public artifact supports official OpenAI and Gemini APIs plus local
vLLM-compatible serving. Proxy and intermediary API routes are intentionally not
included.

## Repository Layout

- [configs/inclusive_vlm_lep/](configs/inclusive_vlm_lep/) contains the public
  full-dataset evaluation config.
- [scripts/core/](scripts/core/) contains shared I/O, metrics, logging, and
  backend code.
- [scripts/inclusive_vlm_lep/](scripts/inclusive_vlm_lep/) contains the
  Limb-Evidence Grounding and Prosthesis Matching benchmark code.
- [REPRODUCE.md](REPRODUCE.md) contains full-dataset reproduction commands.
- [index.html](index.html) is the anonymous project page. The page uses only
  paper figure composites confirmed from the submitted PDF and re-rendered as
  metadata-stripped PNG files under [assets/figures/](assets/figures/).

## Scope

InclusiveVLM-LEP is a diagnostic evaluation artifact. It is not intended to
evaluate prosthesis recommendation, clinical suitability, or real-world
assistive deployment. Scores should be interpreted as benchmark-defined
evidence-preservation measurements under the paper's tasks and metrics.
