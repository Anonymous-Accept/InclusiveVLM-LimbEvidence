# Reproduce InclusiveVLM-LEP

These commands assume the externally supplied full dataset has been placed at
`data/inclusive_vlm_lep/`.

Repository code positions after upload:

- `scripts/inclusive_vlm_lep/` for Limb-Evidence Grounding and Prosthesis
  Matching code.
- `scripts/core/` for shared runtime utilities and official API backends.
- `configs/inclusive_vlm_lep/` for full-dataset evaluation configs.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure one backend:

```bash
export OPENAI_API_KEY="..."
export GEMINI_API_KEY="..."
export VLLM_API_BASE="http://localhost:8000/v1"
export VLLM_API_KEY="EMPTY"
```

## Limb-Evidence Grounding

Run Presence and Attribution-Constrained inference:

```bash
python -m scripts.inclusive_vlm_lep.limb_evidence_grounding.inference_runner \
  --queries-path data/inclusive_vlm_lep/limb_evidence_grounding/queries_v2.jsonl \
  --model-name gpt-5 \
  --output-path outputs/inclusive_vlm_lep/limb_evidence_grounding/runs/gpt-5/predictions.jsonl
```

Evaluate predictions:

```bash
python -m scripts.inclusive_vlm_lep.limb_evidence_grounding.evaluate \
  --queries-path data/inclusive_vlm_lep/limb_evidence_grounding/queries_v2.jsonl \
  --results-path outputs/inclusive_vlm_lep/limb_evidence_grounding/runs/gpt-5/predictions.jsonl \
  --model-name gpt-5 \
  --metrics-path outputs/inclusive_vlm_lep/limb_evidence_grounding/metrics/gpt-5.metrics.json
```

## Attribution-FreeForm

Run free-form inference:

```bash
python -m scripts.inclusive_vlm_lep.freeform.run_inference \
  --subset-jsonl data/inclusive_vlm_lep/freeform/freeform_queries.jsonl \
  --model-name gpt-5 \
  --output-dir outputs/inclusive_vlm_lep/freeform/runs/gpt-5
```

Evaluate free-form outputs:

```bash
python -m scripts.inclusive_vlm_lep.freeform.evaluate_attribution_freeform \
  --subset-jsonl data/inclusive_vlm_lep/freeform/freeform_queries.jsonl \
  --predictions outputs/inclusive_vlm_lep/freeform/runs/gpt-5/predictions.jsonl \
  --output-dir outputs/inclusive_vlm_lep/freeform/metrics/gpt-5
```

## Prosthesis Matching

Run Compatibility-Category:

```bash
python -m scripts.inclusive_vlm_lep.prosthesis_match.run_inference \
  --items-path data/inclusive_vlm_lep/prosthesis_match/items/compatibility_category.jsonl \
  --model-name gpt-5 \
  --output-dir outputs/inclusive_vlm_lep/prosthesis_matching/runs/gpt-5/compatibility_category
```

Run Compatibility-Diversity:

```bash
python -m scripts.inclusive_vlm_lep.prosthesis_match.run_inference \
  --items-path data/inclusive_vlm_lep/prosthesis_match/items/prosthesis_match.jsonl \
  --model-name gpt-5 \
  --output-dir outputs/inclusive_vlm_lep/prosthesis_matching/runs/gpt-5/compatibility_diversity
```

Evaluate a Prosthesis Matching run:

```bash
python -m scripts.inclusive_vlm_lep.prosthesis_match.eval_prosthesis_match \
  --predictions-path outputs/inclusive_vlm_lep/prosthesis_matching/runs/gpt-5/compatibility_category/predictions.jsonl \
  --items-path data/inclusive_vlm_lep/prosthesis_match/items/compatibility_category.jsonl \
  --output-path outputs/inclusive_vlm_lep/prosthesis_matching/metrics/gpt-5/compatibility_category.metrics.json
```

## Aggregate Metrics

After predictions are available for the benchmark variants, recompute and
aggregate metrics:

```bash
python -m scripts.inclusive_vlm_lep.orchestrator.run_all \
  --config configs/inclusive_vlm_lep/orchestrator_default.yaml
```

The generated outputs are written under `outputs/` and are intentionally not
part of this reviewer code repository.
