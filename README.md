# ai4science-benchmarking

A modular, open-source benchmarking suite for evaluating AI co-scientist tools.

## Architecture

One subpackage per axis, one module per metric.

```
benchmarking_pipeline/
  core/            # shared abstractions
    models.py      #   Tool, EvaluationRun, Hypothesis(Set), Claim, Reference, Entity
    metric.py      #   Metric interface: HypothesisMetric / SetMetric, MetricResult, Axis
    registry.py    #   @register — metrics self-register; axes/pipeline discover them
    config.py      #   RunConfig: weights + thresholds (vary this for sensitivity analysis)
    context.py     #   Context: services + config handed to every metric
    scoring.py     #   floor, safety gate, best-hypothesis (top-k), weighted composite
  services/        # cross-cutting capabilities
    embeddings.py  #   embed() + cosine — the primitive shared by novelty/robustness/reproducibility
    llm_judge.py   #   LLM-as-judge protocol
    literature/    #   CrossRef / Semantic Scholar protocol (citation + novelty lookups)
    biodb.py       #   UniProt / KEGG protocol (entity validation)
  io/
    parsers.py     #   captured tool output -> HypothesisSet
    tool_adapters/ #   how a run is produced (file-based today, live APIs later)
  axes/
    details_features/  # Tier 1 — structured audit checklist
    accuracy/          # Tier 2 — citation existence/support, entity, epistemic calibration, logical consistency
    quality/            # Tier 3 — robustness, reproducibility, adversarial, tractability, safety
    novelty/            # Tier 4 — size of leap, diversity
  pipeline.py      # prompt -> adapter -> axes -> scoring
  report/          # JSON results
  cli.py           # `benchmarking run ...`
```

### Design principles

- **Metrics are pluggable.** Subclass `HypothesisMetric`, `SetMetric`, or
  `MultiRunMetric`, set `axis`, decorate with `@register`. That's the whole
  contract.
- **Services are behind protocols.** Embedding model, judge model, and
  database clients are chosen per run and attached to `Context`; no metric
  depends on a concrete backend. None is attached by default (`--judge none`
  etc.) — pick one explicitly. Backends: `AnthropicJudge` / `OpenAIJudge` /
  `OllamaJudge` (local, no API key); `SentenceTransformerEmbedding` (SPECTER2,
  SciBERT) / `OpenAIEmbedding` / `OllamaEmbedding` / `HashingEmbedding`
  (placeholder). Judge and extraction backends share one
  `StructuredChatClient` primitive (`services/structured_chat.py`). Pin your
  choices for a defensible benchmark.
- **Rubric rules live in one place.** Accuracy floor, safety gate, top-k
  best-hypothesis focus, and weighting are all in `core/scoring.py`, driven
  by `RunConfig`.

## Status

Fully implemented: core abstractions, scoring layer, Tier 1 checklist, every
Accuracy and Novelty metric, and Quality's `reproducibility`, `robustness`,
and `adversarial`. Each needs the right service attached to score rather than
report "not assessed":

| Metric | Needs |
|---|---|
| `citation_accuracy` / `citation_support` | literature client |
| `entity_accuracy` | biodb client |
| `epistemic_calibration` | literature client + judge |
| `logical_consistency` | judge |
| `size_of_leap` | literature client + embeddings (judge optional, gates anchor relevance) |
| `diversity` | embeddings |
| `adversarial` | judge + `--adversarial-ground-truth` |

`tractability` and `safety` are stubs — both need a domain expert's
judgement (SME review, red-team detection), not just more code. See each
module's docstring for the planned design.

Multi-run metrics score via `evaluate_bundle(RunBundle, ctx)`, where a
`RunBundle` is a base run plus repeated and/or perturbed runs.

Backends install as optional extras (one `uv sync` call — separate calls
replace rather than add): `uv sync --extra anthropic --extra
local-embeddings` (add `--extra openai` / `--extra ollama` as needed).
Default embedding is `HashingEmbedding`, a non-semantic placeholder — swap in
a real model before drawing scientific conclusions.

## Usage

### Capturing a tool's output

Already in the JSON shape below? Skip to *Scoring*. Otherwise, draft a
captured JSON from an exported docx/PDF report with an LLM-assisted
extraction pass — review the claims and references before scoring against
it, this is a draft, not ground truth:

```bash
uv sync --extra openai
export OPENAI_API_KEY=...
uv run benchmarking extract --input tool_report.pdf --out captured_output.json \
  --backend openai
# review/edit captured_output.json, then proceed to Scoring
```

**Fully offline with a local model via [Ollama](https://ollama.com):**

```bash
ollama serve                       # in another terminal
ollama pull llama3.1                # or any model that supports structured output

uv sync --extra ollama
uv run benchmarking extract --input tool_report.pdf --out captured_output.json \
  --backend ollama --model llama3.1
```

Input capture format (`--output` for `run`, and what `extract` produces):

```json
{
  "hypotheses": [
    {"id": "h1", "rank": 1, "text": "...",
     "claims": [{"text": "...", "references": [{"raw": "...", "doi": "..."}]}]}
  ]
}
```
