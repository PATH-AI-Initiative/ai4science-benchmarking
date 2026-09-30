# AI Co-Scientist Benchmarking Suite

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
    scoring.py     #   safety gate, best-hypothesis (top-k), weighted composite
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
- **Rubric rules live in one place.** Safety gate, top-k best-hypothesis
  focus, and weighting are all in `core/scoring.py`, driven by `RunConfig`.

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

### Quickstart

`examples/example_capture.json` is a small synthetic capture (two
textbook-mechanism hypotheses, not real tool output) so you can see the
pipeline run with zero setup — no API key, no captured data of your own:

```bash
uv run benchmarking run --tool "Example Tool" \
  --output examples/example_capture.json \
  --prompt "novel therapeutic mechanisms for metabolic disease" \
  --out results.json
```

`diversity` scores for real out of the box (`HashingEmbedding` needs no
service). Everything else reports "not assessed" until you attach a
literature client, biodb client, and/or judge — see *Scoring* below.

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

### Scoring

Attach whichever services you want the run scored against — see *Status*
above for what each metric needs:

```bash
uv sync --extra openai --extra local-embeddings
export OPENAI_API_KEY=...
uv run benchmarking run --tool "My Tool" --output captured_output.json \
  --prompt "..." \
  --embeddings specter2 --judge openai --literature composite --biodb uniprot \
  --out results.json
```

Add reproducibility/robustness by passing repeated and/or reworded captures
of the same prompt (`--repeat`, `--perturb reword=...`); see `benchmarking
run --help` for the full flag list, including `--multi-run-comparison` and
`--adversarial-ground-truth`.

## Acknowledgements


This work was supported through AI for Development Science Breakthroughs (AI4DSB), funded by the UK Government's Foreign, Commonwealth & Development Office (FCDO) through its Global Research and Technology Development (GRTD) portfolio. The views expressed do not necessarily reflect the UK government’s official policies.
<img width="468" height="82" alt="image" src="https://github.com/user-attachments/assets/7a76d4d7-6c7a-4c93-8ff9-13afafddf258" />

