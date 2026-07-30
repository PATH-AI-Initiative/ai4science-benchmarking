# ai4science-benchmarking

A modular, open-source benchmarking suite for evaluating AI co-scientist tools.

## Architecture

The package mirrors the framework: one subpackage per axis, one module per
metric. 

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
    accuracy/          # Tier 2 — citation, entity, epistemic calibration, logical consistency
    quality/           # Tier 3 — robustness, reproducibility, adversarial, tractability, safety
    novelty/           # Tier 4 — size of leap, diversity
  pipeline.py      # prompt -> adapter -> axes -> scoring
  report/          # JSON results + (planned) plain-language Playbook
  cli.py           # `benchmarking run ...`
```

### Design principles

- **Metrics are pluggable.** Subclass `HypothesisMetric` (scores one hypothesis),
  `SetMetric` (scores one output set), or `MultiRunMetric` (scores a set of
  runs — reproducibility, robustness), set `axis`, and decorate with
  `@register`. That is the entire contract.
- **Services are behind protocols.** The embedding model, judge model, and
  database clients are chosen per run and attached to the `Context`; no metric
  depends on a concrete backend. Concrete backends provided: `AnthropicJudge`
  (default) / `OpenAIJudge` / `OllamaJudge` (fully local, no API key);
  `SentenceTransformerEmbedding` (SPECTER2, SciBERT) / `OpenAIEmbedding` /
  `OllamaEmbedding` (local) / `HashingEmbedding` (placeholder). The judge and
  extraction backends share one `StructuredChatClient` primitive
  (`services/structured_chat.py`) — adding a new provider means adding one
  class there. Choices should be pinned/versioned for a defensible benchmark.
- **Rubric rules live in one place.** Accuracy floor, safety gate, top-k
  best-hypothesis focus, and weighting are all in `core/scoring.py`, driven by
  `RunConfig` — so stress-testing the weights is a matter of varying one object.

## Status

Runnable skeleton. Fully implemented: the core abstractions, scoring layer,
Tier 1 checklist, the `diversity` novelty metric, and the `reproducibility` and
`robustness` multi-run metrics (all embedding-based). `citation_accuracy` is
implemented but needs a literature client attached. Every other metric is a
registered stub returning a "not assessed" result, with its planned
implementation described in its module docstring.

Multi-run metrics are scored via `evaluate_bundle(RunBundle, ctx)`, where a
`RunBundle` collects a base run plus repeated and/or perturbed runs.

Backends install as optional extras (combine them in one `uv sync` call —
separate invocations replace rather than add to the installed set):
`uv sync --extra anthropic --extra local-embeddings` (add `--extra openai`
and/or `--extra ollama` as needed). The out-of-the-box default embedding is
`HashingEmbedding`, a deterministic non-semantic stand-in — swap in SPECTER2
or another real model before drawing scientific conclusions.

## Usage

### Capturing a tool's output

If the tool's raw output is already in the JSON shape below, skip to *Scoring*.
If it's an exported docx/PDF report, draft a captured JSON with an LLM-assisted
extraction pass first — this is a first draft, not a capture, so review the
claims and references it pulls out before scoring against it:

```bash
uv sync --extra anthropic
export OPENAI_API_KEY=...
uv run benchmarking extract --input tool_report.pdf --out captured_output.json
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

