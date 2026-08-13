"""Command-line entry point.

Extract a captured JSON from an exported tool report (docx/PDF) — writes a
draft you should review before scoring. Works fully offline with a local
Ollama model:
    benchmarking extract --input tool_report.pdf --out captured.json \\
        --backend ollama --model llama3.1

Single run:
    benchmarking run --tool "Google Hypothesis Generation" --output captured.json \\
        --prompt "artemisinin-resistant Plasmodium knowlesi" \\
        --embeddings specter2 --judge anthropic --out results.json

Bundle run (adds reproducibility/robustness — needs repeated and/or perturbed
captures of the same prompt):
    benchmarking run --tool "..." --output base.json --prompt "..." \\
        --repeat repeat1.json --repeat repeat2.json \\
        --perturb reword=reword1.json \\
        --embeddings specter2 --out results.json
    # --perturb also accepts kb_removal=... (a capture from a knowledge-base-
    # removed run), reported by robustness for interpretation but never
    # scored -- omitted from the example above since no tool we've evaluated
    # currently exposes a way to tweak its knowledge base.

Loads captured tool output(s) and runs the full evaluation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .core.config import RunConfig
from .core.context import Context
from .core.models import RunBundle, Tool
from .io.extraction import extract_hypotheses
from .io.readers import read_document_text
from .io.tool_adapters.file_adapter import FileAdapter
from .pipeline import evaluate, evaluate_bundle
from .report import results
from .services.embeddings import (
    SCIBERT,
    SPECTER2,
    HashingEmbedding,
    OllamaEmbedding,
    OpenAIEmbedding,
    SentenceTransformerEmbedding,
)
from .services.biodb import UniProtClient
from .services.literature import CompositeLiteratureClient, CrossRefClient, SemanticScholarClient
from .services.llm_judge import AnthropicJudge, OllamaJudge, OpenAIJudge, ProgressJudge
from .services.structured_chat import AnthropicChat, OllamaChat, OpenAIChat

_EMBEDDING_BACKENDS = {
    "hashing": lambda a: HashingEmbedding(),
    "specter2": lambda a: SentenceTransformerEmbedding(a.embeddings_model or SPECTER2),
    "scibert": lambda a: SentenceTransformerEmbedding(a.embeddings_model or SCIBERT),
    "openai": lambda a: OpenAIEmbedding(a.embeddings_model or "text-embedding-3-large"),
    "ollama": lambda a: OllamaEmbedding(a.embeddings_model or "nomic-embed-text",
                                       host=a.embeddings_host),
}

_JUDGE_BACKENDS = {
    "none": lambda a: None,
    "anthropic": lambda a: AnthropicJudge(a.judge_model or "claude-opus-4-8"),
    "openai": lambda a: OpenAIJudge(a.judge_model or "gpt-4.1"),
    "ollama": lambda a: OllamaJudge(a.judge_model or "llama3.1", host=a.judge_host),
}

_CHAT_BACKENDS = {
    "anthropic": lambda a: AnthropicChat(a.model or "claude-opus-4-8"),
    "openai": lambda a: OpenAIChat(a.model or "gpt-4.1", base_url=a.base_url),
    "ollama": lambda a: OllamaChat(a.model or "llama3.1", host=a.host),
}

_LITERATURE_BACKENDS = {
    "none": lambda a: None,
    "crossref": lambda a: CrossRefClient(mailto=a.literature_mailto),
    "semantic_scholar": lambda a: SemanticScholarClient(),
    "composite": lambda a: CompositeLiteratureClient(mailto=a.literature_mailto),
}

_BIODB_BACKENDS = {
    "none": lambda a: None,
    "uniprot": lambda a: UniProtClient(),
}


def _build_context(args: argparse.Namespace) -> Context:
    embeddings = _EMBEDDING_BACKENDS[args.embeddings](args)
    judge = _JUDGE_BACKENDS[args.judge](args)
    if judge is not None and args.judge_progress:
        # No fixed total: a run scores multiple hypotheses across several
        # judge-dependent metrics, so the total call count isn't known ahead
        # of time. A running per-call count with elapsed time is still real
        # feedback against a silent, possibly multi-minute wait.
        judge = ProgressJudge(judge)
    literature = _LITERATURE_BACKENDS[args.literature](args)
    biodb = _BIODB_BACKENDS[args.biodb](args)
    config = RunConfig(multi_run_comparison=args.multi_run_comparison)
    return Context(config=config, embeddings=embeddings, judge=judge, literature=literature, biodb=biodb)


def _cmd_extract(args: argparse.Namespace) -> int:
    raw_text = read_document_text(args.input)
    chat = _CHAT_BACKENDS[args.backend](args)
    data = extract_hypotheses(raw_text, chat, max_tokens=args.max_tokens)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(data, indent=2))
    print(
        f"Draft extraction written to {args.out} (backend: {chat.name}) — "
        f"review it (claims, references) before running `benchmarking run` on it.",
        file=sys.stderr,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    ctx = _build_context(args)
    tool = Tool(name=args.tool, version=args.version)

    base = FileAdapter(tool=tool, output_path=args.output).run(prompt=args.prompt)

    if args.repeat or args.perturb:
        repeats = [
            FileAdapter(tool=tool, output_path=p).run(prompt=args.prompt) for p in args.repeat
        ]
        perturbations = []
        for spec in args.perturb:
            ptype, _, path = spec.partition("=")
            run = FileAdapter(tool=tool, output_path=path).run(prompt=args.prompt)
            run.metadata["perturbation"] = ptype
            perturbations.append(run)
        bundle = RunBundle(base=base, repeats=repeats, perturbations=perturbations)
        score = evaluate_bundle(bundle, ctx)
    else:
        score = evaluate(base, ctx)

    if args.out:
        results.write_json(score, args.out)
        print(f"Results written to {args.out}", file=sys.stderr)
    else:
        print(json.dumps(results.to_dict(score), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="benchmarking", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    extract_p = sub.add_parser(
        "extract", help="draft a captured JSON from an exported tool report (docx/PDF)"
    )
    extract_p.add_argument("--input", required=True, type=Path,
                           help="exported tool report (.docx, .pdf, .md/.markdown, or .txt)")
    extract_p.add_argument("--backend", choices=sorted(_CHAT_BACKENDS), default="anthropic",
                           help="model backend used for extraction (default: anthropic; "
                                "use ollama to run fully offline)")
    extract_p.add_argument("--model", default=None,
                           help="override the default model name for the chosen backend")
    extract_p.add_argument("--host", default=None,
                           help="Ollama server host (only used with --backend ollama; "
                                "default: http://localhost:11434)")
    extract_p.add_argument("--base-url", default=None,
                           help="OpenAI-compatible server base URL (only used with "
                                "--backend openai, to point at a non-OpenAI server)")
    extract_p.add_argument("--max-tokens", type=int, default=8192,
                           help="output token budget for the extraction call (default: 8192). "
                                "A large report with many hypotheses can exceed this, causing the "
                                "model to close the JSON early with only some hypotheses captured "
                                "-- raise this if `extract`'s output has fewer hypotheses than the "
                                "source document.")
    extract_p.add_argument("--out", required=True, type=Path,
                           help="write the draft captured JSON here")
    extract_p.set_defaults(func=_cmd_extract)

    run_p = sub.add_parser("run", help="evaluate captured tool output(s)")
    run_p.add_argument("--tool", required=True, help="tool name")
    run_p.add_argument("--version", default=None, help="tool version / model weights")
    run_p.add_argument("--output", required=True, type=Path,
                       help="captured tool output for the base run (JSON)")
    run_p.add_argument("--prompt", default="", help="the research prompt that was submitted")
    run_p.add_argument("--repeat", action="append", default=[], type=Path,
                       help="captured output from an identical repeat run "
                            "(repeatable; enables reproducibility)")
    run_p.add_argument("--perturb", action="append", default=[],
                       help="TYPE=path.json, e.g. reword=out2.json (repeatable; enables "
                            "robustness). TYPE also accepts kb_removal, reported for "
                            "interpretation but never scored -- not shown as a primary "
                            "example since no tool we've evaluated currently exposes a "
                            "way to tweak its knowledge base")
    run_p.add_argument("--multi-run-comparison", choices=["whole_set", "top_hypothesis"],
                       default="whole_set",
                       help="what reproducibility/robustness compare across runs (default: "
                            "whole_set, mean-pooled; top_hypothesis compares only the rank-1 "
                            "idea across runs — sharper but assumes rank 1 is a stable slot)")
    run_p.add_argument("--embeddings", choices=sorted(_EMBEDDING_BACKENDS), default="hashing",
                       help="embedding backend (default: hashing, a non-semantic placeholder; "
                            "use ollama to run fully offline)")
    run_p.add_argument("--embeddings-model", default=None,
                       help="override the default model name for the chosen embedding backend")
    run_p.add_argument("--embeddings-host", default=None,
                       help="Ollama server host (only used with --embeddings ollama)")
    run_p.add_argument("--judge", choices=sorted(_JUDGE_BACKENDS), default="none",
                       help="LLM-judge backend (default: none -> judge-dependent metrics "
                            "report 'not assessed'; use ollama to run fully offline)")
    run_p.add_argument("--judge-model", default=None,
                       help="override the default model name for the chosen judge backend")
    run_p.add_argument("--judge-host", default=None,
                       help="Ollama server host (only used with --judge ollama)")
    run_p.add_argument("--judge-progress", action="store_true",
                       help="print a line per judge call (verdict + elapsed time) to stderr "
                            "as the run progresses, instead of a silent wait")
    run_p.add_argument("--literature", choices=sorted(_LITERATURE_BACKENDS), default="none",
                       help="literature client for citation_accuracy (default: none -> "
                            "reports 'not assessed'). 'composite' tries CrossRef first, "
                            "falls back to Semantic Scholar for low-confidence matches.")
    run_p.add_argument("--literature-mailto", default=None,
                       help="email for CrossRef's polite pool (optional; higher rate limits)")
    run_p.add_argument("--biodb", choices=sorted(_BIODB_BACKENDS), default="none",
                       help="biological database client for entity_accuracy (default: none -> "
                            "reports 'not assessed'). 'uniprot' resolves gene/protein entities "
                            "against UniProtKB.")
    run_p.add_argument("--out", type=Path, default=None,
                       help="write results JSON here (default: stdout)")
    run_p.set_defaults(func=_cmd_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
