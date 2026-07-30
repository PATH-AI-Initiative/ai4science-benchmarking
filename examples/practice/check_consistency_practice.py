"""Practice script: extract claims from one hypothesis, then check whether
those claims are logically consistent with each other.

Chains extract_claims_practice.py's output into the real LogicalConsistency
metric. A single claim is always trivially consistent (nothing to compare it
against) -- the meaningful check is across the several claims one hypothesis
decomposes into.

Run from the project root with:
    uv run python examples/practice/check_consistency_practice.py
    uv run python examples/practice/check_consistency_practice.py "Your hypothesis text here"
    uv run python examples/practice/check_consistency_practice.py path/to/one_hypothesis.pdf
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from extract_claims_practice import (  # noqa: E402
    _EXAMPLE_HYPOTHESIS,
    extract_claims,
    resolve_hypothesis_text,
)

from benchmarking_pipeline.axes.accuracy.logical_consistency import LogicalConsistency
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import (
    Claim,
    ClaimRole,
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    Reference,
    Tool,
)
from benchmarking_pipeline.services.llm_judge import OpenAIJudge, ProgressJudge
from benchmarking_pipeline.services.structured_chat import OpenAIChat

if __name__ == "__main__":
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else _EXAMPLE_HYPOTHESIS
    hypothesis_text = resolve_hypothesis_text(raw_arg)

    chat = OpenAIChat("gpt-4o-mini")
    raw_claims = extract_claims(hypothesis_text, chat)

    claims = [
        Claim(
            text=c["text"],
            role=ClaimRole(c["role"]),
            references=[Reference(raw=r["raw"]) for r in c.get("references", [])],
        )
        for c in raw_claims
    ]

    #print(f"Hypothesis:\n  {hypothesis_text}\n")
    print(f"Decomposed into {len(claims)} claim(s):")
    for i, c in enumerate(claims, start=1):
        print(f"  {i}. [{c.role.value}] {c.text}")
    print()

    hyp = Hypothesis(id="h1", text=hypothesis_text, rank=1, claims=claims)
    run = EvaluationRun(tool=Tool(name="practice"), prompt="p",
                        outputs=HypothesisSet(hypotheses=[hyp]))

    # Expected judge calls: one role-classification per claim missing a role
    # (0 here, since extraction already set it) + one edge check per pair.
    n_roles_needed = sum(1 for c in claims if c.role is None)
    expected_calls = n_roles_needed + comb(len(claims), 2)
    judge = ProgressJudge(OpenAIJudge("gpt-4o-mini"), total=expected_calls, stream=sys.stdout)
    ctx = Context(config=RunConfig(), judge=judge)

    print(f"Checking logical consistency ({expected_calls} judge call(s) expected)...")
    result = LogicalConsistency().score(hyp, run, ctx)
    print(f"logical_consistency score: {result.score}  ({result.anchor})")
    print()
    print("Pairwise checks:")
    for edge in result.evidence.get("edges_checked", []):
        i, j, verdict = edge["i"], edge["j"], edge["verdict"]
        print(f"  claim {i+1} vs claim {j+1}: {verdict}")
    if result.evidence.get("contradiction_pairs"):
        print()
        print("Contradictions found:")
        for pair in result.evidence["contradiction_pairs"]:
            print(f"  [{pair['severity']}] \"{pair['claim_i']}\"  <-->  \"{pair['claim_j']}\"")
