"""Practice script: extract the claims from a single hypothesis's text.

Isolates just the "decompose one hypothesis into claims" step from the real
extraction pipeline (benchmarking_pipeline/io/extraction.py), which normally
does this as part of a bigger job: find all the hypotheses in a whole
document, THEN decompose each one into claims. Here we skip straight to the
second part, given one hypothesis's text directly.

Run from the project root with:
    uv run python examples/practice/extract_claims_practice.py
    uv run python examples/practice/extract_claims_practice.py "Your hypothesis text here"
    uv run python examples/practice/extract_claims_practice.py path/to/one_hypothesis.pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarking_pipeline.core.models import ClaimRole
from benchmarking_pipeline.io.readers import read_document_text
from benchmarking_pipeline.services.structured_chat import OpenAIChat

_ROLE_VALUES = [r.value for r in ClaimRole]

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": _ROLE_VALUES,
                        "description": (
                            "premise: a foundational fact taken as given. "
                            "mechanistic_step: an explanation of how/why something happens. "
                            "prediction: a testable outcome the hypothesis implies. "
                            "background_assumption: contextual framing not central to the argument."
                        ),
                    },
                    "references": {
                        "type": "array",
                        "description": "Citations attached to this claim, verbatim as they appear",
                        "items": {
                            "type": "object",
                            "properties": {"raw": {"type": "string"}},
                            "required": ["raw"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "role", "references"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_SYSTEM_PROMPT = (
    "You extract structured data from documents precisely and conservatively. "
    "Never invent content that is not present in the source text."
)

_PROMPT_TEMPLATE = """\
Below is a single hypothesis from a co-scientist tool's output. Break it down
into the individual factual claims it rests on, tag each claim with its role
(premise, mechanistic_step, prediction, or background_assumption), and attach
any citation given for each claim (verbatim). If the hypothesis has no clearly
separable sub-claims, return it as a single claim whose text is the hypothesis
statement itself.

--- HYPOTHESIS ---
{hypothesis_text}
--- END HYPOTHESIS ---
"""


def extract_claims(hypothesis_text: str, chat) -> list[dict]:
    data = chat.complete(
        system=_SYSTEM_PROMPT,
        user=_PROMPT_TEMPLATE.format(hypothesis_text=hypothesis_text),
        schema=_CLAIMS_SCHEMA,
        max_tokens=2048,
    )
    return data["claims"]


_EXAMPLE_HYPOTHESIS = (
    "Kelch13 propeller-domain mutations, analogous to those documented in P. "
    "falciparum, confer artemisinin resistance in P. knowlesi through reduced "
    "ring-stage susceptibility, and this resistance is expected to correlate "
    "with increased treatment failure rates observed in field surveillance."
)


def resolve_hypothesis_text(arg: str) -> str:
    """If ``arg`` is a path to an existing .pdf/.docx, read its text.
    Otherwise treat ``arg`` as the literal hypothesis text."""
    path = Path(arg)
    if path.suffix.lower() in (".pdf", ".docx") and path.exists():
        return read_document_text(path).strip()
    return arg


if __name__ == "__main__":
    raw_arg = sys.argv[1] if len(sys.argv) > 1 else _EXAMPLE_HYPOTHESIS
    hypothesis_text = resolve_hypothesis_text(raw_arg)

    chat = OpenAIChat("gpt-4o-mini")
    claims = extract_claims(hypothesis_text, chat)

    print(f"Hypothesis:\n  {hypothesis_text}\n")
    print(f"Extracted {len(claims)} claim(s):\n")
    for i, c in enumerate(claims, start=1):
        print(f"{i}. [{c['role']}] {c['text']}")
        for ref in c.get("references", []):
            print(f"     ref: {ref['raw']}")
