"""LLM-assisted structured extraction from raw tool-report text.

Co-scientist tools export reports in wildly different shapes (prose
paragraphs, numbered lists, section headers), so a single rule-based parser
can't cover them. This module asks a model to decompose the raw text into the
same schema :func:`~benchmarking_pipeline.io.parsers.parse_dict` expects:
ranked hypotheses, each broken into claims with their cited references.

Runs as two passes rather than one big structured-output call:

1. **List pass** — identify the hypotheses (text/category/self-reported
   assessment), no claim decomposition yet. Cheap: proportional to the
   document's hypothesis *titles*, not their full argument text.
2. **Claims pass** — one focused call per hypothesis, asked to find that one
   hypothesis in the full document and decompose only its argument into
   claims/references/entities.

A single call asked to do both across every hypothesis at once was tried
first and silently under-delivers on documents with many rich hypotheses: a
JSON-schema-constrained model can hit its output budget and still emit valid,
complete-looking JSON by simply closing the hypotheses array early -- so a
15-hypothesis report can come back with 2-4 hypotheses and no error at all.
Splitting into one small call per hypothesis keeps each call's output bounded
regardless of how many hypotheses the document has, at the cost of N+1 calls
instead of 1 -- worthwhile for a draft that gets reviewed before use anyway.

The output of :func:`extract_hypotheses` is a plain dict, meant to be written
to disk and reviewed/edited by a human before it is used for scoring — this
step produces the record the Accuracy axis will later check for hallucinated
claims and citations, so an unreviewed extraction defeats the point of the
benchmark. Treat it as a first draft, not a capture.

Any :class:`~benchmarking_pipeline.services.structured_chat.StructuredChatClient`
works here — Anthropic, OpenAI, or a local Ollama model — so extraction can run
entirely offline if that's what the workflow needs.
"""

from __future__ import annotations

from ..core.models import ClaimRole
from ..services.structured_chat import StructuredChatClient

_ROLE_VALUES = [r.value for r in ClaimRole]

_SELF_REPORTED_SCHEMA = {
    "type": "object",
    "description": (
        "The tool's own labeled self-assessment of this hypothesis, if it "
        "gives one (e.g. explicit 'Novelty' or 'Feasibility' sections). "
        "These are the tool's judgments, not factual claims -- do not put "
        "them in `claims`. Leave every field empty ('' or []) if the source "
        "doesn't include this kind of self-assessment."
    ),
    "properties": {
        "novelty_tier": {
            "type": "string",
            "description": "The tool's own novelty label, verbatim (e.g. 'First-in-class', 'Novel for Pk').",
        },
        "novelty_rationale": {
            "type": "string",
            "description": "The tool's justification for that novelty label.",
        },
        "feasibility_tier": {
            "type": "string",
            "description": "The tool's own feasibility/tractability label, verbatim (e.g. 'High', 'Moderate').",
        },
        "feasibility_rationale": {
            "type": "string",
            "description": "The tool's justification for that feasibility label.",
        },
        "key_risks": {
            "type": "array",
            "description": "Risks/caveats the tool itself flagged for this hypothesis.",
            "items": {"type": "string"},
        },
    },
    "required": [
        "novelty_tier", "novelty_rationale",
        "feasibility_tier", "feasibility_rationale", "key_risks",
    ],
    "additionalProperties": False,
}

_HYPOTHESIS_LIST_SCHEMA = {
    "type": "object",
    "properties": {
        "hypotheses": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "rank": {"type": "integer", "description": "1 = the tool's top-ranked hypothesis"},
                    "text": {
                        "type": "string",
                        "description": (
                            "The hypothesis's own statement/title, verbatim or lightly cleaned -- "
                            "specific enough to unambiguously identify its section in the source "
                            "text (a later pass looks it up again by this text)."
                        ),
                    },
                    "category": {
                        "type": "string",
                        "description": (
                            "The section/category this hypothesis is grouped under, verbatim "
                            "from the source (e.g. a heading like 'Category 1: ...'). Empty "
                            "string if the tool presents a flat ranked list with no grouping."
                        ),
                    },
                    "self_reported": _SELF_REPORTED_SCHEMA,
                },
                "required": ["rank", "text", "category", "self_reported"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["hypotheses"],
    "additionalProperties": False,
}

_CLAIM_ITEM_SCHEMA = {
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
            "description": (
                "Citations attached to this claim. `raw` is the citation "
                "marker verbatim as it appears (e.g. '[107]' or 'Smith et "
                "al. 2020'). If the document includes a resolving "
                "bibliography (e.g. a numbered reference list mapping [N] "
                "to a real paper), look up this citation there and fill in "
                "title/doi/year/authors from the matching entry. Never "
                "invent these -- leave each one as an empty string/array "
                "if no resolving bibliography entry exists for this citation."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "raw": {"type": "string"},
                    "title": {"type": "string"},
                    "doi": {"type": "string"},
                    "year": {
                        "type": "string",
                        "description": "Publication year as a string, e.g. '2025'.",
                    },
                    "authors": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["raw", "title", "doi", "year", "authors"],
                "additionalProperties": False,
            },
        },
        "entities": {
            "type": "array",
            "description": (
                "Named biological entities this claim references (genes, "
                "proteins, pathways, drug targets), verbatim -- e.g. a gene "
                "symbol or database ID like 'PKNH_1436200'."
            ),
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {
                        "type": "string",
                        "description": "e.g. 'gene', 'protein', 'pathway'. Empty string if unclear.",
                    },
                },
                "required": ["name", "kind"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["text", "role", "references", "entities"],
    "additionalProperties": False,
}

_CLAIMS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "description": "The hypothesis decomposed into its constituent factual claims",
            "items": _CLAIM_ITEM_SCHEMA,
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

_EXTRACTION_SYSTEM_PROMPT = (
    "You extract structured data from documents precisely and conservatively. "
    "Never invent content that is not present in the source text."
)

_HYPOTHESIS_LIST_PROMPT = """\
You are preparing a co-scientist tool's raw output for a benchmarking pipeline.
Below is the raw text exported from the tool. Identify the distinct hypotheses
it proposes, in the order/ranking the tool presented them, WITHOUT decomposing
their claims yet (a later pass handles that per hypothesis) -- just identify
each one and its top-level metadata.

If the tool groups hypotheses under section headings (e.g. categories or
themes rather than a flat rank), record that heading verbatim as `category`.

If the tool gives its own labeled self-assessment of a hypothesis (e.g.
explicit "Novelty" or "Feasibility" sections/ratings, or a list of risks it
flags), capture that verbatim in `self_reported` — these are the tool's own
judgments about itself, not factual claims. Leave `self_reported` fields empty
('' or []) if the source has nothing of this kind for a given hypothesis.

Do not invent hypotheses that are not present in the text.

--- RAW TOOL OUTPUT ---
{raw_text}
--- END RAW TOOL OUTPUT ---
"""

_CLAIMS_PROMPT = """\
Below is the full raw text of a co-scientist tool's report. Focus ONLY on the
hypothesis identified as:

    "{hypothesis_text}"

Find that hypothesis's section in the text below and break its argument
(mechanistic rationale, and any target/feasibility/novelty/risk discussion —
excluding a labeled self-assessment already captured elsewhere) down into the
individual factual claims it rests on. Tag each claim with its role in the
argument (premise, mechanistic_step, prediction, or background_assumption),
attach any citation given for each claim (verbatim — do not normalize or
invent citation formats), and list any named biological entities (genes,
proteins, pathways) the claim mentions. Do not invent claims, references, or
entities that are not present in the text. If the hypothesis has no clearly
separable sub-claims, return it as a single claim whose text is the hypothesis
statement itself.

If the document has a numbered reference list (a bibliography resolving
citation markers like [107] to actual papers), resolve every inline citation
against it and populate that reference's title/doi/year/authors from the
matching bibliography entry — do this by matching the bracketed number, not by
guessing from context. If no such bibliography exists, or a citation number
has no matching entry, leave those fields empty rather than inventing them.

--- FULL RAW TOOL OUTPUT ---
{raw_text}
--- END FULL RAW TOOL OUTPUT ---
"""


def extract_claims(
    hypothesis_text: str, raw_text: str, chat: StructuredChatClient, *, max_tokens: int = 8192,
) -> list[dict]:
    """Decompose one hypothesis's argument into claims/references/entities.

    ``hypothesis_text`` identifies which hypothesis within the full ``raw_text``
    to focus on (see :data:`_CLAIMS_PROMPT`). Exposed on its own, separate from
    :func:`extract_hypotheses`, so a single hypothesis's decomposition can be
    redone in isolation -- e.g. retrying with a stronger model after a cheaper
    one collapsed it into one claim or dropped its citations.
    """
    claims_data = chat.complete(
        system=_EXTRACTION_SYSTEM_PROMPT,
        user=_CLAIMS_PROMPT.format(hypothesis_text=hypothesis_text, raw_text=raw_text),
        schema=_CLAIMS_SCHEMA,
        max_tokens=max_tokens,
    )
    return claims_data["claims"]


def extract_hypotheses(raw_text: str, chat: StructuredChatClient, *, max_tokens: int = 8192) -> dict:
    """Ask ``chat`` to decompose raw tool-report text into the hypotheses/claims schema.

    Returns a plain dict matching the shape ``io.parsers.parse_dict`` expects.
    This is a draft — review it (and fix any misattributed claim or dropped
    citation) before using it as a captured run.

    Makes 1 + len(hypotheses) calls to ``chat`` (see module docstring for why).
    """
    list_data = chat.complete(
        system=_EXTRACTION_SYSTEM_PROMPT,
        user=_HYPOTHESIS_LIST_PROMPT.format(raw_text=raw_text),
        schema=_HYPOTHESIS_LIST_SCHEMA,
        max_tokens=max_tokens,
    )
    hypotheses = list_data["hypotheses"]

    for hyp in hypotheses:
        hyp["claims"] = extract_claims(hyp["text"], raw_text, chat, max_tokens=max_tokens)

    return {"hypotheses": hypotheses}
