"""Parse captured tool output into a HypothesisSet.

Tools return hypotheses in varied formats (free text, structured summaries,
ranked lists). Each format gets a parser here; all produce the same
:class:`HypothesisSet` so downstream metrics are format-agnostic.

A minimal JSON parser makes the pipeline runnable. Expected shape::

    {
      "raw_text": "...",
      "hypotheses": [
        {"id": "h1", "text": "...", "rank": 1, "category": "...",
         "claims": [{"text": "...", "role": "premise",
                     "references": [{"raw": "..."}],
                     "entities": [{"name": "...", "kind": "gene"}]}],
         "self_reported": {"novelty_tier": "...", "feasibility_tier": "...", ...}}
      ]
    }

``role`` is optional -- logical-consistency classifies it lazily via the
judge when absent. ``category``, ``entities``, and ``self_reported`` are
optional too; ``self_reported`` (a tool's own novelty/feasibility
self-assessment) lands in ``Hypothesis.metadata``, not ``claims``, since it's
the tool's own judgment rather than a checkable claim. Top-level ``raw_text``
is also optional -- see :func:`parse_raw_text`.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..core.models import Claim, ClaimRole, Entity, Hypothesis, HypothesisSet, Reference


def _parse_role(raw_role: str | None) -> ClaimRole | None:
    if raw_role is None:
        return None
    try:
        return ClaimRole(raw_role)
    except ValueError:
        return None


def _parse_reference(r) -> Reference:
    if not isinstance(r, dict):
        return Reference(raw=r)
    year_raw = r.get("year")
    year = int(year_raw) if isinstance(year_raw, str) and year_raw.strip().isdigit() else year_raw
    return Reference(
        raw=r["raw"],
        title=r.get("title") or None,
        doi=r.get("doi") or None,
        year=year if isinstance(year, int) else None,
        authors=r.get("authors") or [],
    )


def _parse_metadata(h: dict) -> dict:
    """Carries through free-form tool-reported fields (e.g. ``self_reported``)
    without interpreting them -- empty/blank ones are dropped so ``metadata``
    stays empty for tools that don't report anything of this kind."""
    self_reported = h.get("self_reported")
    if not self_reported or not any(self_reported.values()):
        return {}
    return {"self_reported": self_reported}


def parse_dict(data: dict) -> HypothesisSet:
    hypotheses = []
    for i, h in enumerate(data.get("hypotheses", []), start=1):
        claims = [
            Claim(
                text=c["text"],
                role=_parse_role(c.get("role")),
                references=[_parse_reference(r) for r in c.get("references", [])],
                entities=[Entity(**e) if isinstance(e, dict) else Entity(name=e)
                          for e in c.get("entities", [])],
            )
            for c in h.get("claims", [])
        ]
        hypotheses.append(
            Hypothesis(
                id=str(h.get("id", f"h{i}")),
                text=h["text"],
                rank=h.get("rank", i),
                category=h.get("category") or None,
                claims=claims,
                raw=h.get("raw"),
                metadata=_parse_metadata(h),
            )
        )
    return HypothesisSet(hypotheses=hypotheses)


def parse_json(path: str | Path) -> HypothesisSet:
    return parse_dict(json.loads(Path(path).read_text()))


def parse_raw_text(path: str | Path) -> str | None:
    """The capture's top-level ``raw_text``, if present (populated by
    :func:`~benchmarking_pipeline.io.extraction.extract_hypotheses`; absent in
    older captures or hand-written ones). Separate from :func:`parse_json` so
    that function's return type stays a plain ``HypothesisSet``."""
    return json.loads(Path(path).read_text()).get("raw_text")
