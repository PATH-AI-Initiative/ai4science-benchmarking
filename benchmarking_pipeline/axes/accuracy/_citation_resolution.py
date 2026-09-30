"""Shared citation-resolution step for ``citation_accuracy`` and
``citation_support``.

Both metrics need the same existence + title-match check before doing
anything metric-specific: accuracy scores existence itself; support checks
whether the resolved paper's abstract backs its claim. Resolving
independently in each would hit the literature API and title-match judge
call twice per citation, so results are cached on ``ctx.citation_cache``,
keyed by the reference's identity (DOI, or else title/raw text) -- safe
regardless of which metric runs first, since both read/fill the cache the
same way.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...core.context import Context
from ...core.models import Reference
from ...services.relationship_judge import judge_relationship

# "different_paper" listed last: judge_relationship's fallback-on-unparseable-
# verdict, and the conservative choice when a fuzzy bibliographic-search hit's
# actual relevance is ambiguous -- a citation isn't "verified" on a shrug.
_TITLE_MATCH_CHOICES = ["same_paper", "different_paper"]

_TITLE_MATCH_INSTRUCTIONS = """\
A bibliographic search for a cited reference returned this candidate. Search \
relevance ranking is based on shared wording, not topic -- a generic phrase \
like "phase 3 trial" or "meets primary endpoint" can make an unrelated paper \
(different drug, different disease) score as a strong textual match. Choose \
exactly one:
- same_paper: the candidate is clearly about the same subject as the cited \
reference (same drug/gene/finding), consistent with actually being that paper.
- different_paper: the candidate is about a different subject despite \
wording overlap, or there isn't enough shared subject matter to tell.
Default to different_paper unless clearly the same paper.
"""


def _judge_title_match(judge, cited_reference: str, candidate_title: str) -> str:
    return judge_relationship(
        judge,
        label_a="Cited reference", text_a=cited_reference,
        label_b="Search result title", text_b=candidate_title,
        instructions=_TITLE_MATCH_INSTRUCTIONS,
        choices=_TITLE_MATCH_CHOICES,
    )


@dataclass
class ResolvedCitation:
    exists: bool
    matched_title: str | None
    title_match: str | None
    abstract: str | None


def resolve_citation(ref: Reference, ctx: Context) -> ResolvedCitation:
    """Existence + title-match check for one reference, cached on ``ctx``."""
    key = ref.doi or ref.title or ref.raw
    cached = ctx.citation_cache.get(key)
    if cached is not None:
        return cached

    record = None
    is_exact_doi = False
    if ref.doi:
        record = ctx.literature.lookup_doi(ref.doi)
        is_exact_doi = record is not None
    if record is None:
        hits = ctx.literature.search(ref.title or ref.raw, limit=1)
        record = hits[0] if hits else None
    matched = record is not None and record.match_score >= 0.7

    title_match = None
    if matched and not is_exact_doi and ctx.judge is not None and record.title:
        title_match = _judge_title_match(ctx.judge, ref.title or ref.raw, record.title)
    exists = matched and (title_match is None or title_match == "same_paper")

    resolved = ResolvedCitation(
        exists=exists,
        matched_title=record.title if record else None,
        title_match=title_match,
        abstract=record.abstract if (exists and record) else None,
    )
    ctx.citation_cache[key] = resolved
    return resolved
