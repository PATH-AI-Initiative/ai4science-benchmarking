"""Literature-lookup clients used by the Accuracy axis (citation checking) and
the Novelty axis (nearest-neighbour distance against a literature corpus).

``CrossRefClient`` and ``SemanticScholarClient`` are free, no-API-key public
APIs; ``CompositeLiteratureClient`` tries CrossRef first (DOI-exact, then
bibliographic search) and falls back to Semantic Scholar when CrossRef's
confidence is low — this catches grey literature/preprints CrossRef alone
sometimes misses. All three satisfy ``LiteratureClient`` so metrics can be
tested against a fake instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


@dataclass
class PaperRecord:
    title: str | None
    doi: str | None
    year: int | None
    match_score: float  # source-specific confidence that this is the queried paper
    abstract: str | None = None  # used to check whether the source supports a claim


@runtime_checkable
class LiteratureClient(Protocol):
    name: str

    def lookup_doi(self, doi: str) -> PaperRecord | None:
        """Exact DOI resolution, or ``None`` if not found."""
        ...

    def search(self, query: str, limit: int = 3) -> list[PaperRecord]:
        """Bibliographic search, best match first."""
        ...


def _jaccard(a: str, b: str) -> float:
    """Token-overlap similarity between two title strings."""
    def tokens(s: str) -> set[str]:
        return set(re.sub(r"[^\w\s]", "", s.lower()).split())
    a_tok, b_tok = tokens(a), tokens(b)
    if not a_tok or not b_tok:
        return 0.0
    return len(a_tok & b_tok) / len(a_tok | b_tok)


# CrossRef bibliographic-search relevance scores are unbounded; empirically,
# scores at or above this scale correspond to a confident match.
_CROSSREF_SCORE_SCALE = 50.0

# Genuine primary-literature CrossRef work types. Commentary/wrapper records
# (e.g. Faculty Opinions "recommendation of <title>" entries, typed "dataset"
# or "peer-review" by CrossRef) embed the real paper's title verbatim and can
# out-score the actual paper on both CrossRef's relevance score and simple
# text similarity -- record type is the signal that actually distinguishes
# them, so primary types are preferred over any-type-goes ranking.
_PRIMARY_WORK_TYPES = frozenset({
    "journal-article", "proceedings-article", "posted-content",
    "book-chapter", "monograph", "report", "dissertation",
})


def _crossref_year(work: dict) -> int | None:
    for key in ("published", "published-print", "published-online", "issued"):
        date_parts = (work.get(key) or {}).get("date-parts")
        if date_parts and date_parts[0] and date_parts[0][0]:
            return date_parts[0][0]
    return None


_JATS_TAG_RE = re.compile(r"<[^>]+>")


def _crossref_abstract(work: dict) -> str | None:
    """CrossRef abstracts, where present, are JATS-XML-wrapped (e.g.
    ``<jats:p>...</jats:p>``); coverage is publisher-dependent and spotty."""
    raw = work.get("abstract")
    if not raw:
        return None
    return _JATS_TAG_RE.sub("", raw).strip() or None


class CrossRefClient:
    """CrossRef's public works API. Free, no API key required.

    ``mailto`` identifies the caller per CrossRef's "polite pool" etiquette —
    higher rate limits, no functional effect otherwise.
    """

    name = "crossref"
    _BASE = "https://api.crossref.org/works"

    def __init__(
        self, mailto: str | None = None, timeout: float = 10.0,
        client: httpx.Client | None = None,
    ):
        self._mailto = mailto
        self._client = client or httpx.Client(timeout=timeout)

    def lookup_doi(self, doi: str) -> PaperRecord | None:
        params = {"mailto": self._mailto} if self._mailto else {}
        try:
            resp = self._client.get(f"{self._BASE}/{doi}", params=params)
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        work = resp.json()["message"]
        return PaperRecord(
            title=(work.get("title") or [None])[0],
            doi=doi,
            year=_crossref_year(work),
            match_score=1.0,  # exact DOI resolution
            abstract=_crossref_abstract(work),
        )

    def search(self, query: str, limit: int = 3) -> list[PaperRecord]:
        # Over-fetch: the top-scoring raw results can be entirely wrapper/
        # commentary records (see _PRIMARY_WORK_TYPES), so we need enough
        # candidates for a primary-typed match to have a chance to surface.
        params = {"query.bibliographic": query, "rows": max(limit * 5, 10)}
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            resp = self._client.get(self._BASE, params=params)
            resp.raise_for_status()
        except httpx.HTTPError:
            return []
        items = resp.json()["message"]["items"]
        items.sort(key=lambda item: (
            item.get("type") not in _PRIMARY_WORK_TYPES,  # primary types first
            -item.get("score", 0.0),  # then by CrossRef's own relevance score
        ))
        return [
            PaperRecord(
                title=(item.get("title") or [None])[0],
                doi=item.get("DOI"),
                year=_crossref_year(item),
                match_score=min(item.get("score", 0.0) / _CROSSREF_SCORE_SCALE, 1.0),
                abstract=_crossref_abstract(item),
            )
            for item in items[:limit]
        ]


class SemanticScholarClient:
    """Semantic Scholar's Graph API. Free for the public rate-limited tier."""

    name = "semantic_scholar"
    _BASE = "https://api.semanticscholar.org/graph/v1/paper"
    _FIELDS = "title,year,externalIds,abstract"

    def __init__(self, timeout: float = 10.0, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=timeout)

    def lookup_doi(self, doi: str) -> PaperRecord | None:
        try:
            resp = self._client.get(f"{self._BASE}/DOI:{doi}", params={"fields": self._FIELDS})
        except httpx.HTTPError:
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        return PaperRecord(
            title=data.get("title"),
            doi=(data.get("externalIds") or {}).get("DOI", doi),
            year=data.get("year"),
            match_score=1.0,
            abstract=data.get("abstract"),
        )

    def search(self, query: str, limit: int = 3) -> list[PaperRecord]:
        try:
            resp = self._client.get(
                f"{self._BASE}/search",
                params={"query": query, "fields": self._FIELDS, "limit": limit},
            )
        except httpx.HTTPError:
            return []
        if resp.status_code != 200:
            return []
        items = resp.json().get("data", [])
        results = [
            PaperRecord(
                title=item.get("title"),
                doi=(item.get("externalIds") or {}).get("DOI"),
                year=item.get("year"),
                match_score=_jaccard(query, item.get("title") or ""),
                abstract=item.get("abstract"),
            )
            for item in items
        ]
        results.sort(key=lambda r: r.match_score, reverse=True)
        return results[:limit]


class CompositeLiteratureClient:
    """CrossRef primary, Semantic Scholar fallback when CrossRef's confidence
    is low. Mirrors the two-source verification strategy that catches grey
    literature/preprints CrossRef alone sometimes misses.
    """

    name = "crossref+semantic_scholar"

    def __init__(
        self,
        mailto: str | None = None,
        fallback_below: float = 0.85,
        crossref: LiteratureClient | None = None,
        s2: LiteratureClient | None = None,
    ):
        self._crossref = crossref or CrossRefClient(mailto=mailto)
        self._s2 = s2 or SemanticScholarClient()
        self._fallback_below = fallback_below

    def lookup_doi(self, doi: str) -> PaperRecord | None:
        record = self._crossref.lookup_doi(doi)
        if record is not None:
            return record
        return self._s2.lookup_doi(doi)

    def search(self, query: str, limit: int = 3) -> list[PaperRecord]:
        cr_results = self._crossref.search(query, limit=limit)
        best_cr = cr_results[0] if cr_results else None
        if best_cr is not None and best_cr.match_score >= self._fallback_below:
            return cr_results

        s2_results = self._s2.search(query, limit=limit)
        combined = cr_results + s2_results
        if not combined:
            return cr_results
        combined.sort(key=lambda r: r.match_score, reverse=True)
        return combined[:limit]
