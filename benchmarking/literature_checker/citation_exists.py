import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from .citation_extractor import Reference

_CROSSREF_BASE = "https://api.crossref.org/works"
_S2_SEARCH = "https://api.semanticscholar.org/graph/v1/paper/search"
_MAILTO = "snsen@path.org"

# CrossRef bibliographic search scores: reliable matches typically score 50–80,
# hallucinated/unrelated papers score < 30.
_SCORE_SCALE = 50.0
_EXISTS_THRESHOLD = 0.7
# Also try Semantic Scholar when CrossRef confidence is below this, even if it
# cleared the exists threshold — catches borderline cases and grey literature.
_S2_FALLBACK_BELOW = 0.85


@dataclass
class VerificationResult:
    reference: Reference
    exists: bool
    confidence: float  # 0.0–1.0
    matched_title: str | None = None
    matched_doi: str | None = None
    matched_year: int | None = None
    year_mismatch: bool = False  # true when extracted year != matched year (e.g. reprints)
    source: str = "crossref"


def _parse_year(work: dict) -> int | None:
    parts = work.get("published", {}).get("date-parts") or [[None]]
    return parts[0][0]


def _lookup_doi(doi: str, ref: Reference, client: httpx.Client) -> VerificationResult | None:
    """Exact DOI lookup. Returns None on unexpected errors (falls through to search)."""
    resp = client.get(f"{_CROSSREF_BASE}/{doi}", params={"mailto": _MAILTO})
    if resp.status_code == 404:
        return VerificationResult(reference=ref, exists=False, confidence=1.0)
    if resp.status_code != 200:
        return None
    work = resp.json()["message"]
    return VerificationResult(
        reference=ref,
        exists=True,
        confidence=1.0,
        matched_title=(work.get("title") or [None])[0],
        matched_doi=doi,
        matched_year=_parse_year(work),
    )


def _jaccard(a: str, b: str) -> float:
    """Token-overlap similarity between two title strings."""
    def tokens(s: str) -> set[str]:
        return set(re.sub(r'[^\w\s]', '', s.lower()).split())
    a_tok, b_tok = tokens(a), tokens(b)
    if not a_tok or not b_tok:
        return 0.0
    return len(a_tok & b_tok) / len(a_tok | b_tok)


# Only report a matched DOI when the returned title is this similar to the query.
# Prevents leaking a wrong paper's DOI for borderline/hallucinated matches.
_DOI_TITLE_SIM_THRESHOLD = 0.8


def _search_bibliographic(ref: Reference, client: httpx.Client) -> VerificationResult:
    """Fuzzy bibliographic search using the full raw citation string."""
    resp = client.get(
        _CROSSREF_BASE,
        params={"query.bibliographic": ref.raw, "rows": 3, "mailto": _MAILTO},
    )
    resp.raise_for_status()
    items = resp.json()["message"]["items"]

    if not items:
        return VerificationResult(reference=ref, exists=False, confidence=0.9)

    best = items[0]
    cr_score: float = best.get("score", 0.0)
    matched_title = (best.get("title") or [None])[0]
    matched_year = _parse_year(best)

    confidence = min(cr_score / _SCORE_SCALE, 1.0)
    year_mismatch = bool(ref.year and matched_year and ref.year != matched_year)
    title_sim = _jaccard(ref.raw, matched_title or "")
    matched_doi = best.get("DOI") if title_sim >= _DOI_TITLE_SIM_THRESHOLD else None

    return VerificationResult(
        reference=ref,
        exists=confidence >= _EXISTS_THRESHOLD,
        confidence=round(confidence, 3),
        matched_title=matched_title,
        matched_doi=matched_doi,
        matched_year=matched_year,
        year_mismatch=year_mismatch,
    )


def _search_semantic_scholar(ref: Reference, client: httpx.Client) -> VerificationResult | None:
    """Semantic Scholar title search. Returns None on API errors."""
    try:
        resp = client.get(
            _S2_SEARCH,
            params={"query": ref.raw, "fields": "title,year,externalIds", "limit": 3},
        )
    except httpx.HTTPError:
        return None
    if resp.status_code != 200:
        return None

    items = resp.json().get("data", [])
    if not items:
        return VerificationResult(reference=ref, exists=False, confidence=0.9, source="semantic_scholar")

    best = max(items, key=lambda x: _jaccard(ref.raw, x.get("title") or ""))
    sim = _jaccard(ref.raw, best.get("title") or "")
    matched_year = best.get("year")
    matched_doi = (best.get("externalIds") or {}).get("DOI")
    year_mismatch = bool(ref.year and matched_year and ref.year != matched_year)

    return VerificationResult(
        reference=ref,
        exists=sim >= _EXISTS_THRESHOLD,
        confidence=round(sim, 3),
        matched_title=best.get("title"),
        matched_doi=matched_doi,
        matched_year=matched_year,
        year_mismatch=year_mismatch,
        source="semantic_scholar",
    )


def verify_reference(ref: Reference, client: httpx.Client | None = None) -> VerificationResult:
    """Verify a single reference. Passes an existing client to avoid per-call connection overhead."""
    own_client = client is None
    if own_client:
        client = httpx.Client(timeout=10.0)
    try:
        if ref.doi:
            result = _lookup_doi(ref.doi, ref, client)
            if result is not None:
                return result
        result = _search_bibliographic(ref, client)
        if result.confidence < _S2_FALLBACK_BELOW:
            s2 = _search_semantic_scholar(ref, client)
            if s2 is not None and s2.confidence > result.confidence:
                return s2
        return result
    finally:
        if own_client:
            client.close()


def verify_references(refs: list[Reference]) -> list[VerificationResult]:
    """Verify a list of references, reusing a single HTTP connection."""
    with httpx.Client(timeout=10.0) as client:
        return [verify_reference(ref, client) for ref in refs]


def write_results_json(results: list[VerificationResult], path: str | Path) -> None:
    """Write verification results to a JSON file."""
    rows = []
    for i, r in enumerate(results, start=1):
        d = asdict(r)
        rows.append({
            "index": i,
            "query": d["reference"]["raw"],
            "exists": d["exists"],
            "confidence": d["confidence"],
            "source": d["source"],
            "matched_title": d["matched_title"],
            "matched_doi": d["matched_doi"],
            "matched_year": d["matched_year"],
            "year_mismatch": d["year_mismatch"],
        })
    Path(path).write_text(json.dumps(rows, indent=2))
