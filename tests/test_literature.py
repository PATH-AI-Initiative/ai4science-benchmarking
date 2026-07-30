"""Tests for the literature clients: request/response parsing via
``httpx.MockTransport`` (no real network call), and the composite client's
fallback logic via simple fake ``LiteratureClient`` stand-ins.
"""

from __future__ import annotations

import httpx

from benchmarking_pipeline.services.literature import (
    CompositeLiteratureClient,
    CrossRefClient,
    PaperRecord,
    SemanticScholarClient,
)


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class FakeLiteratureClient:
    """Minimal LiteratureClient stand-in for testing composite fallback logic."""

    def __init__(self, name: str, doi_result=None, search_results=None):
        self.name = name
        self._doi_result = doi_result
        self._search_results = search_results or []

    def lookup_doi(self, doi: str):
        return self._doi_result

    def search(self, query: str, limit: int = 3):
        return self._search_results[:limit]


# ---- CrossRefClient ----

def test_crossref_lookup_doi_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/works/10.1038/nature12876"
        return httpx.Response(200, json={
            "message": {
                "title": ["A molecular marker of artemisinin-resistant malaria"],
                "published": {"date-parts": [[2014]]},
            }
        })

    client = CrossRefClient(client=_mock_client(handler))
    record = client.lookup_doi("10.1038/nature12876")

    assert record.title == "A molecular marker of artemisinin-resistant malaria"
    assert record.year == 2014
    assert record.match_score == 1.0


def test_crossref_lookup_doi_not_found():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    client = CrossRefClient(client=_mock_client(handler))
    assert client.lookup_doi("10.9999/fake") is None


def test_crossref_search_scales_score_and_extracts_fields():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {
                "items": [
                    {"title": ["Best match"], "DOI": "10.1/best", "score": 45.0,
                    "published-print": {"date-parts": [[2020]]}},
                    {"title": ["Worse match"], "DOI": "10.1/worse", "score": 10.0,
                    "issued": {"date-parts": [[2019]]}},
                ]
            }
        })

    client = CrossRefClient(client=_mock_client(handler))
    results = client.search("some query", limit=2)

    assert len(results) == 2
    assert results[0].title == "Best match"
    assert results[0].match_score == 0.9  # 45 / 50
    assert results[0].year == 2020
    assert results[1].match_score == 0.2  # 10 / 50


def test_crossref_search_prefers_primary_type_over_higher_scoring_wrapper_record():
    """Regression test: Faculty Opinions "recommendation of X" records are
    typed "dataset" by CrossRef, embed X's title verbatim, and can out-score
    the real paper X on CrossRef's own relevance metric. A lower-scoring but
    genuinely primary-typed record must still win.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "message": {
                "items": [
                    {"type": "dataset", "title": ["Faculty Opinions recommendation of Real Paper"],
                    "DOI": "10.3410/wrapper", "score": 57.8},
                    {"type": "journal-article", "title": ["Real Paper"],
                    "DOI": "10.1038/real", "score": 40.0,
                    "published": {"date-parts": [[2014]]}},
                ]
            }
        })

    client = CrossRefClient(client=_mock_client(handler))
    results = client.search("Real Paper", limit=1)

    assert results[0].title == "Real Paper"
    assert results[0].doi == "10.1038/real"


def test_crossref_search_empty_results():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"items": []}})

    client = CrossRefClient(client=_mock_client(handler))
    assert client.search("nothing matches this") == []


def test_crossref_search_http_error_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = CrossRefClient(client=_mock_client(handler))
    assert client.search("query") == []


# ---- SemanticScholarClient ----

def test_semantic_scholar_lookup_doi_success():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "DOI:10.1038/nature12876" in request.url.path
        return httpx.Response(200, json={
            "title": "A molecular marker of artemisinin-resistant malaria",
            "year": 2014,
            "externalIds": {"DOI": "10.1038/nature12876"},
        })

    client = SemanticScholarClient(client=_mock_client(handler))
    record = client.lookup_doi("10.1038/nature12876")

    assert record.title == "A molecular marker of artemisinin-resistant malaria"
    assert record.year == 2014
    assert record.match_score == 1.0


def test_semantic_scholar_search_ranks_by_title_similarity():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "data": [
                {"title": "Completely unrelated paper about frogs", "year": 2001,
                "externalIds": {}},
                {"title": "Kelch13 mutations and artemisinin resistance", "year": 2014,
                "externalIds": {"DOI": "10.1/match"}},
            ]
        })

    client = SemanticScholarClient(client=_mock_client(handler))
    results = client.search("Kelch13 mutations and artemisinin resistance")

    assert results[0].title == "Kelch13 mutations and artemisinin resistance"
    assert results[0].match_score == 1.0  # identical token set -> jaccard 1.0
    assert results[0].doi == "10.1/match"


def test_semantic_scholar_search_http_error_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429)

    client = SemanticScholarClient(client=_mock_client(handler))
    assert client.search("query") == []


# ---- CompositeLiteratureClient ----

def test_composite_uses_crossref_when_confident():
    high_confidence = PaperRecord(title="X", doi="10.1/x", year=2020, match_score=0.95)
    crossref = FakeLiteratureClient("crossref", search_results=[high_confidence])
    s2 = FakeLiteratureClient("s2", search_results=[
        PaperRecord(title="Y", doi="10.1/y", year=2021, match_score=0.99)
    ])

    client = CompositeLiteratureClient(crossref=crossref, s2=s2)
    results = client.search("query")

    assert results == [high_confidence]  # never touched s2's results


def test_composite_falls_back_to_semantic_scholar_when_crossref_unconfident():
    low_confidence = PaperRecord(title="X", doi="10.1/x", year=2020, match_score=0.3)
    better_s2_match = PaperRecord(title="Y", doi="10.1/y", year=2021, match_score=0.9)
    crossref = FakeLiteratureClient("crossref", search_results=[low_confidence])
    s2 = FakeLiteratureClient("s2", search_results=[better_s2_match])

    client = CompositeLiteratureClient(crossref=crossref, s2=s2)
    results = client.search("query")

    assert results[0] == better_s2_match  # s2's better match sorts first


def test_composite_falls_back_when_crossref_finds_nothing():
    s2_match = PaperRecord(title="Y", doi="10.1/y", year=2021, match_score=0.8)
    crossref = FakeLiteratureClient("crossref", search_results=[])
    s2 = FakeLiteratureClient("s2", search_results=[s2_match])

    client = CompositeLiteratureClient(crossref=crossref, s2=s2)
    assert client.search("query") == [s2_match]


def test_composite_lookup_doi_prefers_crossref():
    cr_record = PaperRecord(title="CR", doi="10.1/x", year=2020, match_score=1.0)
    crossref = FakeLiteratureClient("crossref", doi_result=cr_record)
    s2 = FakeLiteratureClient("s2", doi_result=PaperRecord(
        title="S2", doi="10.1/x", year=2020, match_score=1.0))

    client = CompositeLiteratureClient(crossref=crossref, s2=s2)
    assert client.lookup_doi("10.1/x") == cr_record


def test_composite_lookup_doi_falls_back_when_crossref_has_no_record():
    s2_record = PaperRecord(title="S2", doi="10.1/x", year=2020, match_score=1.0)
    crossref = FakeLiteratureClient("crossref", doi_result=None)
    s2 = FakeLiteratureClient("s2", doi_result=s2_record)

    client = CompositeLiteratureClient(crossref=crossref, s2=s2)
    assert client.lookup_doi("10.1/x") == s2_record
