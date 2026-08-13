"""Tests for UniProtClient: request/response parsing via httpx.MockTransport
(no real network call), covering the response shapes actually observed
against the live API -- reviewed entries with a recommendedName, unreviewed
(TrEMBL) entries with only submissionNames, and entries with no protein name
at all (just a gene symbol or ORF/systematic identifier).
"""

from __future__ import annotations

import httpx

from benchmarking_pipeline.services.biodb import UniProtClient


def _mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_resolve_uses_broad_free_text_query_not_a_gene_field_query():
    """Non-model organisms are frequently indexed under a systematic/ORF
    identifier rather than a canonical gene symbol -- a field-scoped
    ``gene:`` query finds nothing for those (confirmed against the live API).
    The query sent must be the raw name, not wrapped in a gene: filter."""
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == "PKNH_1436200"
        assert "gene:" not in request.url.params["query"]
        return httpx.Response(200, json={"results": []})

    client = UniProtClient(client=_mock_client(handler))
    client.resolve("PKNH_1436200")


def test_resolve_reviewed_entry_uses_recommended_name():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{
            "primaryAccession": "Q9NR33",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Dihydroorotate dehydrogenase"}},
            },
            "genes": [{"geneName": {"value": "DHODH"}}],
        }]})

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("DHODH", "gene")

    assert record.exists is True
    assert record.normalized_id == "Q9NR33"
    assert record.canonical_name == "Dihydroorotate dehydrogenase"


def test_resolve_unreviewed_entry_falls_back_to_submission_name():
    """The real case observed for PKNH_1436200: an unreviewed (TrEMBL) entry
    with no recommendedName, only a submissionNames entry."""
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{
            "primaryAccession": "A0A384KT14",
            "proteinDescription": {
                "submissionNames": [
                    {"fullName": {"value": "Cell traversal protein for ookinetes and sporozoites"}},
                ],
            },
            "genes": [{"orfNames": [{"value": "PKNH_1436200"}]}],
        }]})

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("PKNH_1436200")

    assert record.exists is True
    assert record.normalized_id == "A0A384KT14"
    assert record.canonical_name == "Cell traversal protein for ookinetes and sporozoites"


def test_resolve_falls_back_to_orf_name_when_no_protein_name_at_all():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{
            "primaryAccession": "A0A999XYZ0",
            "proteinDescription": {},
            "genes": [{"orfNames": [{"value": "PKNH_9999900"}]}],
        }]})

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("PKNH_9999900")

    assert record.exists is True
    assert record.canonical_name == "PKNH_9999900"


def test_resolve_no_results_reports_does_not_exist():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": []})

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("not_a_real_gene_xyzabc")

    assert record.exists is False
    assert record.normalized_id is None
    assert record.canonical_name is None
    assert record.query == "not_a_real_gene_xyzabc"


def test_resolve_http_error_reports_does_not_exist_not_a_crash():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("DHODH")

    assert record.exists is False


def test_resolve_with_organism_tries_qualified_query_first():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["query"] == 'DHODH AND organism_name:"Plasmodium knowlesi"'
        return httpx.Response(200, json={"results": [{
            "primaryAccession": "I3L449",
            "proteinDescription": {
                "recommendedName": {"fullName": {"value": "Dihydroorotate dehydrogenase (quinone)"}},
            },
        }]})

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("DHODH", organism="Plasmodium knowlesi")

    assert record.exists is True
    assert record.normalized_id == "I3L449"


def test_resolve_falls_back_to_bare_query_when_organism_qualified_finds_nothing():
    """A thinly-annotated organism's entry may not be indexed under that
    exact organism filter -- the qualified query finding nothing must not be
    reported as "doesn't exist", it should fall back to the bare query."""
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        query = request.url.params["query"]
        calls.append(query)
        if "organism_name" in query:
            return httpx.Response(200, json={"results": []})
        return httpx.Response(200, json={"results": [{
            "primaryAccession": "A0A384KT14",
            "proteinDescription": {"submissionNames": [{"fullName": {"value": "Some protein"}}]},
        }]})

    client = UniProtClient(client=_mock_client(handler))
    record = client.resolve("PKNH_1436200", organism="Plasmodium knowlesi")

    assert len(calls) == 2  # qualified query tried first, then the fallback
    assert record.exists is True
    assert record.normalized_id == "A0A384KT14"


def test_resolve_without_organism_skips_the_qualified_query_entirely():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "organism_name" not in request.url.params["query"]
        return httpx.Response(200, json={"results": []})

    client = UniProtClient(client=_mock_client(handler))
    client.resolve("DHODH")
