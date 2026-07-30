"""Tests for parse_dict's handling of category, entities, and self-reported
metadata -- the fields extraction.py added for tools whose reports go beyond
a flat ranked list of single-sentence hypotheses."""

from __future__ import annotations

from benchmarking_pipeline.io.parsers import parse_dict


def test_category_empty_string_becomes_none():
    data = {"hypotheses": [{"id": "h1", "text": "flat, unranked tool output", "rank": 1,
                            "category": "", "claims": []}]}
    hyp = parse_dict(data).hypotheses[0]
    assert hyp.category is None


def test_category_is_preserved_when_present():
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1,
                            "category": "Category 1: Metabolic Vulnerabilities", "claims": []}]}
    hyp = parse_dict(data).hypotheses[0]
    assert hyp.category == "Category 1: Metabolic Vulnerabilities"


def test_claim_entities_are_parsed():
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1, "claims": [
        {"text": "DHODH is the rate-limiting enzyme.", "role": "premise",
         "references": [], "entities": [{"name": "PKNH_1436200", "kind": "gene"}]},
    ]}]}
    claim = parse_dict(data).hypotheses[0].claims[0]
    assert len(claim.entities) == 1
    assert claim.entities[0].name == "PKNH_1436200"
    assert claim.entities[0].kind == "gene"


def test_self_reported_lands_in_hypothesis_metadata_not_claims():
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1, "claims": [],
                            "self_reported": {"novelty_tier": "First-in-class",
                                              "novelty_rationale": "never explored before",
                                              "feasibility_tier": "", "feasibility_rationale": "",
                                              "key_risks": []}}]}
    hyp = parse_dict(data).hypotheses[0]
    assert hyp.metadata["self_reported"]["novelty_tier"] == "First-in-class"
    assert hyp.claims == []


def test_all_blank_self_reported_is_dropped():
    """A self_reported block with nothing in it (the tool didn't self-assess)
    shouldn't leave a hollow entry cluttering metadata."""
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1, "claims": [],
                            "self_reported": {"novelty_tier": "", "novelty_rationale": "",
                                              "feasibility_tier": "", "feasibility_rationale": "",
                                              "key_risks": []}}]}
    hyp = parse_dict(data).hypotheses[0]
    assert hyp.metadata == {}


def test_reference_resolved_against_bibliography_is_parsed_with_metadata():
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1, "claims": [
        {"text": "The cycle is ~24h.", "role": "premise", "entities": [], "references": [
            {"raw": "[107]", "title": "Human Infections and Detection of Plasmodium knowlesi",
             "doi": "10.1128/cmr.00079-12", "year": "2013", "authors": ["Singh B", "Daneshvar C"]},
        ]},
    ]}]}
    ref = parse_dict(data).hypotheses[0].claims[0].references[0]
    assert ref.raw == "[107]"
    assert ref.title == "Human Infections and Detection of Plasmodium knowlesi"
    assert ref.doi == "10.1128/cmr.00079-12"
    assert ref.year == 2013
    assert ref.authors == ["Singh B", "Daneshvar C"]


def test_unresolved_reference_keeps_bare_marker_with_no_invented_fields():
    """A citation the extractor couldn't resolve against any bibliography --
    e.g. a bare [N] with no matching numbered entry -- should stay honest
    (raw only, everything else None/empty), not get fabricated metadata."""
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1, "claims": [
        {"text": "...", "role": "premise", "entities": [],
         "references": [{"raw": "[999]", "title": "", "doi": "", "year": "", "authors": []}]},
    ]}]}
    ref = parse_dict(data).hypotheses[0].claims[0].references[0]
    assert ref.raw == "[999]"
    assert ref.title is None
    assert ref.doi is None
    assert ref.year is None
    assert ref.authors == []


def test_missing_optional_fields_dont_break_parsing():
    """Older/hand-authored captures without category/entities/self_reported
    should still parse -- these fields are additive, not required."""
    data = {"hypotheses": [{"id": "h1", "text": "...", "rank": 1,
                            "claims": [{"text": "a claim", "references": []}]}]}
    hyp = parse_dict(data).hypotheses[0]
    assert hyp.category is None
    assert hyp.metadata == {}
    assert hyp.claims[0].entities == []
