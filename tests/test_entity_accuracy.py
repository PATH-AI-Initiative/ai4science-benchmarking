"""Tests for entity_accuracy: resolution existence checking against a biodb
client, and the gene/protein-vs-other-kind filtering that keeps it from
sending a category-error lookup (e.g. a species name) to UniProt.
"""

from __future__ import annotations

from benchmarking_pipeline.axes.accuracy.entity_accuracy import EntityAccuracy
from benchmarking_pipeline.core.config import RunConfig
from benchmarking_pipeline.core.context import Context
from benchmarking_pipeline.core.models import (
    Claim,
    Entity,
    EvaluationRun,
    Hypothesis,
    HypothesisSet,
    Tool,
)
from benchmarking_pipeline.services.biodb import EntityRecord
from benchmarking_pipeline.services.llm_judge import Judgement

METRIC = EntityAccuracy()


class FakeBioDatabase:
    """Keyed by entity name -> EntityRecord (or None -> not found). Records
    every ``resolve`` call (including the organism hint passed) for tests
    that need to assert on it."""

    name = "fake"

    def __init__(self, records: dict[str, EntityRecord | None]):
        self._records = records
        self.calls: list[tuple[str, str | None, str | None]] = []

    def resolve(self, name: str, kind: str | None = None, organism: str | None = None) -> EntityRecord:
        self.calls.append((name, kind, organism))
        record = self._records.get(name)
        if record is not None:
            return record
        return EntityRecord(query=name, normalized_id=None, canonical_name=None, exists=False)


class FakeJudge:
    """Returns "same_entity" for any candidate name in ``same_entity_names``,
    "different_entity" otherwise (the safe default)."""

    name = "fake"

    def __init__(self, same_entity_names: set[str] = frozenset()):
        self.same_entity_names = same_entity_names
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        verdict = "different_entity"
        for candidate_name in self.same_entity_names:
            if candidate_name in prompt:
                verdict = "same_entity"
                break
        return Judgement(verdict=verdict, confidence=1.0)


def _hyp(claims: list[Claim]) -> tuple[Hypothesis, EvaluationRun]:
    hyp = Hypothesis(id="h1", text="...", rank=1, claims=claims)
    run = EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=[hyp]))
    return hyp, run


def test_no_entities_reports_not_assessed():
    hyp, run = _hyp([Claim(text="a claim with no entities")])
    result = METRIC.score(hyp, run, Context(config=RunConfig(), biodb=FakeBioDatabase({})))
    assert result.score is None
    assert result.evidence["status"] == "no_entities"


def test_no_biodb_client_reports_not_assessed():
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="gene")])
    hyp, run = _hyp([claim])
    result = METRIC.score(hyp, run, Context(config=RunConfig(), biodb=None))
    assert result.score is None
    assert result.evidence["status"] == "no_biodb_client"


def test_real_gene_resolves_and_scores_one():
    real = EntityRecord(query="DHODH", normalized_id="Q9NR33",
                        canonical_name="Dihydroorotate dehydrogenase", exists=True)
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="gene")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"DHODH": real}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.anchor == "all entities resolve"
    assert result.evidence["checked"][0]["normalized_id"] == "Q9NR33"


def test_fabricated_gene_scores_zero():
    claim = Claim(text="x", entities=[Entity(name="NotARealGeneXYZ", kind="gene")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["exists"] is False


def test_species_kind_is_skipped_not_looked_up():
    """A species name isn't a UniProt gene/protein record -- resolving it
    there would be a category error, not a meaningful accuracy check."""
    claim = Claim(text="x", entities=[Entity(name="Plasmodium knowlesi", kind="species")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_lookupable_entities"
    assert result.evidence["skipped"][0]["reason"] == "not_a_gene_or_protein"


def test_unlabeled_kind_is_still_attempted():
    """Extraction couldn't tell what kind of entity this is (kind="") --
    still worth attempting, unlike a definite non-gene/protein kind."""
    real = EntityRecord(query="PKNH_1436200", normalized_id="A0A384KT14",
                        canonical_name="Some protein", exists=True)
    claim = Claim(text="x", entities=[Entity(name="PKNH_1436200", kind="")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"PKNH_1436200": real}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["skipped"] == []


def test_synonym_kind_not_in_examples_is_still_attempted():
    """Regression test: a real extraction (gpt-4.1 on the pknowlesi report)
    tagged DHODH as kind="enzyme", not "gene"/"protein" -- the two kinds the
    extraction prompt's schema description literally names as examples.
    kind is free text an LLM fills in, not a fixed enum, so an allowlist of
    just the schema's examples would keep silently skipping real proteins
    tagged with a synonym the prompt never mentioned. Only kinds clearly
    NOT a gene/protein (species, drug, pathway, ...) should be skipped."""
    real = EntityRecord(query="DHODH", normalized_id="Q9NR33",
                        canonical_name="Dihydroorotate dehydrogenase", exists=True)
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="enzyme")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"DHODH": real}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["skipped"] == []


def test_mixed_entities_score_only_over_lookupable_ones():
    real_gene = EntityRecord(query="DHODH", normalized_id="Q9NR33",
                             canonical_name="Dihydroorotate dehydrogenase", exists=True)
    claims = [
        Claim(text="a", entities=[Entity(name="DHODH", kind="gene")]),
        Claim(text="b", entities=[Entity(name="FakeGeneXYZ", kind="protein")]),
        Claim(text="c", entities=[Entity(name="Plasmodium knowlesi", kind="species")]),
    ]
    hyp, run = _hyp(claims)
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"DHODH": real_gene}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.5  # DHODH exists, FakeGeneXYZ doesn't; species skipped entirely
    assert result.evidence["n_total"] == 2
    assert len(result.evidence["skipped"]) == 1


def test_coincidental_text_match_is_rejected_with_a_judge():
    """Regression test: searching "dihydrofolate reductase" against the real
    UniProt API returned an unrelated enzyme (a coincidental text match, not
    the actual DHFR record) -- the same class of false positive
    citation_accuracy/size_of_leap guard against for their own fuzzy matches.
    """
    wrong_match = EntityRecord(query="dihydrofolate reductase", normalized_id="P08773",
                               canonical_name="Deoxycytidylate 5-hydroxymethyltransferase",
                               exists=True)
    claim = Claim(text="x", entities=[Entity(name="dihydrofolate reductase", kind="enzyme")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(same_entity_names=set())  # everything is "different_entity"
    ctx = Context(config=RunConfig(),
                 biodb=FakeBioDatabase({"dihydrofolate reductase": wrong_match}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["exists"] is False
    assert result.evidence["checked"][0]["identity_match"] == "different_entity"
    # The record is still reported for review, not hidden just because it was rejected.
    assert result.evidence["checked"][0]["canonical_name"] == "Deoxycytidylate 5-hydroxymethyltransferase"


def test_genuine_match_still_scores_with_a_judge():
    real = EntityRecord(query="DHODH", normalized_id="Q9NR33",
                        canonical_name="Dihydroorotate dehydrogenase", exists=True)
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="gene")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(same_entity_names={"Dihydroorotate dehydrogenase"})
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"DHODH": real}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["identity_match"] == "same_entity"
    assert judge.calls == 1


def test_no_judge_trusts_resolved_record_at_face_value():
    """No judge configured -> identity check is skipped entirely (can't run
    it), so a resolved record is trusted as before -- graceful degradation,
    not a stricter default that would need a judge to ever pass."""
    maybe_wrong = EntityRecord(query="dihydrofolate reductase", normalized_id="P08773",
                               canonical_name="Deoxycytidylate 5-hydroxymethyltransferase",
                               exists=True)
    claim = Claim(text="x", entities=[Entity(name="dihydrofolate reductase", kind="enzyme")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(),
                 biodb=FakeBioDatabase({"dihydrofolate reductase": maybe_wrong}), judge=None)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["identity_match"] is None


def test_cooccurring_species_entity_is_passed_as_organism_hint():
    """A gene/protein mention is routinely discussed alongside the organism
    it's about in the same claim -- that species name should reach the
    biodb client as a disambiguating hint, since a bare gene name isn't
    species-specific (UniProt indexes every organism's ortholog separately).
    """
    claim = Claim(text="P. knowlesi shows greater susceptibility to DHODH inhibitors.",
                  entities=[Entity(name="Plasmodium knowlesi", kind="species"),
                            Entity(name="DHODH", kind="enzyme")])
    hyp, run = _hyp([claim])
    biodb = FakeBioDatabase({})
    ctx = Context(config=RunConfig(), biodb=biodb)

    METRIC.score(hyp, run, ctx)

    dhodh_calls = [c for c in biodb.calls if c[0] == "DHODH"]
    assert dhodh_calls == [("DHODH", "enzyme", "Plasmodium knowlesi")]


def test_no_cooccurring_species_means_no_organism_hint():
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="enzyme")])
    hyp, run = _hyp([claim])
    biodb = FakeBioDatabase({})
    ctx = Context(config=RunConfig(), biodb=biodb)

    METRIC.score(hyp, run, ctx)

    assert biodb.calls == [("DHODH", "enzyme", None)]
