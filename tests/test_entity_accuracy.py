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
    """Keyed by entity name -> EntityRecord, a list of EntityRecord (ranked
    candidates), or None -> not found. Records every ``resolve_candidates``
    call (including the organism hint passed) for tests that need to assert
    on it."""

    name = "fake"

    def __init__(self, records: dict[str, EntityRecord | list[EntityRecord] | None]):
        self._records = records
        self.calls: list[tuple[str, str | None, str | None]] = []

    def resolve(self, name: str, kind: str | None = None, organism: str | None = None) -> EntityRecord:
        candidates = self.resolve_candidates(name, kind, organism)
        return candidates[0] if candidates else EntityRecord(
            query=name, normalized_id=None, canonical_name=None, exists=False,
        )

    def resolve_candidates(
        self, name: str, kind: str | None = None, organism: str | None = None, limit: int = 5,
    ) -> list[EntityRecord]:
        self.calls.append((name, kind, organism))
        record = self._records.get(name)
        if record is None:
            return []
        return record if isinstance(record, list) else [record]


class FakeJudge:
    """Dispatches on ``choices`` to answer either judge call this metric
    makes: the identity-match check ("same_entity" for any candidate name in
    ``same_entity_names``, "different_entity" otherwise) or the blank-kind
    classification ("other" for any entity name in ``other_kind_names``,
    "gene_or_protein" otherwise -- the safe default)."""

    name = "fake"

    def __init__(self, same_entity_names: set[str] = frozenset(),
                other_kind_names: set[str] = frozenset()):
        self.same_entity_names = same_entity_names
        self.other_kind_names = other_kind_names
        self.calls = 0

    def judge(self, prompt: str, *, choices=None) -> Judgement:
        self.calls += 1
        if choices == ["gene_or_protein", "other"]:
            verdict = "gene_or_protein"
            for name in self.other_kind_names:
                if name in prompt:
                    verdict = "other"
                    break
            return Judgement(verdict=verdict, confidence=1.0)
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


def test_organism_hint_falls_back_to_a_sibling_claim_in_the_same_hypothesis():
    """Regression test for a real case: a hypothesis's claims each cover a
    different sub-point, so a species is only named once, in a different
    claim than the gene/protein mention -- the gene-bearing claim itself has
    no species tagged. The hint should still reach the biodb client rather
    than being lost because it happened to land on a sibling claim."""
    claims = [
        Claim(text="Atovaquone is highly potent against Plasmodium knowlesi.",
             entities=[Entity(name="Plasmodium knowlesi", kind="species")]),
        Claim(text="Dual-site inhibition of bc1 requires mutations in cytochrome b.",
             entities=[Entity(name="cytochrome b", kind="protein")]),
    ]
    hyp = Hypothesis(id="h1", text="...", rank=1, claims=claims)
    run = EvaluationRun(tool=Tool(name="t"), prompt="p", outputs=HypothesisSet(hypotheses=[hyp]))
    biodb = FakeBioDatabase({})
    ctx = Context(config=RunConfig(), biodb=biodb)

    METRIC.score(hyp, run, ctx)

    cytb_calls = [c for c in biodb.calls if c[0] == "cytochrome b"]
    assert cytb_calls == [("cytochrome b", "protein", "Plasmodium knowlesi")]


def test_falls_through_to_next_candidate_when_top_hit_is_wrong():
    """Regression test for a real case: querying "PI4K" against the live
    UniProt API returned an unrelated human protein ("Hyccin 2") as the
    single top hit on text overlap alone. Rather than rejecting the entity
    outright, a second, genuinely-matching candidate further down the ranked
    list should be found and used instead."""
    wrong_top = EntityRecord(query="PI4K", normalized_id="Q8IXS8",
                             canonical_name="Hyccin 2", exists=True)
    real_match = EntityRecord(query="PI4K", normalized_id="P42356",
                              canonical_name="Phosphatidylinositol 4-kinase alpha", exists=True)
    claim = Claim(text="x", entities=[Entity(name="PI4K", kind="protein")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(same_entity_names={"Phosphatidylinositol 4-kinase alpha"})
    ctx = Context(config=RunConfig(),
                 biodb=FakeBioDatabase({"PI4K": [wrong_top, real_match]}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["checked"][0]["normalized_id"] == "P42356"
    assert result.evidence["checked"][0]["identity_match"] == "same_entity"
    assert judge.calls == 2  # top hit rejected, second hit checked and accepted


def test_all_candidates_rejected_scores_zero_and_reports_the_top_one():
    wrong_a = EntityRecord(query="X", normalized_id="A1", canonical_name="Unrelated protein A", exists=True)
    wrong_b = EntityRecord(query="X", normalized_id="A2", canonical_name="Unrelated protein B", exists=True)
    claim = Claim(text="x", entities=[Entity(name="X", kind="protein")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(same_entity_names=set())  # everything rejected
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"X": [wrong_a, wrong_b]}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 0.0
    assert result.evidence["checked"][0]["identity_match"] == "different_entity"
    # Still reports the top candidate for review, not hidden just because rejected.
    assert result.evidence["checked"][0]["normalized_id"] == "A1"
    assert judge.calls == 2  # both candidates checked, none matched


def test_non_atomic_entity_kinds_are_skipped():
    """A protein complex, a structural sub-region, a process, or a drug class
    is not a single resolvable gene/protein record -- looking it up is a
    category error the same way "species" already is."""
    claims = [
        Claim(text="a", entities=[Entity(name="bc1 complex", kind="protein complex")]),
        Claim(text="b", entities=[Entity(name="Qo site", kind="protein substructure")]),
        Claim(text="c", entities=[Entity(name="mitochondrial", kind="organelle")]),
        Claim(text="d", entities=[Entity(name="artemisinin activation", kind="biochemical process")]),
        Claim(text="e", entities=[Entity(name="quinolones", kind="drug class")]),
    ]
    hyp, run = _hyp(claims)
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({}))

    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_lookupable_entities"
    assert len(result.evidence["skipped"]) == 5


def test_blank_kind_non_gene_entity_is_skipped_via_judge():
    """Regression test for a real case: a drug name ("artemisinin"), a
    species ("Annona muricata"), or a research method ("metabolomics") can
    reach UniProt unfiltered when extraction leaves kind blank rather than
    mislabeling it -- the kind-based blocklist has nothing to match against.
    With a judge available, a blank-kind entity is classified before any
    lookup is attempted."""
    claim = Claim(text="x", entities=[Entity(name="artemisinin", kind="")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(other_kind_names={"artemisinin"})
    biodb = FakeBioDatabase({})
    ctx = Context(config=RunConfig(), biodb=biodb, judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score is None
    assert result.evidence["status"] == "no_lookupable_entities"
    assert result.evidence["skipped"][0]["reason"] == "not_a_gene_or_protein_per_judge"
    assert biodb.calls == []  # never reached the biodb -- skipped before any lookup


def test_blank_kind_gene_entity_still_proceeds_via_judge():
    real = EntityRecord(query="DHODH", normalized_id="Q9NR33",
                        canonical_name="Dihydroorotate dehydrogenase", exists=True)
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="")])
    hyp, run = _hyp([claim])
    # other_kind_names empty -> DHODH classified as gene_or_protein (proceeds);
    # same_entity_names covers the identity-match check that follows.
    judge = FakeJudge(same_entity_names={"Dihydroorotate dehydrogenase"})
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"DHODH": real}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["skipped"] == []


def test_labeled_kind_skips_the_judge_classification_entirely():
    """The judge-classification step is only for blank kind -- an entity
    already labeled (correctly or not) goes straight through the existing
    kind-based blocklist/allowlist path, costing no extra judge call beyond
    the identity-match check that already ran for it."""
    real = EntityRecord(query="DHODH", normalized_id="Q9NR33",
                        canonical_name="Dihydroorotate dehydrogenase", exists=True)
    claim = Claim(text="x", entities=[Entity(name="DHODH", kind="gene")])
    hyp, run = _hyp([claim])
    judge = FakeJudge(same_entity_names={"Dihydroorotate dehydrogenase"})
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"DHODH": real}), judge=judge)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert judge.calls == 1  # only the identity-match check, no kind classification


def test_no_judge_skips_the_blank_kind_classification_entirely():
    """No judge configured -> the blank-kind classification can't run either
    -- graceful degradation, same as every other judge-optional check here:
    the entity is still attempted, trusted at face value."""
    real = EntityRecord(query="artemisinin", normalized_id="X1",
                        canonical_name="Some record", exists=True)
    claim = Claim(text="x", entities=[Entity(name="artemisinin", kind="")])
    hyp, run = _hyp([claim])
    ctx = Context(config=RunConfig(), biodb=FakeBioDatabase({"artemisinin": real}), judge=None)

    result = METRIC.score(hyp, run, ctx)
    assert result.score == 1.0
    assert result.evidence["skipped"] == []
