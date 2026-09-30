"""Entity accuracy: are named biological entities real?

Resolves each entity a claim mentions against ``ctx.biodb`` (UniProt, KEGG,
...) and scores the fraction that resolve to a real record. Existence only --
checking that entities relate to each other correctly is a further layer not
attempted here, the same existence/support split ``citation_accuracy`` draws.

Entities tagged with a kind that clearly isn't a gene/protein record --
``species``, a cell line, a drug/compound, a pathway -- are skipped rather
than looked up (resolving "Plasmodium knowlesi" against UniProt is a category
error). Everything else is attempted, including unlabeled and free-text kinds
like ``enzyme`` or ``receptor``: this is a blocklist, not an allowlist,
because ``kind`` is free text an LLM extraction fills in, and an allowlist of
just "gene"/"protein" would silently miss real proteins tagged otherwise.
Skipped entities are reported, not dropped.

Two failure modes, both handled:

1. **Wrong-species ortholog.** A bare name like "DHODH" isn't
   species-specific -- UniProt indexes each organism's ortholog as a separate
   record. When a claim also names a species/organism, that name is passed to
   ``ctx.biodb.resolve`` as an organism hint (falls back to the unqualified
   query if the hint finds nothing -- see ``biodb.py``).
2. **Coincidental free-text match.** A search can return an unrelated record
   that merely shares descriptive wording -- the same false-positive class
   ``citation_accuracy`` and ``size_of_leap`` guard against. When a judge is
   available, a resolved record is checked for genuine identity before it
   counts as existing.

Degrades gracefully: no entities, or no biodb client -> ``score`` is
``None``. No judge -> identity check is skipped, record trusted at face value.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, HypothesisMetric, MetricResult
from ...core.models import Claim, EvaluationRun, Hypothesis
from ...core.registry import register
from ...services.relationship_judge import judge_relationship

_NOT_A_GENE_OR_PROTEIN_KINDS = {
    "species", "organism", "cell line", "drug", "compound", "chemical",
    "pathway", "disease", "algorithm", "phenotype",
}
_SPECIES_KINDS = {"species", "organism"}

# "different_entity" listed last: judge_relationship's fallback-on-unparseable-
# verdict, and the conservative choice when identity is ambiguous -- a
# resolved record isn't trusted as the right one on a shrug.
_ENTITY_MATCH_CHOICES = ["same_entity", "different_entity"]

_ENTITY_MATCH_INSTRUCTIONS = """\
A biological database search for a named entity returned this candidate \
record. Free-text search can match on coincidental wording overlap rather \
than genuine identity -- e.g. searching "dihydrofolate reductase" can return \
an unrelated enzyme that merely shares some descriptive terms. Choose \
exactly one:
- same_entity: the candidate is genuinely the gene/protein/enzyme named by \
the query (species-specific naming, abbreviations, and synonyms are fine).
- different_entity: the candidate is a different gene/protein/enzyme \
despite any textual similarity.
Default to different_entity unless clearly the same entity.
"""


def _judge_entity_match(judge, queried_name: str, candidate_name: str) -> str:
    return judge_relationship(
        judge,
        label_a="Queried entity", text_a=queried_name,
        label_b="Database record", text_b=candidate_name,
        instructions=_ENTITY_MATCH_INSTRUCTIONS,
        choices=_ENTITY_MATCH_CHOICES,
    )


def _organism_hint(claim: Claim) -> str | None:
    """The first species/organism entity co-occurring in this claim, if any
    -- a claim's gene/protein mentions are routinely discussed alongside the
    organism they're about (e.g. "P. knowlesi already shows greater
    susceptibility to dihydrofolate reductase inhibitors")."""
    for entity in claim.entities:
        if (entity.kind or "").strip().lower() in _SPECIES_KINDS:
            return entity.name
    return None


@register
class EntityAccuracy(HypothesisMetric):
    name = "entity_accuracy"
    axis = Axis.ACCURACY
    is_floor = True

    def score(self, hypothesis: Hypothesis, run: EvaluationRun, ctx: Context) -> MetricResult:
        occurrences = [
            (claim, entity) for claim in hypothesis.claims for entity in claim.entities
        ]
        if not occurrences:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_entities"},
            )
        if ctx.biodb is None:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_biodb_client"},
            )

        checked = []
        skipped = []
        for claim, entity in occurrences:
            kind = (entity.kind or "").strip().lower()
            if kind in _NOT_A_GENE_OR_PROTEIN_KINDS:
                skipped.append({"entity": entity.name, "kind": entity.kind,
                               "reason": "not_a_gene_or_protein"})
                continue

            record = ctx.biodb.resolve(entity.name, entity.kind, organism=_organism_hint(claim))
            exists = record.exists
            identity_match = None
            if exists and ctx.judge is not None and record.canonical_name:
                identity_match = _judge_entity_match(ctx.judge, entity.name, record.canonical_name)
                exists = identity_match == "same_entity"

            checked.append({
                "entity": entity.name, "kind": entity.kind, "exists": exists,
                "normalized_id": record.normalized_id, "canonical_name": record.canonical_name,
                "identity_match": identity_match,
            })

        if not checked:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_lookupable_entities", "skipped": skipped},
            )

        n_exist = sum(1 for c in checked if c["exists"])
        score = n_exist / len(checked)

        return MetricResult(
            metric=self.name, axis=self.axis, score=score,
            anchor=_anchor(score),
            evidence={
                "checked": checked, "skipped": skipped,
                "n_exist": n_exist, "n_total": len(checked),
                "biodb": ctx.biodb.name,
            },
        )


def _anchor(score: float) -> str:
    if score >= 0.95:
        return "all entities resolve"
    if score >= 0.7:
        return "most entities resolve"
    if score >= 0.4:
        return "many entities unresolvable"
    return "entities largely unresolvable"
