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

Extraction sometimes leaves ``kind`` blank rather than mislabeling it, and a
blank kind matches nothing in the blocklist -- so a drug ("artemisinin"), a
species ("Annona muricata"), or a research method ("metabolomics") can still
reach UniProt unfiltered. When a judge is available, a blank-kind entity is
classified once before any lookup is attempted -- a cheap single-text check,
only spent on entities the kind-based blocklist couldn't already rule out.

Two failure modes, both handled:

1. **Wrong-species ortholog.** A bare name like "DHODH" isn't
   species-specific -- UniProt indexes each organism's ortholog as a separate
   record. When a claim also names a species/organism, that name is passed to
   ``ctx.biodb`` as an organism hint (falls back to the unqualified query if
   the hint finds nothing -- see ``biodb.py``).
2. **Coincidental free-text match.** A bare free-text query's single top
   result can be an unrelated record that merely shares descriptive wording,
   the same false-positive class ``citation_accuracy`` and ``size_of_leap``
   guard against. Rather than committing to that top hit,
   ``ctx.biodb.resolve_candidates`` returns several ranked candidates and,
   when a judge is available, they're checked in order for genuine identity
   -- stopping at the first real match rather than rejecting the entity just
   because UniProt's own text-relevance ranking put the wrong record first.

Degrades gracefully: no entities, or no biodb client -> ``score`` is
``None``. No judge -> identity check is skipped, the top-ranked candidate is
trusted at face value.
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
    # Found scoring real captures: these describe something other than a
    # single resolvable gene/protein record (a multi-protein complex, a
    # structural sub-region, a process, a drug class or regimen, ...), so a
    # UniProt lookup is a category error the same way "species" already was
    # -- it was never going to resolve to one correct record.
    "protein complex", "complex", "protein substructure", "organelle",
    "biochemical process", "process", "drug class", "compound/class",
    "drug protocol", "life cycle stage", "toxicity endpoint", "adverse effect",
    "mutation",
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


_ENTITY_KIND_CHOICES = ["gene_or_protein", "other"]

_ENTITY_KIND_PROMPT = """\
Classify whether this named entity, as used in a scientific hypothesis, refers \
to a specific gene or protein -- choose exactly one:
- gene_or_protein: a specific gene, protein, enzyme, or protein complex subunit \
(e.g. "DHODH", "cytochrome b", "K13", "ATP4").
- other: anything else -- a drug/compound (e.g. "artemisinin"), a species or \
organism (e.g. "Annona muricata"), a cell type/state, a biological process, a \
research method, or a drug class/protocol.
Default to gene_or_protein unless clearly something else.

Entity: "{name}"
"""


def _judge_is_gene_or_protein(judge, entity_name: str) -> bool:
    """True unless the judge is clearly convinced otherwise -- an unnecessary
    lookup costs a wasted API call the downstream identity-match check
    catches anyway, but wrongly skipping a real gene/protein costs a
    dropped accuracy check with no way to recover it. An unparseable verdict
    gets the same benefit of the doubt."""
    verdict = judge.judge(_ENTITY_KIND_PROMPT.format(name=entity_name),
                          choices=_ENTITY_KIND_CHOICES).verdict
    return verdict != "other"


def _organism_hint(claim: Claim, hypothesis: Hypothesis) -> str | None:
    """A species/organism entity to disambiguate a bare gene/protein name
    against. Prefers one co-occurring in this claim -- a gene mention is
    routinely discussed alongside the organism it's about in the same
    sentence (e.g. "P. knowlesi already shows greater susceptibility to
    dihydrofolate reductase inhibitors") -- but falls back to any
    species/organism named elsewhere in the same hypothesis: a hypothesis
    decomposes into several claims that don't each repeat context a sibling
    claim already established (observed for real: a claim naming
    "cytochrome b" with no species of its own, in a hypothesis whose other
    claims establish it's about Plasmodium knowlesi)."""
    for entity in claim.entities:
        if (entity.kind or "").strip().lower() in _SPECIES_KINDS:
            return entity.name
    for other_claim in hypothesis.claims:
        for entity in other_claim.entities:
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
            if not kind and ctx.judge is not None and not _judge_is_gene_or_protein(ctx.judge, entity.name):
                skipped.append({"entity": entity.name, "kind": entity.kind,
                               "reason": "not_a_gene_or_protein_per_judge"})
                continue

            candidates = ctx.biodb.resolve_candidates(entity.name, entity.kind,
                                                      organism=_organism_hint(claim, hypothesis))
            if not candidates:
                checked.append({
                    "entity": entity.name, "kind": entity.kind, "exists": False,
                    "normalized_id": None, "canonical_name": None, "identity_match": None,
                })
                continue

            top = candidates[0]
            record, exists, identity_match = top, True, None
            if ctx.judge is not None and top.canonical_name:
                # Walk ranked candidates best-match-first; stop at the first
                # one that's genuinely the entity meant, rather than
                # rejecting outright just because UniProt's own text-
                # relevance ranking put the wrong record first.
                matched = next(
                    (c for c in candidates if c.canonical_name
                     and _judge_entity_match(ctx.judge, entity.name, c.canonical_name) == "same_entity"),
                    None,
                )
                record = matched if matched is not None else top
                identity_match = "same_entity" if matched is not None else "different_entity"
                exists = matched is not None

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
