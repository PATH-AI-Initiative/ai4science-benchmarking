"""Size of scientific leap: distance from each hypothesis to a fixed set of
reference anchor papers.

Anchors are resolved once per evaluation, via a literature search on the run's
prompt (not per hypothesis), and cached on ``ctx.anchor_cache``. Every
hypothesis, and every run in a bundle, is scored against the same anchor set.

Anchors are auto-selected via literature search, not curated by a domain
expert. That's a real limitation: anchor-set composition shapes what counts
as "novel", so the anchors are reported in full in ``evidence["anchors"]``.

Score is distance to the nearest anchor -- not the four-way
restatement/recombination/extension/novel classification the framework
ultimately wants, which needs a curated, calibrated anchor set. Distance is
not a validity signal: a hypothesis far from every anchor must still be read
against the Accuracy axis, since "distant" and "wrong" look the same here.

Anchor search is bibliographic (keyword/relevance-based), not semantic, so a
returned anchor can share vocabulary with the prompt while being off-topic --
the same failure mode ``citation_accuracy`` guards against. When a judge is
available, each anchor is checked for genuine topical relevance before it can
serve as a hypothesis's nearest match.

Degrades gracefully: no embeddings, no literature client, or no anchors found
for the prompt -> score is None. No judge -> topical-relevance check is
skipped. A hypothesis with no topically relevant anchor is excluded from the
mean and reported separately.
"""

from __future__ import annotations

import numpy as np

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, HypothesisSet
from ...core.registry import register
from ...services.embeddings import cosine_similarity
from ...services.literature import PaperRecord
from ...services.relationship_judge import judge_relationship
from ..quality._shared import hypothesis_text

# "different_topic" listed last: judge_relationship's fallback-on-unparseable-
# verdict, and the conservative choice when topical relevance is ambiguous --
# an anchor match isn't trusted on a shrug.
_TOPICAL_MATCH_CHOICES = ["same_topic", "different_topic"]

_TOPICAL_MATCH_INSTRUCTIONS = """\
A bibliographic search for related work returned this candidate reference \
point. Search relevance ranking is based on shared wording, not subject \
matter -- generic methodological terms (e.g. "meta-learning", "model \
misspecification", "closed-loop", "conformal prediction") can make a paper \
from a completely unrelated field or application domain score as a strong \
textual match. Choose exactly one:
- same_topic: the candidate addresses the same subject matter/research problem \
as the hypothesis (not necessarily identical, but genuinely related work).
- different_topic: the candidate is from an unrelated field or application \
domain, sharing only generic terminology with the hypothesis.
Default to different_topic unless clearly the same topic.
"""


def _judge_topical_match(judge, hyp_text: str, candidate_text: str) -> str:
    return judge_relationship(
        judge,
        label_a="Hypothesis", text_a=hyp_text,
        label_b="Reference anchor", text_b=candidate_text,
        instructions=_TOPICAL_MATCH_INSTRUCTIONS,
        choices=_TOPICAL_MATCH_CHOICES,
    )


def _paper_text(record: PaperRecord) -> str:
    if record.abstract:
        return f"{record.title or ''} {record.abstract}".strip()
    return record.title or ""


_ANCHOR_LIMIT = 5


def _resolve_anchors(query: str, ctx: Context) -> list[tuple[PaperRecord, np.ndarray]] | str:
    """The fixed anchor set for this evaluation: up to ``_ANCHOR_LIMIT`` papers
    found by searching the literature for ``query``, embedded once, and cached
    on ``ctx`` so every run and hypothesis compares against the same set.

    Returns a list of ``(record, embedding_vector)`` pairs, or the string
    ``"no_literature_found"`` on a cache miss with no hits.
    """
    cached = ctx.anchor_cache.get(query)
    if cached is not None:
        return cached

    hits = [h for h in ctx.literature.search(query, limit=_ANCHOR_LIMIT) if h.title or h.abstract]
    if not hits:
        ctx.anchor_cache[query] = "no_literature_found"
        return "no_literature_found"

    vectors = ctx.embeddings.embed([_paper_text(h) for h in hits])
    resolved = list(zip(hits, vectors))
    ctx.anchor_cache[query] = resolved
    return resolved


@register
class SizeOfLeap(SetMetric):
    name = "size_of_leap"
    axis = Axis.NOVELTY

    def score(self, outputs: HypothesisSet, run: EvaluationRun, ctx: Context) -> MetricResult:
        if ctx.embeddings is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_embedding_model"})
        if ctx.literature is None:
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": "no_literature_client"})

        anchors = _resolve_anchors(run.prompt, ctx)
        if isinstance(anchors, str):
            return MetricResult(metric=self.name, axis=self.axis, score=None,
                                evidence={"status": anchors})

        assessed = []
        unassessed = []
        for h in outputs:
            h_text = hypothesis_text(h)
            h_vec = ctx.embeddings.embed([h_text])[0]
            similarities = [cosine_similarity(h_vec, vec) for _, vec in anchors]
            ranked = sorted(range(len(anchors)), key=lambda i: similarities[i], reverse=True)

            if ctx.judge is not None:
                # Lazily check candidates best-match-first; stop at the first
                # one that's genuinely on-topic rather than spending a judge
                # call on every anchor.
                best_i = next(
                    (i for i in ranked
                     if _judge_topical_match(ctx.judge, h_text, _paper_text(anchors[i][0])) == "same_topic"),
                    None,
                )
                if best_i is None:
                    unassessed.append({"hypothesis_id": h.id, "rank": h.rank,
                                       "reason": "no_topically_relevant_anchor"})
                    continue
            else:
                best_i = ranked[0]

            best_sim = similarities[best_i]
            distance = max(0.0, min(1.0, 1.0 - best_sim))
            nearest = anchors[best_i][0]
            assessed.append({
                "hypothesis_id": h.id, "rank": h.rank,
                "distance": round(distance, 4),
                "nearest_anchor_similarity": round(best_sim, 4),
                "nearest_anchor_title": nearest.title,
            })

        anchor_evidence = [
            {"title": rec.title, "doi": rec.doi, "year": rec.year} for rec, _ in anchors
        ]

        if not assessed:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_hypotheses_could_be_assessed",
                         "unassessed": unassessed, "anchors": anchor_evidence},
            )

        mean_distance = sum(a["distance"] for a in assessed) / len(assessed)

        return MetricResult(
            metric=self.name, axis=self.axis,
            score=round(mean_distance, 4),
            anchor=_anchor(mean_distance),
            evidence={
                "n_hypotheses": len(outputs),
                "n_assessed": len(assessed),
                "per_hypothesis": assessed,
                "unassessed": unassessed,
                "anchors": anchor_evidence,
                "embedding_model": ctx.embeddings.name,
                "note": "distance from a fixed reference set is not itself validity -- "
                        "interpret alongside the Accuracy axis",
            },
        )


def _anchor(mean_distance: float) -> str:
    if mean_distance >= 0.6:
        return "far from the reference anchors (verify grounding via the Accuracy axis)"
    if mean_distance >= 0.3:
        return "moderate distance from the reference anchors"
    return "closely matches the reference anchors (restatement-like)"
