"""Size of scientific leap: how far is each hypothesis from the nearest known idea?

Searches the literature for each hypothesis's own text — reusing the same
"search + embed + cosine" pattern ``citation_accuracy`` and
``epistemic_calibration`` already use, rather than pre-building a static
reference corpus — and computes cosine distance to the nearest result found.
Reported as a continuous distance score per hypothesis plus a set-level mean,
not the four-way restatement/recombination/extension/novel classification the
framework ultimately wants — that needs "spiked reference points" to anchor an
absolute scale, which is calibration data we don't have yet.

A hypothesis distant from anything found in the literature is not
automatically novel *in a good way* — it must be read alongside the Accuracy
axis: distant *and* well-grounded is a genuine originality signal; distant
*without* grounding may simply be wrong. This metric only measures distance,
not validity.

Degrades gracefully: no embeddings or no literature client -> score is None. A
hypothesis with no search results is excluded from the mean and reported
separately, not silently dropped and not scored via an ungrounded guess.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, Hypothesis, HypothesisSet
from ...core.registry import register
from ...services.embeddings import cosine_similarity
from ...services.literature import PaperRecord


def _paper_text(record: PaperRecord) -> str:
    if record.abstract:
        return f"{record.title or ''} {record.abstract}".strip()
    return record.title or ""


def _nearest_neighbor(hyp: Hypothesis, ctx: Context, limit: int = 5):
    """Returns ``(distance, nearest_record, similarity)``, or ``None`` if no
    usable search results were found."""
    hits = [h for h in ctx.literature.search(hyp.text, limit=limit) if h.title or h.abstract]
    if not hits:
        return None

    texts = [hyp.text] + [_paper_text(h) for h in hits]
    vectors = ctx.embeddings.embed(texts)
    hyp_vec, paper_vecs = vectors[0], vectors[1:]

    similarities = [cosine_similarity(hyp_vec, v) for v in paper_vecs]
    best_i = max(range(len(similarities)), key=lambda i: similarities[i])
    best_sim = similarities[best_i]
    distance = max(0.0, min(1.0, 1.0 - best_sim))
    return distance, hits[best_i], best_sim


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

        assessed = []
        unassessed = []
        for h in outputs:
            result = _nearest_neighbor(h, ctx)
            if result is None:
                unassessed.append({"hypothesis_id": h.id, "rank": h.rank,
                                   "reason": "no_literature_found"})
                continue
            distance, nearest, similarity = result
            assessed.append({
                "hypothesis_id": h.id, "rank": h.rank,
                "distance": round(distance, 4),
                "nearest_neighbor_similarity": round(similarity, 4),
                "nearest_neighbor_title": nearest.title,
            })

        if not assessed:
            return MetricResult(
                metric=self.name, axis=self.axis, score=None,
                evidence={"status": "no_hypotheses_could_be_assessed", "unassessed": unassessed},
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
                "embedding_model": ctx.embeddings.name,
                "note": "distance from existing literature is not itself validity -- "
                        "interpret alongside the Accuracy axis",
            },
        )


def _anchor(mean_distance: float) -> str:
    if mean_distance >= 0.6:
        return "far from existing literature (verify grounding via the Accuracy axis)"
    if mean_distance >= 0.3:
        return "moderate distance from existing literature"
    return "closely matches existing literature (restatement-like)"
