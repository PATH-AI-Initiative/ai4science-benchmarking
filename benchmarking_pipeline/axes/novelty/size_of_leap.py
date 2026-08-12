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

The literature search is bibliographic (keyword/relevance-based), not
semantic, so its top hits can share generic methodological vocabulary
("meta-learning", "model misspecification", "closed-loop") with the
hypothesis while being from a completely unrelated field -- the same failure
mode ``citation_accuracy`` guards against for its existence check. Left
unguarded here, a hypothesis can be scored as "closely matches existing
literature" against a paper about an entirely different subject, which reads
as a false claim of unoriginality rather than an honest "couldn't find
anything relevant." When a judge is available, each search result is checked
for genuine topical relevance (not just vocabulary overlap) before it's
allowed to serve as the nearest neighbor; a hypothesis whose only literature
hits are topically irrelevant is treated the same as "no results found"
rather than silently scored against a wrong match.

Degrades gracefully: no embeddings or no literature client -> score is None.
No judge -> the topical-relevance check is skipped (the best embedding match
is trusted at face value, the old behavior). A hypothesis with no search
results, or none that pass the topical check, is excluded from the mean and
reported separately, not silently dropped and not scored via an ungrounded
guess.
"""

from __future__ import annotations

from ...core.context import Context
from ...core.metric import Axis, MetricResult, SetMetric
from ...core.models import EvaluationRun, Hypothesis, HypothesisSet
from ...core.registry import register
from ...services.embeddings import cosine_similarity
from ...services.literature import PaperRecord
from ...services.relationship_judge import judge_relationship

# "different_topic" listed last: judge_relationship's fallback-on-unparseable-
# verdict, and the conservative choice when topical relevance is ambiguous --
# a nearest-neighbor match isn't trusted on a shrug.
_TOPICAL_MATCH_CHOICES = ["same_topic", "different_topic"]

_TOPICAL_MATCH_INSTRUCTIONS = """\
A bibliographic search for related work on a hypothesis returned this candidate. \
Search relevance ranking is based on shared wording, not subject matter -- \
generic methodological terms (e.g. "meta-learning", "model misspecification", \
"closed-loop", "conformal prediction") can make a paper from a completely \
unrelated field or application domain score as a strong textual match. Choose \
exactly one:
- same_topic: the candidate addresses the same subject matter/research problem \
as the hypothesis (not necessarily identical, but genuinely related work).
- different_topic: the candidate is from an unrelated field or application \
domain, sharing only generic terminology with the hypothesis.
Default to different_topic unless clearly the same topic.
"""


def _judge_topical_match(judge, hypothesis_text: str, candidate_text: str) -> str:
    return judge_relationship(
        judge,
        label_a="Hypothesis", text_a=hypothesis_text,
        label_b="Literature search result", text_b=candidate_text,
        instructions=_TOPICAL_MATCH_INSTRUCTIONS,
        choices=_TOPICAL_MATCH_CHOICES,
    )


def _paper_text(record: PaperRecord) -> str:
    if record.abstract:
        return f"{record.title or ''} {record.abstract}".strip()
    return record.title or ""


_MAX_QUERY_CLAIMS = 3


def _search_query(hyp: Hypothesis, prompt: str) -> str:
    """Builds a search query specific enough to find real prior art.

    A hypothesis's own title is often domain-free on its own (e.g.
    "Conformalized model selection with finite-sample coverage guarantees"
    never says "diffusion MRI"), which starves the search of the one word
    that would separate real prior art from generically-worded papers in
    unrelated fields. The hypothesis's own claim text is the fix, not the
    run's shared prompt: claims routinely name the field explicitly (a
    premise like "existing methods for diffusion MRI optimize acquisition
    for...") *and*, unlike the prompt, differ per hypothesis -- so distinct
    hypotheses searching from the same prompt don't all converge on
    whatever generic paper best matches the prompt's own wording. ``prompt``
    is only used as a fallback when a hypothesis has no claims to draw on
    (e.g. an unreviewed/undecomposed capture).
    """
    extra = " ".join(c.text for c in hyp.claims[:_MAX_QUERY_CLAIMS]) if hyp.claims else prompt
    return f"{hyp.text}. {extra}" if extra else hyp.text


def _nearest_neighbor(hyp: Hypothesis, ctx: Context, prompt: str = "", limit: int = 5):
    """Returns ``(distance, nearest_record, similarity)`` on success, or a
    string failure reason ("no_literature_found" / "no_topically_relevant_match").

    Not used for the embedding-similarity computation below -- that measures
    the hypothesis's own semantic distance from the nearest paper, and
    extending every hypothesis's embedding with extra query text would
    change what's actually being compared.
    """
    query = _search_query(hyp, prompt)
    hits = [h for h in ctx.literature.search(query, limit=limit) if h.title or h.abstract]
    if not hits:
        return "no_literature_found"

    texts = [hyp.text] + [_paper_text(h) for h in hits]
    vectors = ctx.embeddings.embed(texts)
    hyp_vec, paper_vecs = vectors[0], vectors[1:]
    similarities = [cosine_similarity(hyp_vec, v) for v in paper_vecs]
    ranked = sorted(range(len(hits)), key=lambda i: similarities[i], reverse=True)

    if ctx.judge is not None:
        # Lazily check candidates best-match-first; stop at the first one that's
        # genuinely on-topic rather than spending a judge call on every candidate.
        best_i = next(
            (i for i in ranked
             if _judge_topical_match(ctx.judge, query, _paper_text(hits[i])) == "same_topic"),
            None,
        )
        if best_i is None:
            return "no_topically_relevant_match"
    else:
        best_i = ranked[0]

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
            result = _nearest_neighbor(h, ctx, prompt=run.prompt)
            if isinstance(result, str):
                unassessed.append({"hypothesis_id": h.id, "rank": h.rank, "reason": result})
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
