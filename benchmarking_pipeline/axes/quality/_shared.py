"""Shared helpers for reproducibility/robustness: turn one run's output into a
single comparison vector, per ``config.multi_run_comparison`` (see
``RunConfig``), and (for ``top_hypothesis`` mode) a judge-based, rank-aware
check for whether two runs' top ideas are the same core proposal -- not just
whether rank-1 matches, but whether a run's top idea survives anywhere in the
other run's top-k (``config.top_k_hypotheses``), credited by how far it
dropped.
"""

from __future__ import annotations

import numpy as np

from ...core.context import Context
from ...core.models import EvaluationRun, Hypothesis
from ...services.embeddings import embed_texts_mean
from ...services.relationship_judge import judge_relationship


def hypothesis_text(hyp: Hypothesis) -> str:
    """``hyp.text`` enriched with its claims.

    Extraction often leaves ``hyp.text`` as just a short title (e.g. "H1 --
    Force the heme feed"), with the substantive content in its claims
    instead -- a bare title gives embedding similarity and the
    mechanism-match judge little to compare on.
    """
    if not hyp.claims:
        return hyp.text
    claim_text = " ".join(c.text for c in hyp.claims)
    return f"{hyp.text}. {claim_text}"


def top_k_hypotheses(outputs, k: int) -> list[Hypothesis]:
    """The k best-ranked hypotheses in ``outputs`` (rank 1 = top; unranked
    sort last) -- same ordering rule as ``core.scoring._top_k``, duplicated
    here rather than imported since that one operates on post-scoring
    ``HypothesisScore`` objects, not the raw ``Hypothesis`` list this module
    works with pre-scoring."""
    ordered = sorted(
        outputs.hypotheses, key=lambda h: (h.rank is None, h.rank if h.rank is not None else 0)
    )
    return ordered[:k]


def run_vector(run: EvaluationRun, ctx: Context) -> np.ndarray | None:
    """The embedding vector representing ``run``, or ``None`` if it has no
    hypotheses to embed."""
    comparison = ctx.config.multi_run_comparison
    if comparison == "top_hypothesis":
        hyp = run.outputs.at_rank(1)
        if hyp is None:
            return None
        return ctx.embeddings.embed([hypothesis_text(hyp)])[0]

    texts = [hypothesis_text(h) for h in run.outputs]
    if not texts:
        return None
    return embed_texts_mean(ctx.embeddings, texts)


# "different_mechanism" listed last: judge_relationship's fallback-on-
# unparseable-verdict, and the conservative choice when it's ambiguous
# whether two proposals are really the same one.
_MECHANISM_MATCH_CHOICES = ["same_mechanism", "different_mechanism"]

_MECHANISM_MATCH_INSTRUCTIONS = """\
Two runs of the identical prompt each produced a top-ranked hypothesis. \
Embedding similarity can read two hypotheses as close just because they're in \
the same broad research area (e.g. both about antimalarial drug resistance), \
even when they propose entirely different specific interventions -- a \
repurposed kinase inhibitor and an established antimalarial combination can \
score as similar on topic alone despite being nothing alike as proposals. \
Choose exactly one:
- same_mechanism: both hypotheses propose essentially the same core \
intervention (the same drug, drug combination, or target), allowing for \
differences in wording or supporting detail.
- different_mechanism: the hypotheses propose genuinely different \
interventions or mechanisms, even if topically related.
Default to different_mechanism unless clearly the same core proposal.
"""


def judge_mechanism_match(judge, hyp_a_text: str, hyp_b_text: str) -> str:
    """Are ``hyp_a_text`` and ``hyp_b_text`` genuinely the same core proposal?

    In ``top_hypothesis`` mode, this is the scored signal for reproducibility/
    robustness when a judge is available -- embedding similarity barely
    discriminates "same research question" from "same specific proposal"
    (two unrelated questions embed ~0.83 cosine similarity; two runs
    proposing different drugs for the identical question still embed
    0.91-0.96), so it's reported as context rather than trusted as the
    discriminating number.
    """
    return judge_relationship(
        judge,
        label_a="Hypothesis from run A", text_a=hyp_a_text,
        label_b="Hypothesis from run B", text_b=hyp_b_text,
        instructions=_MECHANISM_MATCH_INSTRUCTIONS,
        choices=_MECHANISM_MATCH_CHOICES,
    )


def mechanism_match_credit(
    judge, top_a: list[Hypothesis], top_b: list[Hypothesis],
) -> tuple[float, dict]:
    """How well does A's rank-1 idea survive anywhere in B's top-k, and vice
    versa -- not just whether A's rank-1 equals B's rank-1.

    A top idea dropping from rank 1 to rank 2 is meaningfully different from
    vanishing entirely, but a strict rank-1-vs-rank-1 comparison scores both
    as a flat miss. Checks A's rank-1 against every hypothesis in B's top-k
    (and vice versa), crediting a match at rank r as ``1/r``. The two
    directional credits are averaged for a single symmetric score.

    A[0] vs B[0] is asked only once and reused for both directions, since
    ``judge_mechanism_match`` is assumed direction-symmetric.
    """
    a_to_b = [judge_mechanism_match(judge, hypothesis_text(top_a[0]), hypothesis_text(hb)) for hb in top_b]
    b_to_a = [
        a_to_b[0] if i == 0 else judge_mechanism_match(judge, hypothesis_text(top_b[0]), hypothesis_text(ha))
        for i, ha in enumerate(top_a)
    ]

    def _credit(verdicts: list[str]) -> float:
        for rank, verdict in enumerate(verdicts, start=1):
            if verdict == "same_mechanism":
                return round(1.0 / rank, 4)
        return 0.0

    credit_a_to_b, credit_b_to_a = _credit(a_to_b), _credit(b_to_a)
    return (
        round((credit_a_to_b + credit_b_to_a) / 2, 4),
        {"a_rank1_vs_b": a_to_b, "b_rank1_vs_a": b_to_a,
         "credit_a_to_b": credit_a_to_b, "credit_b_to_a": credit_b_to_a},
    )


def mechanism_anchor(
    rate: float, mean_sim: float,
    labels: tuple[str, str, str] = ("highly reproducible", "reproducible with some variation", "low reproducibility"),
) -> str:
    """Human-readable anchor for a mechanism-match-rate score, e.g. from
    :func:`judge_mechanism_match` aggregated across run pairs. Always names
    the embedding similarity alongside it, since it's still worth seeing --
    just not as the thing being scored. ``labels`` are (high, mid, low) tier
    names -- override for a caller whose vocabulary differs (e.g.
    robustness's "stable under rewording")."""
    high, mid, low = labels
    if rate >= 0.8:
        base = high
    elif rate >= 0.5:
        base = mid
    else:
        base = low
    return (
        f"{base} (mechanism-verified; embedding similarity alone reads {round(mean_sim, 4)}, "
        "which reflects shared topic more than shared mechanism)"
    )
