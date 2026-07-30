"""Shared helper for reproducibility/robustness: turn one run's output into a
single comparison vector, per ``config.multi_run_comparison`` (see RunConfig).
"""

from __future__ import annotations

import numpy as np

from ...core.context import Context
from ...core.models import EvaluationRun
from ...services.embeddings import embed_texts_mean


def run_vector(run: EvaluationRun, ctx: Context) -> np.ndarray | None:
    """The embedding vector representing ``run``, or ``None`` if it has no
    hypotheses to embed."""
    comparison = ctx.config.multi_run_comparison
    if comparison == "top_hypothesis":
        hyp = run.outputs.at_rank(1)
        if hyp is None:
            return None
        return ctx.embeddings.embed([hyp.text])[0]

    texts = [h.text for h in run.outputs]
    if not texts:
        return None
    return embed_texts_mean(ctx.embeddings, texts)
