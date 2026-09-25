"""Evaluation context.

``Context`` is passed to every metric and carries the shared services a
metric may need -- embeddings, an LLM judge, external database clients --
plus the run configuration. Metrics depend on the protocols defined in
``benchmarking_pipeline.services``, never a concrete backend, so the choice
of embedding model or judge model is swappable and doesn't leak into metric
code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .config import RunConfig

if TYPE_CHECKING:
    from ..services.biodb import BioDatabase
    from ..services.embeddings import EmbeddingModel
    from ..services.literature import LiteratureClient
    from ..services.llm_judge import LLMJudge


@dataclass
class Context:
    config: RunConfig
    embeddings: "EmbeddingModel | None" = None
    judge: "LLMJudge | None" = None
    literature: "LiteratureClient | None" = None
    biodb: "BioDatabase | None" = None
    # Shared cache for citation resolution (existence + title-match), keyed by
    # reference identity -- citation_accuracy and citation_support both need
    # the same resolution before doing anything metric-specific with it, and
    # neither the literature client nor the judge cache calls on their own
    # (see benchmarking_pipeline.axes.accuracy._citation_resolution), so
    # without this a run would hit the literature API and judge twice per
    # citation once both metrics are registered.
    citation_cache: dict = field(default_factory=dict)
    # size_of_leap's fixed anchor set (literature search hits + their
    # embeddings), keyed by the search query -- resolved once per evaluation
    # and reused across every hypothesis and every run in a bundle, rather
    # than re-searching per hypothesis (see axes.novelty.size_of_leap).
    anchor_cache: dict = field(default_factory=dict)
