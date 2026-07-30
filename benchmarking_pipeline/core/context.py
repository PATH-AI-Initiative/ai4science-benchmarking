"""Evaluation context.

The ``Context`` is passed to every metric and carries the shared services a
metric may need — embeddings, an LLM judge, external database clients — plus the
run configuration. Metrics depend on the *protocols* defined in
``benchmarking_pipeline.services``, never on a concrete backend, so the choice of
embedding model or judge model is swappable and does not leak into metric code.
"""

from __future__ import annotations

from dataclasses import dataclass
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
