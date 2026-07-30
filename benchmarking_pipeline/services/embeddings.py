"""Embeddings and cosine similarity — the primitive shared across axes.

``EmbeddingModel`` is a protocol so the concrete backend (a hosted embedding
API, a local sentence-transformer, etc.) can be chosen per run without touching
metric code. For the benchmark to be defensible the chosen backend should be
pinned/versioned and recorded in the run metadata.

``HashingEmbedding`` is a dependency-free, deterministic placeholder so the
pipeline runs out of the box and tests are reproducible. It is a bag-of-tokens
hash, NOT a semantic model — replace it with a real backend before drawing any
scientific conclusions.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingModel(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an ``(n, d)`` array of embeddings for ``texts``."""
        ...


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def pairwise_cosine(vectors: np.ndarray) -> np.ndarray:
    """Full cosine-similarity matrix for a set of row vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = vectors / norms
    return unit @ unit.T


def embed_texts_mean(model: "EmbeddingModel", texts: list[str]) -> np.ndarray:
    """Collapse a set of texts to a single vector (mean of their embeddings).

    Used to represent a whole run's output set as one point, so runs can be
    compared to each other (reproducibility, robustness).
    """
    if not texts:
        return np.zeros(1, dtype=np.float64)
    return model.embed(texts).mean(axis=0)


class HashingEmbedding:
    """Deterministic placeholder embedding. Not semantic — for wiring only."""

    name = "hashing-placeholder"

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                h = int(hashlib.md5(token.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out


# Convenient model identifiers for the scientific backends. SPECTER2 is
# citation-trained on scientific documents and is the recommended default for
# the novelty axis (distance in a literature space); SciBERT is a scientific
# BERT better suited to token tasks than to sentence similarity.
SPECTER2 = "allenai/specter2_base"
SCIBERT = "allenai/scibert_scivocab_uncased"


class SentenceTransformerEmbedding:
    """Local sentence-transformer backend (SPECTER2, SciBERT, general models).

    Requires the ``sentence-transformers`` extra. The model is loaded once and
    reused. Record ``name`` in run metadata so results are traceable to the
    exact model.
    """

    def __init__(self, model_name: str = SPECTER2, *, device: str | None = None):
        from sentence_transformers import SentenceTransformer  # noqa: PLC0415

        self._model = SentenceTransformer(model_name, device=device)
        self.name = f"st:{model_name}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=False),
            dtype=np.float64,
        )


class OpenAIEmbedding:
    """OpenAI embeddings backend (a strong general-purpose comparison point).

    Requires the ``openai`` extra and ``OPENAI_API_KEY`` in the environment.
    """

    def __init__(self, model: str = "text-embedding-3-large"):
        from openai import OpenAI  # noqa: PLC0415

        self._client = OpenAI()
        self.model = model
        self.name = f"openai:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embeddings.create(model=self.model, input=texts)
        return np.asarray([d.embedding for d in resp.data], dtype=np.float64)


class OllamaEmbedding:
    """Local embeddings via a model served by Ollama — no API key, no network call.

    Requires ``pip install ollama`` and a running Ollama server (``ollama
    serve``) with an embedding model pulled, e.g. ``ollama pull nomic-embed-text``.
    Not a scientific-literature model like SPECTER2, but fully local.
    """

    def __init__(self, model: str = "nomic-embed-text", *, host: str | None = None):
        import ollama  # noqa: PLC0415

        self._client = ollama.Client(host=host) if host else ollama.Client()
        self.model = model
        self.name = f"ollama:{model}"

    def embed(self, texts: list[str]) -> np.ndarray:
        resp = self._client.embed(model=self.model, input=texts)
        return np.asarray(resp.embeddings, dtype=np.float64)
