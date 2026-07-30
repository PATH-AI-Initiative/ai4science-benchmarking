"""Shared, cross-cutting capabilities used by more than one axis.

The framework notes that several metrics rely on the same underlying operation —
embed two pieces of text and compute their cosine similarity — differing only in
what each output is compared against (the literature, a perturbed twin, a
repeated run). That primitive lives in :mod:`embeddings`. LLM-as-judge and
external database access live alongside it.
"""
