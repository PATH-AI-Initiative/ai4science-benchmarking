"""Curated biological database lookups (UniProt, KEGG, ...).

Used by the Accuracy axis to validate that named entities — genes, proteins,
pathways — are real and correctly related. Behind a protocol so metrics can run
against a fake in tests and so additional databases can be added later.

Concrete HTTP-backed clients are not implemented yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class EntityRecord:
    query: str
    normalized_id: str | None
    canonical_name: str | None
    exists: bool


@runtime_checkable
class BioDatabase(Protocol):
    name: str

    def resolve(self, name: str, kind: str | None = None) -> EntityRecord:
        """Resolve an entity name against the database."""
        ...
