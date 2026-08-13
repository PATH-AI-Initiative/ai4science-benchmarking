"""Curated biological database lookups (UniProt, KEGG, ...).

Used by the Accuracy axis to validate that named entities — genes, proteins,
pathways — are real and correctly related. Behind a protocol so metrics can run
against a fake in tests and so additional databases can be added later.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import httpx


@dataclass
class EntityRecord:
    query: str
    normalized_id: str | None
    canonical_name: str | None
    exists: bool


@runtime_checkable
class BioDatabase(Protocol):
    name: str

    def resolve(self, name: str, kind: str | None = None, organism: str | None = None) -> EntityRecord:
        """Resolve an entity name against the database.

        ``organism``, if given, is a hint (e.g. a species entity co-occurring
        in the same claim) -- implementations should prefer a match qualified
        by it but fall back to an unqualified lookup rather than reporting
        "doesn't exist" just because a thinly-annotated organism's entry
        doesn't happen to be indexed under that exact organism filter.
        """
        ...


def _uniprot_canonical_name(entry: dict) -> str | None:
    """Best-effort human-readable name from a UniProtKB entry's nested JSON.

    Non-model organisms (e.g. most Plasmodium species other than falciparum)
    are thinly annotated: a reviewed entry's ``recommendedName`` is common for
    well-studied proteins, but many hits are unreviewed (TrEMBL) records with
    only a ``submissionNames`` entry, or no protein name at all -- just a gene
    symbol or ORF/systematic identifier. Try each in the order most likely to
    give a real name, and settle for the raw identifier only as a last resort.
    """
    description = entry.get("proteinDescription", {})
    recommended = description.get("recommendedName", {}).get("fullName", {}).get("value")
    if recommended:
        return recommended
    submitted = description.get("submissionNames", [])
    if submitted:
        name = submitted[0].get("fullName", {}).get("value")
        if name:
            return name
    for gene in entry.get("genes", []):
        gene_name = gene.get("geneName", {}).get("value")
        if gene_name:
            return gene_name
        for orf in gene.get("orfNames", []):
            if orf.get("value"):
                return orf["value"]
    return None


class UniProtClient:
    """UniProtKB's public REST API. Free, no API key required.

    Uses a broad free-text query rather than a field-scoped one (e.g.
    ``gene:DHODH``): non-model organisms are frequently indexed under a
    systematic/ORF identifier (like PlasmoDB's ``PKNH_1436200``) rather than a
    canonical gene symbol, and a field-scoped query on ``gene`` finds nothing
    for those -- confirmed empirically against the live API before writing
    this client, not assumed.

    A bare name search (e.g. "DHODH" or "dihydroorotate dehydrogenase") isn't
    species-specific -- it can resolve to *some* organism's ortholog, not
    necessarily the one the claim is actually about, since UniProt indexes
    orthologs across every organism it covers as separate records. When
    ``organism`` is given, an organism-qualified query is tried first; if that
    finds nothing (a thinly-annotated organism's entry may not be indexed
    under that exact organism filter), it falls back to the bare query rather
    than reporting "doesn't exist" over a filter that was too strict.
    """

    name = "uniprot"
    _BASE = "https://rest.uniprot.org/uniprotkb/search"

    def __init__(self, timeout: float = 10.0, client: httpx.Client | None = None):
        self._client = client or httpx.Client(timeout=timeout)

    def resolve(self, name: str, kind: str | None = None, organism: str | None = None) -> EntityRecord:
        if organism:
            qualified = self._search(name, query=f'{name} AND organism_name:"{organism}"')
            if qualified is not None:
                return qualified
        return self._search(name, query=name) or EntityRecord(
            query=name, normalized_id=None, canonical_name=None, exists=False,
        )

    def _search(self, name: str, *, query: str) -> EntityRecord | None:
        """Runs ``query`` against the API. Returns ``None`` (not a "doesn't
        exist" EntityRecord) on no results or an HTTP error, so the caller
        can distinguish "this specific query found nothing" from "this
        entity doesn't exist at all" and fall back to a broader query."""
        params = {
            "query": query,
            "format": "json",
            "fields": "accession,gene_names,protein_name,organism_name",
            "size": 1,
        }
        try:
            resp = self._client.get(self._BASE, params=params)
            resp.raise_for_status()
        except httpx.HTTPError:
            return None

        results = resp.json().get("results", [])
        if not results:
            return None

        entry = results[0]
        return EntityRecord(
            query=name,
            normalized_id=entry.get("primaryAccession"),
            canonical_name=_uniprot_canonical_name(entry),
            exists=True,
        )
