import re
from dataclasses import dataclass

_DOI_RE = re.compile(r'10\.\d{4,9}/[^\s,;)\]]+')
_YEAR_RE = re.compile(r'\b(19|20)\d{2}\b')


@dataclass
class Reference:
    raw: str
    doi: str | None = None
    year: int | None = None


def _extract_doi(text: str) -> str | None:
    m = _DOI_RE.search(text)
    return m.group(0).rstrip(".,)") if m else None


def _extract_year(text: str) -> int | None:
    matches = _YEAR_RE.findall(text)
    return int(matches[0]) if matches else None


def parse_reference(raw: str) -> Reference:
    return Reference(raw=raw.strip(), doi=_extract_doi(raw), year=_extract_year(raw))


def parse_references(raws: list[str]) -> list[Reference]:
    return [parse_reference(r) for r in raws]
