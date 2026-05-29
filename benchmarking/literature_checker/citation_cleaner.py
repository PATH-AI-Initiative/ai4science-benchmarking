import re
from pathlib import Path

from docx import Document

from .citation_extractor import Reference

# Matches lines like "[1] Some Title" or "[12] Some Title"
_REF_LINE_RE = re.compile(r'^\[(\d+)\]\s+(.+)$')

# Trailing source indicators appended by Co-Scientist, e.g. "| medRxiv" or "- diva-portal.org"
_SOURCE_SUFFIX_RE = re.compile(
    r'\s*\|\s*\w+\s*$'             # | medRxiv, | bioRxiv, | Nature, etc.
    r'|\s*-\s*[\w.-]+\.\w{2,10}\s*$',  # - diva-portal.org, - website.com, etc.
    re.IGNORECASE,
)


def _clean_title(raw_title: str) -> str:
    """Strip trailing source hints like '| medRxiv' or '- diva-portal.org'."""
    cleaned = _SOURCE_SUFFIX_RE.sub('', raw_title).strip()
    # Fall back to original if cleaning removed too much (< 10 chars left)
    return cleaned if len(cleaned) >= 10 else raw_title.strip()


def read_co_scientist_refs(path: str | Path) -> list[Reference]:
    """
    Parse a Co-Scientist hypothesis output docx and return a list of References.

    Expects lines of the form:
        [1] Title of the paper
        [2] Another title | medRxiv
    """
    doc = Document(str(path))
    refs: list[Reference] = []

    for para in doc.paragraphs:
        text = para.text.strip()
        m = _REF_LINE_RE.match(text)
        if not m:
            continue
        title = _clean_title(m.group(2))
        refs.append(Reference(raw=title))

    return refs
