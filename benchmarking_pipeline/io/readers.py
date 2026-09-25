"""Extract plain text from exported tool reports (docx, PDF, markdown, plain text).

Step one of capturing a tool's output: pull raw text out of whatever document
format the tool exported, before structured extraction
(:mod:`benchmarking_pipeline.io.extraction`) turns it into a
:class:`~benchmarking_pipeline.core.models.HypothesisSet`.
"""

from __future__ import annotations

from pathlib import Path

_PLAIN_TEXT_SUFFIXES = {".md", ".markdown", ".txt"}


def read_docx_text(path: str | Path) -> str:
    from docx import Document  # noqa: PLC0415

    doc = Document(str(path))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())


def read_pdf_text(path: str | Path) -> str:
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_document_text(path: str | Path) -> str:
    """Dispatch on file extension. Supports .docx, .pdf, .md/.markdown, and .txt."""
    suffix = Path(path).suffix.lower()
    if suffix == ".docx":
        return read_docx_text(path)
    if suffix == ".pdf":
        return read_pdf_text(path)
    if suffix in _PLAIN_TEXT_SUFFIXES:
        return Path(path).read_text()
    raise ValueError(
        f"Unsupported document type: {suffix} "
        f"(expected .docx, .pdf, .md/.markdown, or .txt)"
    )
