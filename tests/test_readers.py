"""Tests for read_document_text's format dispatch, including the plain-text
formats (.md/.markdown/.txt) that need no parsing library at all."""

from __future__ import annotations

import pytest

from benchmarking_pipeline.io.readers import read_document_text


@pytest.mark.parametrize("suffix", [".md", ".markdown", ".txt"])
def test_plain_text_formats_are_read_verbatim(tmp_path, suffix):
    content = "# Hypothesis\n\nSome markdown *content* with formatting."
    path = tmp_path / f"doc{suffix}"
    path.write_text(content)

    assert read_document_text(path) == content


def test_unsupported_extension_raises_with_helpful_message():
    with pytest.raises(ValueError, match="Unsupported document type"):
        read_document_text("/tmp/nonexistent.html")
