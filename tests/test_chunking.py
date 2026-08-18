from pathlib import Path

import pytest

from app.chunking import (
    chunk_document,
    chunk_text,
    count_tokens,
    normalize,
    split_paragraphs,
)


def test_normalize_collapses_pdf_whitespace():
    assert normalize("a  \t b\r\n\n\n\nc") == "a b\n\nc"


def test_split_paragraphs():
    assert split_paragraphs("one\n\ntwo\n\n\nthree") == ["one", "two", "three"]


def test_empty_text_yields_no_chunks():
    assert chunk_text("   \n\n  ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("A short policy statement.", 100, 20) == [
        "A short policy statement."
    ]


def test_chunks_respect_size_budget():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, chunk_size=120, overlap=20)
    assert len(chunks) > 1
    # Allow one paragraph of slack: a chunk closes only once adding the next
    # paragraph would exceed the budget.
    assert all(count_tokens(c) <= 120 + 60 for c in chunks)


def test_overlap_repeats_trailing_content():
    paragraphs = [f"Distinct sentence number {i} about policy." for i in range(12)]
    chunks = chunk_text("\n\n".join(paragraphs), chunk_size=40, overlap=15)
    assert len(chunks) > 1
    # The tail of chunk N should reappear at the head of chunk N+1.
    assert any(
        chunks[i].split("\n\n")[-1] in chunks[i + 1] for i in range(len(chunks) - 1)
    )


def test_oversized_paragraph_is_split():
    giant = "token " * 800
    chunks = chunk_text(giant, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    assert all(count_tokens(c) <= 100 for c in chunks)


def test_no_overlap_is_allowed():
    text = "\n\n".join(f"Para {i} " + "word " * 30 for i in range(6))
    chunks = chunk_text(text, chunk_size=60, overlap=0)
    assert len(chunks) > 1


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, 150), (100, -1)])
def test_invalid_parameters_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)


def test_chunk_document_produces_stable_ids(tmp_path: Path):
    path = tmp_path / "policy-notes.md"
    path.write_text("\n\n".join(f"Section {i} body text." for i in range(30)))

    first = chunk_document(path, 50, 10)
    second = chunk_document(path, 50, 10)

    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)
    assert first[0].source == "policy-notes.md"
    assert first[0].title == "policy notes"
    assert [c.chunk_index for c in first] == list(range(len(first)))
