"""Document loading and token-aware chunking.

Chunking is deliberately pure/offline so it can be tuned and unit-tested without
touching Azure. Chunk size and overlap were the two knobs that moved answer
quality most, so they live in settings rather than being hard-coded here.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import tiktoken

# cl100k_base is close enough to the tokenizer behind the Azure OpenAI embedding
# deployments for the purpose of sizing chunks.
_ENCODER = tiktoken.get_encoding("cl100k_base")

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass
class Chunk:
    id: str
    content: str
    source: str
    title: str
    chunk_index: int
    token_count: int
    embedding: list[float] = field(default_factory=list)

    def to_search_document(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "source": self.source,
            "title": self.title,
            "chunk_index": self.chunk_index,
            "content_vector": self.embedding,
        }


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def normalize(text: str) -> str:
    """Collapse the whitespace noise that PDF extraction leaves behind."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def _split_oversized(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """Slice a single paragraph that is larger than one chunk, by tokens."""
    tokens = _ENCODER.encode(paragraph)
    step = max(chunk_size - overlap, 1)
    pieces = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        pieces.append(_ENCODER.decode(window).strip())
        if start + chunk_size >= len(tokens):
            break
    return pieces


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Group paragraphs into ~chunk_size token windows with a token overlap.

    Paragraph boundaries are respected where possible: splitting mid-sentence was
    a common cause of retrieved chunks that looked relevant but did not contain
    the answer.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    text = normalize(text)
    if not text:
        return []

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in split_paragraphs(text):
        p_tokens = count_tokens(paragraph)

        if p_tokens > chunk_size:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_split_oversized(paragraph, chunk_size, overlap))
            continue

        if current_tokens + p_tokens > chunk_size and current:
            chunks.append("\n\n".join(current))
            # Carry the tail of the previous chunk forward so a fact split across
            # the boundary stays retrievable from both sides.
            current, current_tokens = _carry_overlap(current, overlap)

        current.append(paragraph)
        current_tokens += p_tokens

    if current:
        chunks.append("\n\n".join(current))

    return [c for c in (c.strip() for c in chunks) if c]


def _carry_overlap(paragraphs: list[str], overlap: int) -> tuple[list[str], int]:
    if overlap == 0:
        return [], 0
    carried: list[str] = []
    total = 0
    for paragraph in reversed(paragraphs):
        p_tokens = count_tokens(paragraph)
        if total + p_tokens > overlap:
            break
        carried.insert(0, paragraph)
        total += p_tokens
    return carried, total


def read_document(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def chunk_document(path: Path, chunk_size: int, overlap: int) -> list[Chunk]:
    text = read_document(path)
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    chunks = []
    for index, body in enumerate(chunk_text(text, chunk_size, overlap)):
        digest = hashlib.sha1(f"{path.name}:{index}".encode()).hexdigest()[:16]
        chunks.append(
            Chunk(
                id=f"{digest}",
                content=body,
                source=path.name,
                title=title,
                chunk_index=index,
                token_count=count_tokens(body),
            )
        )
    return chunks


def iter_documents(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path
