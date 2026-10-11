"""A3: split Documents into chunks small enough to embed and specific enough to find.

A chunk is what search returns and what the LLM reads, so it has to stand on
its own: a reader who sees only this chunk must know what it is about. Two
strategies are kept side by side so the difference can be measured
(scripts/a3_chunk.py):

  fixed_size_chunks  every N words, with overlap -- the naive baseline
  heading_chunks     one section per chunk, split further only at paragraph
                     and line boundaries, table header repeated, and the
                     document title + section name written on top

Sizes are counted in words, not model tokens: English text is roughly 1.3
tokens per word, and the embedding model (A4) reads up to 512 tokens, so a
150-word chunk stays well inside the limit. A4 re-checks with the real tokenizer.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.ingest.documents import Document


@dataclass(frozen=True)
class Chunk:
    text: str  # what gets embedded and what the LLM reads
    doc: Document  # the card it was cut from: source, page, row, audience, date travel along
    section: str | None  # heading it sits under, None for text before the first heading
    index: int  # position inside its Document: 0, 1, 2 ...

    @property
    def location(self) -> str:
        """Human-readable pointer for citations, e.g. "manuals/pulse-user-manual.pdf, page 3"."""
        return self.doc.location

    def to_dict(self) -> dict:
        # The Document's labels, with the chunk's own text in place of the whole page.
        return self.doc.to_dict() | {"text": self.text, "section": self.section, "index": self.index}


def word_count(text: str) -> int:
    return len(text.split())


# ---------------------------------------------------------------------------
# Strategy 1: fixed size (baseline)
# ---------------------------------------------------------------------------


def fixed_size_chunks(doc: Document, size: int = 100, overlap: int = 20) -> list[Chunk]:
    """Cut every `size` words; consecutive chunks share `overlap` words.

    Blind to structure: it cuts through sentences, table rows and headings, and
    only the first chunk of a document still contains its title.
    """
    if not 0 <= overlap < size:
        raise ValueError("overlap must be at least 0 and smaller than size")
    words = doc.text.split()
    step = size - overlap
    chunks: list[Chunk] = []
    start = 0
    while True:
        piece = " ".join(words[start : start + size])
        chunks.append(Chunk(text=piece, doc=doc, section=None, index=len(chunks)))
        if start + size >= len(words):
            return chunks
        start += step


# ---------------------------------------------------------------------------
# Strategy 2: by heading (what we use)
# ---------------------------------------------------------------------------


def with_context(title: str, section: str | None, body: str) -> str:
    """Write the document title and section name on top of the chunk body.

    Without it, "Code: E-214 | Meaning: ANC microphone calibration failed" could
    belong to any product -- and E-214 means something different on Pulse.
    Search only sees chunk.text, so the context has to be written into it.
    """
    if section is None:
        return f"{title}\n{body}"
    return f"{title} > {section}\n{body}"


def split_sections(text: str) -> list[tuple[str | None, str]]:
    """Split at "## " headings into (heading, body) pairs.

    The "# " document title is dropped: every chunk gets the title through
    with_context, so keeping it in the body would only repeat it.
    """
    sections: list[tuple[str | None, list[str]]] = [(None, [])]
    for line in text.splitlines():
        if line.startswith("## "):
            sections.append((line[3:].strip(), []))
        elif not line.startswith("# "):
            sections[-1][1].append(line)
    return [(heading, "\n".join(lines).strip()) for heading, lines in sections if "\n".join(lines).strip()]


def _blocks(body: str) -> list[str]:
    """Paragraphs, lists and tables: runs of lines separated by a blank line.

    A FAQ question and its answer, or a table, are one block -- the smallest
    unit we try never to cut.
    """
    return [block.strip() for block in body.split("\n\n") if block.strip()]


def _split_block(block: str, max_words: int) -> list[str]:
    """Split one oversized block at line boundaries.

    If it is a Markdown table, every piece starts with the header row again, so
    "| Earbuds | 30 days | 14 days |" never loses the "Unopened | Opened" it
    depends on.
    """
    lines = block.splitlines()
    is_table = len(lines) > 2 and lines[0].startswith("|") and set(lines[1]) <= set("|-: ")
    header = lines[:2] if is_table else []
    rows = lines[2:] if is_table else lines

    pieces: list[str] = []
    current: list[str] = []
    for line in rows:
        if current and word_count("\n".join(header + current + [line])) > max_words:
            pieces.append("\n".join(header + current))
            current = []
        current.append(line)
    if current:
        pieces.append("\n".join(header + current))
    return pieces


def _pack(blocks: list[str], max_words: int) -> list[str]:
    """Greedily put whole blocks together until the next one would not fit."""
    pieces: list[str] = []
    current: list[str] = []
    for block in blocks:
        if word_count(block) > max_words:
            if current:
                pieces.append("\n\n".join(current))
                current = []
            pieces.extend(_split_block(block, max_words))
            continue
        if current and word_count("\n\n".join(current + [block])) > max_words:
            pieces.append("\n\n".join(current))
            current = []
        current.append(block)
    if current:
        pieces.append("\n\n".join(current))
    return pieces


def heading_chunks(doc: Document, max_words: int = 150, context: bool = True) -> list[Chunk]:
    """One chunk per section; long sections are split at block, then line, boundaries.

    `context=False` leaves out the title/section header -- only so the A3 report
    can show what that header is worth.
    """
    chunks: list[Chunk] = []
    for section, body in split_sections(doc.text):
        if section == doc.title:
            section = None  # PDFs repeat their title as the first bold heading
        for piece in _pack(_blocks(body), max_words):
            chunks.append(
                Chunk(
                    text=with_context(doc.title, section, piece) if context else piece,
                    doc=doc,
                    section=section,
                    index=len(chunks),
                )
            )
    return chunks


def chunk_documents(docs: list[Document], max_words: int = 150) -> list[Chunk]:
    """The ingest step after loading: every Document -> its heading chunks."""
    return [chunk for doc in docs for chunk in heading_chunks(doc, max_words)]
