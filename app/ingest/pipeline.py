"""The ingest pipeline: folder of files -> Documents -> chunks -> vectors -> database."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

import psycopg

from app import store
from app.embedding import embed_passages
from app.ingest.chunking import chunk_documents
from app.ingest.documents import Document
from app.ingest.loaders import load_csv, load_markdown, load_pdf, load_xlsx

log = logging.getLogger(__name__)

Loader = Callable[[Path, Path], list[Document]]

LOADERS: dict[str, Loader] = {
    ".md": load_markdown,
    ".pdf": load_pdf,
    ".csv": load_csv,
    ".xlsx": load_xlsx,
}


def load_directory(root: Path) -> list[Document]:
    """Load every supported file under `root`, in a stable (sorted) order.

    Unsupported files are skipped with a warning rather than failing the run:
    a client folder always contains something unexpected, and one stray .docx
    should not block indexing the other 2,000 files.
    """
    documents: list[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            log.warning("Skipping unsupported file: %s", path)
            continue
        documents.extend(loader(path, path.relative_to(root)))
    return documents


def ingest_directory(conn: psycopg.Connection, root: Path) -> int:
    """Run the whole ingest pipeline and replace what the database holds.

    Every step is one call into the module that owns it; this function only
    connects them in order. Returns the number of chunks stored.
    """
    documents = load_directory(root)  # A2: files -> Documents
    chunks = chunk_documents(documents)  # A3: Documents -> Chunks
    vectors = embed_passages([c.text for c in chunks])  # A4: Chunks -> vectors
    store.create_table(conn)  # A5: -> Postgres + pgvector
    return store.replace_chunks(conn, chunks, vectors)
