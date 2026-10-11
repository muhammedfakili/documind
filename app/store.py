"""A5: the chunk table in Postgres + pgvector -- where ingest ends and search begins.

One row per chunk: the text, its 384-number vector, and every label from the
back of the card (source, page, audience, date ...). Search is one SQL query:
order rows by vector distance to the question, keep the closest k.

Two things the database adds over the numpy search of A4:
  an HNSW index   finds the nearest vectors without comparing against every row
  a WHERE filter  audience = 'public' keeps internal documents away from customers

A8 adds keyword search on the same table: Postgres full-text search turns every
chunk into a list of word stems (a tsvector, kept in its own indexed column), and
keyword_search() ranks chunks by BM25 -- the scoring classic search engines use.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from app.config import settings
from app.ingest.chunking import Chunk

TABLE = "chunks"
DIM = 384  # vector size of the embedding model; a different model means a new table

# BM25 knobs, the usual defaults. K1: how fast repeating a word stops adding points.
# B: how much a long chunk is penalised (long chunks contain more words by chance).
BM25_K1 = 1.2
BM25_B = 0.75

Audience = Literal["public", "all"]


@dataclass(frozen=True)
class Hit:
    """One search result: a chunk row plus how close it is to the question."""

    text: str
    source: str
    title: str
    section: str | None
    page: int | None
    row: int | None
    audience: str
    effective_date: date | None
    score: float  # cosine similarity, higher = closer
    # A12: where this chunk stood in each ranked list it went through, e.g.
    # {"vector": 4, "keyword": 1, "hybrid": 2, "reranker": 1} -- the UI's "why this
    # source?" panel. retrieve.rrf_fuse fills it; a list that missed the chunk has no entry.
    ranks: dict[str, int] = field(default_factory=dict, compare=False)

    @property
    def location(self) -> str:
        if self.page is not None:
            return f"{self.source}, page {self.page}"
        if self.row is not None:
            return f"{self.source}, row {self.row}"
        return self.source


CONNECT_TIMEOUT = 5  # seconds. A11: with Docker up but the db container stopped, the port
# accepted the connection and then never answered -- one attempt hung for 260 s.


def connect() -> psycopg.Connection:
    conn = psycopg.connect(settings.database_url, autocommit=True, connect_timeout=CONNECT_TIMEOUT)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)  # numpy array <-> vector column, both directions
    # With a WHERE filter, a plain HNSW scan can return fewer than k rows (it
    # finds k nearest, then the filter drops some). Iterative scan keeps
    # searching until k rows pass the filter. Needs pgvector 0.8+.
    conn.execute("SET hnsw.iterative_scan = relaxed_order")
    return conn


def create_table(conn: psycopg.Connection, table: str = TABLE) -> None:
    t = sql.Identifier(table)
    conn.execute(
        sql.SQL(
            """
            CREATE TABLE IF NOT EXISTS {t} (
                id             bigserial PRIMARY KEY,
                text           text NOT NULL,
                embedding      vector({dim}) NOT NULL,
                source         text NOT NULL,
                title          text NOT NULL,
                doc_type       text NOT NULL,
                audience       text NOT NULL CHECK (audience IN ('public', 'internal')),
                effective_date date,
                page           int,
                row_num        int,
                section        text,
                chunk_index    int NOT NULL
            )
            """
        ).format(t=t, dim=sql.Literal(DIM))
    )
    # HNSW = a graph of "neighbour" links between vectors; search walks the
    # graph instead of reading every row. vector_cosine_ops: index for <=>.
    conn.execute(
        sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING hnsw (embedding vector_cosine_ops)").format(
            i=sql.Identifier(f"{table}_embedding_hnsw"), t=t
        )
    )
    # A8: the keyword side. Postgres computes the word stems itself whenever a row
    # is written ("GENERATED ... STORED"), so ingest code does not change. The
    # 'english' config lowercases, drops stop words (the, does, my) and stems
    # (earbuds -> earbud, battery -> batteri). GIN = an inverted index: for every
    # stem, the list of rows containing it -- the index at the back of a book.
    conn.execute(
        sql.SQL(
            "ALTER TABLE {t} ADD COLUMN IF NOT EXISTS tsv tsvector "
            "GENERATED ALWAYS AS (to_tsvector('english', text)) STORED"
        ).format(t=t)
    )
    conn.execute(
        sql.SQL("CREATE INDEX IF NOT EXISTS {i} ON {t} USING gin (tsv)").format(
            i=sql.Identifier(f"{table}_tsv_gin"), t=t
        )
    )


def replace_chunks(
    conn: psycopg.Connection, chunks: Sequence[Chunk], vectors: np.ndarray, table: str = TABLE
) -> int:
    """Swap the whole table for these chunks, in one transaction.

    Re-ingesting everything is the simplest correct update: no stale chunks of
    an edited document stay behind. Searches running meanwhile see the old
    rows until the new ones are committed, never a half-filled table.
    """
    if len(chunks) != len(vectors) or vectors.shape[1] != DIM:
        raise ValueError(f"need one {DIM}-number vector per chunk")
    t = sql.Identifier(table)
    rows = [
        (
            c.text,
            vec,
            c.doc.source,
            c.doc.title,
            c.doc.doc_type,
            c.doc.audience,
            c.doc.effective_date,
            c.doc.page,
            c.doc.row,
            c.section,
            c.index,
        )
        for c, vec in zip(chunks, vectors)
    ]
    with conn.transaction():
        conn.execute(sql.SQL("TRUNCATE {t} RESTART IDENTITY").format(t=t))
        with conn.cursor() as cur:
            cur.executemany(
                sql.SQL(
                    "INSERT INTO {t} (text, embedding, source, title, doc_type, audience, "
                    "effective_date, page, row_num, section, chunk_index) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                ).format(t=t),
                rows,
            )
    return len(rows)


def search(
    conn: psycopg.Connection,
    query_vec: np.ndarray,
    k: int = 5,
    audience: Audience = "public",
    table: str = TABLE,
) -> list[Hit]:
    """The k chunks closest to the question vector.

    audience="public" (the default) never returns internal documents -- the safe
    choice when a caller forgets to say who is asking. Support agents pass "all".
    """
    # <=> is cosine DISTANCE (0 = same direction), so similarity = 1 - distance.
    query = sql.SQL(
        """
        SELECT text, source, title, section, page, row_num, audience, effective_date,
               1 - (embedding <=> %(q)s) AS score
        FROM {t}
        WHERE %(all)s OR audience = 'public'
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
        """
    ).format(t=sql.Identifier(table))
    rows = conn.execute(query, {"q": query_vec, "all": audience == "all", "k": k}).fetchall()
    return [Hit(*row[:8], score=float(row[8])) for row in rows]


def keyword_search(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    audience: Audience = "public",
    table: str = TABLE,
) -> list[Hit]:
    """The k chunks sharing the most (and the rarest) words with the question, by BM25.

    Why not Postgres' own ts_rank? It has no idea how RARE a word is: "pulse" (in
    half the chunks) counts as much as "-214" (in two). BM25 weighs every word by
    rarity (IDF), so an exact error code or model name dominates the ranking.

    Words are OR-ed: a chunk needs only one of them to be a candidate. Postgres'
    default (AND) would demand every question word, which almost no chunk has.
    """
    query = sql.SQL(
        """
        WITH docs AS (               -- the chunks this audience may see
            SELECT id, tsv FROM {t} WHERE %(all)s OR audience = 'public'
        ),
        stats AS (                   -- corpus size and average chunk length (in stems)
            SELECT count(*)::float AS n, avg(length(tsv))::float AS avgdl FROM docs
        ),
        terms AS (                   -- the question's stems, stop words already gone
            SELECT DISTINCT lexeme FROM unnest(to_tsvector('english', %(q)s))
        ),
        matches AS (                 -- every (chunk, question stem) pair that occurs
            SELECT d.id, w.lexeme, array_length(w.positions, 1)::float AS tf,
                   length(d.tsv)::float AS dl
            FROM docs d
            CROSS JOIN LATERAL unnest(d.tsv) AS w
            JOIN terms USING (lexeme)
            WHERE d.tsv @@ (SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | '))
                            FROM terms)  -- uses the GIN index to skip chunks with no match
        ),
        df AS (                      -- in how many chunks each stem occurs
            SELECT lexeme, count(*)::float AS df FROM matches GROUP BY lexeme
        ),
        scored AS (                  -- BM25 = sum over stems of rarity x saturated frequency
            SELECT m.id,
                   sum(ln(1 + (s.n - df.df + 0.5) / (df.df + 0.5))
                       * m.tf * (%(k1)s + 1)
                       / (m.tf + %(k1)s * (1 - %(b)s + %(b)s * m.dl / s.avgdl))) AS score
            FROM matches m JOIN df USING (lexeme) CROSS JOIN stats s
            GROUP BY m.id
        )
        SELECT c.text, c.source, c.title, c.section, c.page, c.row_num, c.audience,
               c.effective_date, scored.score
        FROM scored JOIN {t} c USING (id)
        ORDER BY scored.score DESC, c.id
        LIMIT %(k)s
        """
    ).format(t=sql.Identifier(table))
    params = {"q": question, "all": audience == "all", "k": k, "k1": BM25_K1, "b": BM25_B}
    rows = conn.execute(query, params).fetchall()
    return [Hit(*row[:8], score=float(row[8])) for row in rows]


def count(conn: psycopg.Connection, table: str = TABLE) -> dict[str, int]:
    """Rows per audience, e.g. {"public": 120, "internal": 37}."""
    query = sql.SQL("SELECT audience, count(*) FROM {t} GROUP BY audience").format(t=sql.Identifier(table))
    return dict(conn.execute(query).fetchall())
