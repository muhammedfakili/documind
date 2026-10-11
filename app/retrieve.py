"""The first step of the query pipeline -- question in, best chunks out.

Two searches look at the question from different sides (A8):
  vector   finds chunks with the same MEANING ("pairing memory full" ~ "too many devices")
  keyword  finds chunks with the same WORDS (exact codes like E-214, model names, numbers)
Reciprocal Rank Fusion merges the two ranked lists into one. Then a cross-encoder
reads the question together with each of the top candidates and re-sorts them (A9).
Its order does not replace the search order, it is fused with it (RRF again):
alone, the reranker sometimes pushes the right chunk down (scripts/a9_rerank.py).
Callers keep calling retrieve().

    question ─┬─ vector  (30) ─┐
              └─ keyword (30) ─┴─ RRF ─▶ top 10 ─┬────────────────────┐
                                                  └─ reranker ─▶ order ─┴─ RRF ─▶ top k

A10, the second try (generate.answer asks for it only after a "not found"): an LLM rewrites
the question into a few search queries in the documents' words (app/rewrite.py). The question and every rewrite run the hybrid
search above on their own, their lists are fused with RRF, and the reranker then
reads the fused top 10 against the ORIGINAL question.

    question ──────────┬─ hybrid ─┐
    rewrite 1, 2, 3 ───┴─ hybrid ─┴─ RRF ─▶ top 10 ─▶ reranker (original question) ─▶ ...
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Literal

import psycopg

from app.embedding import embed_query
from app.llm import LLMProvider
from app.rerank import rerank as cross_encoder_rerank
from app.rewrite import rewrite_query
from app.store import Audience, Hit, keyword_search, search

Mode = Literal["vector", "keyword", "hybrid"]

RRF_K = 60  # the constant from the original RRF paper (Cormack et al., 2009)
CANDIDATES = 30  # how deep each search looks before fusing
RERANK_CANDIDATES = 10  # how many fused results the reranker reads (A9: scripts/a9_rerank.py)


def rrf_fuse(
    rankings: Sequence[Sequence[Hit]], k: int = RRF_K, names: Sequence[str] | None = None
) -> list[Hit]:
    """Merge ranked lists: every list gives each chunk 1 / (k + its rank) points.

    Only ranks count, never the raw scores -- cosine similarity (0..1) and keyword
    scores (unbounded) are not on the same scale, ranks always are. A chunk near
    the top of both lists beats one that is first in only one of them.

    names (A12): one label per list, e.g. ("vector", "keyword"). Each chunk's rank in
    every named list is written into Hit.ranks, next to the ranks it already carried
    from earlier steps -- that record is what the UI shows as "why this source?".
    """
    points: dict[tuple, float] = {}
    first_seen: dict[tuple, Hit] = {}
    ranks: dict[tuple, dict[str, int]] = {}
    for i, ranking in enumerate(rankings):
        for rank, hit in enumerate(ranking, start=1):
            key = (hit.source, hit.page, hit.row, hit.text)  # the same chunk in both lists
            points[key] = points.get(key, 0.0) + 1.0 / (k + rank)
            first_seen.setdefault(key, hit)
            record = ranks.setdefault(key, {})
            for name, earlier in hit.ranks.items():
                record.setdefault(name, earlier)  # the first list that saw the chunk wins
            if names:
                record[names[i]] = rank
    order = sorted(points, key=points.__getitem__, reverse=True)
    return [replace(first_seen[key], score=points[key], ranks=ranks[key]) for key in order]


def retrieve(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    audience: Audience = "public",
    mode: Mode = "hybrid",
    rerank: bool = True,
    rewriter: LLMProvider | None = None,
) -> list[Hit]:
    """The k best chunks for `question`.

    audience="public": a customer is asking, internal documents are never returned.
    audience="all":    a support agent is asking, everything is searchable.
    mode: "hybrid" (default) fuses both searches; "vector" / "keyword" run one alone,
          which the evaluation uses to compare them.
    rerank: let the cross-encoder vote on the order of the top RERANK_CANDIDATES (default).
          Off, the search order is returned as is -- how A5-A8 worked.
    rewriter: an LLM that rewrites the question into extra search queries (A10).
          None (default) = search with the question only, no LLM call.
    """
    depth = max(k, RERANK_CANDIDATES) if rerank else k
    queries = [question] + (rewrite_query(question, rewriter) if rewriter else [])
    if len(queries) == 1:
        hits = _search(conn, question, depth, audience, mode)
    else:
        hits = rrf_fuse([_search(conn, q, max(depth, CANDIDATES), audience, mode) for q in queries])[:depth]
    if not rerank:
        return hits[:k]
    return rrf_fuse([hits, cross_encoder_rerank(question, hits)], names=("hybrid", "reranker"))[:k]


def _search(conn: psycopg.Connection, question: str, k: int, audience: Audience, mode: Mode) -> list[Hit]:
    if mode == "vector":
        return search(conn, embed_query(question), k=k, audience=audience)
    if mode == "keyword":
        return keyword_search(conn, question, k=k, audience=audience)
    depth = max(k, CANDIDATES)
    vector_hits = search(conn, embed_query(question), k=depth, audience=audience)
    keyword_hits = keyword_search(conn, question, k=depth, audience=audience)
    return rrf_fuse([vector_hits, keyword_hits], names=("vector", "keyword"))[:k]
