"""A9: re-sort the search results with a cross-encoder -- a model that READS question and chunk together.

The searches so far never look at the question and a chunk at the same time:
  vector   question and chunk are turned into vectors SEPARATELY (bi-encoder),
           then compared. Fast (chunk vectors are computed once, at ingest),
           but every detail of a chunk is squeezed into 384 numbers in advance.
  keyword  counts shared words, understands nothing.

A cross-encoder gets "question [SEP] chunk" as ONE input and returns one number:
how well this chunk answers this question. It can notice that a chunk is about
Pulse Pro while the question asks about Pulse, or that "other countries" covers
Japan. The price: one model run per (question, chunk) pair at question time,
so it only re-sorts a short list the fast searches already found.

    question ─▶ hybrid search (fast, 157 → 20) ─▶ cross-encoder (slow, 20 → 5) ─▶ LLM
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import lru_cache

from app.config import settings
from app.store import Hit

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")  # same as embedding.py (Windows)


@lru_cache(maxsize=4)
def _model(name: str):
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    # Downloaded once into .cache/models, like the embedding model.
    return TextCrossEncoder(name, cache_dir=str(settings.embed_cache_dir))


def rerank(question: str, hits: list[Hit], k: int | None = None, model: str | None = None) -> list[Hit]:
    """`hits` sorted by the cross-encoder's relevance score, best first; the first k if given.

    Hit.score becomes the cross-encoder score (a raw logit: higher = more relevant,
    no fixed range). Ties keep the incoming order, so the search ranking breaks them.
    """
    if not hits:
        return []
    scores = list(_model(model or settings.rerank_model).rerank(question, [h.text for h in hits]))
    order = sorted(range(len(hits)), key=lambda i: -scores[i])  # stable: ties keep search order
    ranked = [replace(hits[i], score=float(scores[i])) for i in order]
    return ranked[:k] if k is not None else ranked
