"""A4: turn text into vectors -- the "meaning fingerprint" search compares.

An embedding model reads a text and returns a fixed-length list of numbers
(384 for bge-small). Texts with similar meaning get vectors pointing in a
similar direction, even when they share no words: "return my earbuds" and
"send back my headphones" end up close, "shipping to Canada" further away.

Used by both pipelines, which is why it lives in app/ and not in app/ingest/:
  ingest  embed_passages(chunk texts)  -> stored in the database (A5)
  query   embed_query(question)        -> compared against the stored vectors

The vectors come back with length 1 (normalized), so cosine similarity is a
plain dot product: a @ b.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np

from app.config import settings

# Windows cannot create the symlinks huggingface_hub prefers; the cache still
# works without them, the warning is just noise.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# bge v1.5 was trained with this instruction in front of short search queries
# (not in front of the documents). scripts/a4_embed.py measures what it is worth.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

MAX_TOKENS = 512  # the model reads at most this many tokens; the rest is silently cut


@lru_cache(maxsize=1)
def _model():
    from fastembed import TextEmbedding

    # Downloaded once (~130 MB) into .cache/models, then loaded from disk.
    return TextEmbedding(settings.embed_model, cache_dir=str(settings.embed_cache_dir))


def embed_passages(texts: list[str]) -> np.ndarray:
    """Vectors for document chunks, shape (len(texts), 384)."""
    return np.array(list(_model().embed(texts)), dtype=np.float32)


def embed_query(question: str, instruction: bool = True) -> np.ndarray:
    """Vector for one search question, shape (384,)."""
    text = QUERY_INSTRUCTION + question if instruction else question
    return np.array(next(iter(_model().embed([text]))), dtype=np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity. Works for one vector against one, or one against many rows."""
    return b @ a / (np.linalg.norm(b, axis=-1) * np.linalg.norm(a))


def token_count(text: str) -> int:
    """Real model tokens, including the [CLS] and [SEP] markers it adds."""
    return len(_model().model.tokenizer.encode(text).ids)
