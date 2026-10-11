"""Embedding tests. They run the real local model (downloaded once into .cache/models)."""

from __future__ import annotations

import numpy as np

from app.embedding import MAX_TOKENS, cosine, embed_passages, embed_query, token_count


def test_vectors_have_384_numbers_and_length_one():
    vectors = embed_passages(["How do I return my earbuds?", "Battery life is too short."])
    assert vectors.shape == (2, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-3)
    assert embed_query("Can I swim with Pulse?").shape == (384,)


def test_same_meaning_is_closer_than_other_topic():
    returns, send_back, shipping = embed_passages(
        ["How do I return my earbuds?", "Can I send back my headphones?", "What does shipping to Canada cost?"]
    )
    assert cosine(returns, send_back) > cosine(returns, shipping) + 0.2


def test_cosine_equals_dot_product_for_normalized_vectors():
    a, b = embed_passages(["first sentence", "second sentence"])
    assert np.isclose(cosine(a, b), a @ b, atol=1e-5)


def test_cosine_against_many_rows():
    rows = embed_passages(["a dog", "a cat", "a car"])
    scores = cosine(rows[0], rows)
    assert scores.shape == (3,)
    assert np.isclose(scores[0], 1.0, atol=1e-5)


def test_search_finds_the_right_chunk_among_others():
    chunks = [
        "Lumora Shipping Policy > Shipping options\nExpress to the United Kingdom: next business day, £9.99.",
        "Lumora Return Policy > Return window\nOpened earbuds can be returned within 14 days.",
        "Lumora Aria X2 - User Manual > Party Mode\nUp to 4 Aria X2 speakers can be linked.",
    ]
    scores = embed_passages(chunks) @ embed_query("How long can I send back earbuds I opened?")
    assert int(np.argmax(scores)) == 1


def test_token_count_uses_model_tokenizer():
    assert token_count("Lumora") == 4  # [CLS] lu ##mora [SEP]
    assert token_count("word " * 2000) == MAX_TOKENS  # longer text is cut at the model's limit
