"""A9: the reranker's plumbing, with a fake cross-encoder (the real one is a 1 GB download)."""

from __future__ import annotations

import pytest

from app import rerank as rerank_module
from app.rerank import rerank
from app.store import Hit


def hit(text: str, score: float = 0.0) -> Hit:
    return Hit(text=text, source="doc.md", title="Doc", section=None, page=None, row=None,
               audience="public", effective_date=None, score=score)


class FakeCrossEncoder:
    """Scores a chunk by how many of the question's words it contains -- enough to test the sorting."""

    def __init__(self):
        self.calls = []

    def rerank(self, query, documents):
        self.calls.append((query, list(documents)))
        words = set(query.lower().split())
        return [float(len(words & set(d.lower().split()))) for d in documents]


@pytest.fixture()
def fake(monkeypatch):
    model = FakeCrossEncoder()
    monkeypatch.setattr(rerank_module, "_model", lambda name: model)
    return model


def test_best_scored_chunk_comes_first(fake):
    hits = [hit("shipping to canada"), hit("pulse pro e-214 anc calibration"), hit("pulse e-214 battery")]
    ranked = rerank("pulse e-214 battery", hits)
    assert ranked[0].text == "pulse e-214 battery"


def test_score_becomes_the_reranker_score(fake):
    ranked = rerank("pulse battery", [hit("pulse battery", score=0.016)])
    assert ranked[0].score == 2.0  # the RRF score is replaced, not kept


def test_k_cuts_after_sorting(fake):
    hits = [hit("a"), hit("b"), hit("pulse"), hit("c")]
    ranked = rerank("pulse", hits, k=1)
    assert [h.text for h in ranked] == ["pulse"]  # found at position 3, still returned


def test_ties_keep_the_search_order(fake):
    hits = [hit("first"), hit("second"), hit("third")]  # no shared words: all score 0
    assert [h.text for h in rerank("pulse", hits)] == ["first", "second", "third"]


def test_question_and_every_chunk_go_to_the_model_together(fake):
    rerank("is it waterproof", [hit("x"), hit("y")])
    assert fake.calls == [("is it waterproof", ["x", "y"])]  # one call, question paired with each chunk


def test_no_hits_no_model_call(fake):
    assert rerank("anything", []) == []
    assert fake.calls == []
