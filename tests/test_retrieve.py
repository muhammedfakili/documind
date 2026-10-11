"""A8: Reciprocal Rank Fusion. Pure Python, no database needed."""

from __future__ import annotations

import pytest

from app.retrieve import RRF_K, rrf_fuse
from app.store import Hit


def hit(text: str, score: float = 0.0) -> Hit:
    return Hit(text=text, source="doc.md", title="Doc", section=None, page=None, row=None,
               audience="public", effective_date=None, score=score)


def texts(hits: list[Hit]) -> list[str]:
    return [h.text for h in hits]


def test_chunk_high_in_both_lists_wins():
    vector = [hit("A"), hit("B"), hit("C")]
    keyword = [hit("B"), hit("D"), hit("A")]
    # A: 1/61 + 1/63, B: 1/62 + 1/61 -> B first (2nd and 1st beats 1st and 3rd)
    assert texts(rrf_fuse([vector, keyword]))[:2] == ["B", "A"]


def test_found_by_one_search_still_kept():
    fused = rrf_fuse([[hit("A")], [hit("E-214 row")]])
    assert set(texts(fused)) == {"A", "E-214 row"}  # a union, nothing is lost


def test_raw_scores_are_ignored_only_ranks_count():
    # Keyword scores (BM25) are unbounded, cosine is 0..1: a 50.0 must not drown the other list.
    vector = [hit("A", 0.9), hit("B", 0.8)]
    keyword = [hit("B", 50.0), hit("A", 49.0)]
    fused = rrf_fuse([vector, keyword])
    assert fused[0].score == pytest.approx(fused[1].score)  # 1st+2nd == 2nd+1st


def test_score_is_the_fused_points():
    fused = rrf_fuse([[hit("A")], [hit("A")]])
    assert len(fused) == 1  # the same chunk is merged, not duplicated
    assert fused[0].score == pytest.approx(2 / (RRF_K + 1))


def test_empty_lists():
    assert rrf_fuse([[], []]) == []


# --- A12: the rank record behind "why this source?" -------------------------------------


def test_named_lists_record_each_rank():
    fused = rrf_fuse([[hit("A"), hit("B")], [hit("B")]], names=("vector", "keyword"))
    ranks = {h.text: h.ranks for h in fused}
    assert ranks == {"A": {"vector": 1}, "B": {"vector": 2, "keyword": 1}}  # A missed by keyword: no entry


def test_ranks_from_earlier_steps_are_kept():
    searched = rrf_fuse([[hit("A"), hit("B")], [hit("B"), hit("A")]], names=("vector", "keyword"))
    reranked = list(reversed(searched))  # pretend the reranker flipped the order
    final = rrf_fuse([searched, reranked], names=("hybrid", "reranker"))
    a = next(h for h in final if h.text == "A")
    assert a.ranks == {"vector": 1, "keyword": 2, "hybrid": 1, "reranker": 2}


def test_without_names_no_ranks_are_invented():
    assert rrf_fuse([[hit("A")]])[0].ranks == {}
