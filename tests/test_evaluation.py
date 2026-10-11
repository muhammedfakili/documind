"""Evaluation tests: the exam itself must be fair, and the grading must be right."""

from __future__ import annotations

import pytest

from app.config import DATA_DIR
from app.evaluation import (
    REFUSE_CATEGORIES,
    CachedLLM,
    Question,
    Result,
    evidence_recall,
    first_relevant_rank,
    load_questions,
    parse_verdict,
)
from app.generate import NOT_FOUND, Answer
from app.ingest.chunking import chunk_documents
from app.ingest.pipeline import load_directory
from app.llm import LLMAnswer
from app.store import Hit

CATEGORIES = {"single", "table", "multi", "conflict", "internal", "hidden", "unanswerable"}


def hit(text: str) -> Hit:
    return Hit(text=text, source="x.md", title="X", section=None, page=None, row=None,
               audience="public", effective_date=None, score=0.5)


def test_question_file_is_valid():
    questions = load_questions()
    assert len(questions) >= 40
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        assert q.category in CATEGORIES and q.audience in ("public", "all")
        if q.category in REFUSE_CATEGORIES:
            assert not q.evidence and q.audience == "public"
        else:
            assert q.evidence and q.reference


def test_every_evidence_exists_in_the_documents():
    chunks = chunk_documents(load_directory(DATA_DIR / "raw"))
    for q in load_questions():
        for piece in q.evidence:
            found = [c for c in chunks if any(alt.lower() in c.text.lower() for alt in piece)]
            assert found, f"{q.id}: evidence {piece} is in no chunk"
            if q.audience == "public":
                assert any(c.doc.audience == "public" for c in found), f"{q.id}: evidence only internal"


def test_rank_and_recall():
    hits = [hit("nothing"), hit("price is £169"), hit("ANC on Pulse Pro")]
    evidence = [["£169"], ["Pulse Pro has ANC", "ANC on Pulse Pro"], ["never there"]]
    assert first_relevant_rank(hits, evidence) == 2
    assert evidence_recall(hits, evidence) == pytest.approx(2 / 3)
    assert first_relevant_rank(hits, [["missing"]]) is None


def test_judge_json_is_read_even_with_extra_text():
    text = 'Here you go:\n```json\n{"correctness": "partial", "faithful": true, "reason": "misses X"}\n```'
    assert parse_verdict(text) == {"correctness": "partial", "faithful": True, "reason": "misses X"}
    with pytest.raises(ValueError):
        parse_verdict('{"correctness": "maybe"}')


def make_result(category: str, found: bool, verdict: dict | None, hits: list[Hit]) -> Result:
    q = Question(id="q", category=category, audience="public", question="?", reference="r",
                 evidence=[] if category in REFUSE_CATEGORIES else [["key fact"]])
    ans = Answer(question="?", text="answer [1]" if found else NOT_FOUND, sources=hits, found=found)
    return Result(q, hits, ans, verdict)


def test_scores_and_failure_types():
    right = [hit("the key fact")]
    wrong = [hit("other")]
    ok = {"correctness": "correct", "faithful": True, "reason": ""}
    half = {"correctness": "partial", "faithful": True, "reason": ""}

    assert make_result("single", True, ok, right).failure is None
    assert make_result("single", True, half, right).score == 0.5
    assert make_result("single", True, half, right).failure == "cevap"
    assert make_result("single", True, ok | {"correctness": "wrong"}, wrong).failure == "arama"
    assert make_result("single", False, None, right).failure == "yanlis-ret"
    assert make_result("unanswerable", False, None, wrong).score == 1.0
    assert make_result("unanswerable", True, ok | {"correctness": "wrong"}, wrong).failure == "uydurma"


class CountingLLM:
    name, model = "fake", "fake-1"

    def __init__(self) -> None:
        self.n = 0

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        self.n += 1
        return LLMAnswer(text=f"reply {self.n}", model=self.model, input_tokens=1, output_tokens=1)


def test_cache_answers_the_same_prompt_from_disk(tmp_path):
    inner = CountingLLM()
    llm = CachedLLM(inner, cache_dir=tmp_path)
    first = llm.generate("sys", "prompt A")
    again = llm.generate("sys", "prompt A")
    other = llm.generate("sys", "prompt B")
    assert first.text == again.text == "reply 1"
    assert other.text == "reply 2"
    assert inner.n == 2 and llm.cached == 1
