"""Answer generation tests. A fake LLM and a fake retrieve(): no database, no API key."""

from __future__ import annotations

from dataclasses import replace
from datetime import date

import pytest

from app import generate
from app.generate import NOT_FOUND, build_prompt, is_not_found, parse_citations
from app.llm import LLMAnswer
from app.store import Hit


def hit(text: str, source: str = "manuals/pulse-user-manual.pdf", page: int | None = 3) -> Hit:
    return Hit(
        text=text, source=source, title="T", section=None, page=page, row=None,
        audience="public", effective_date=None, score=0.8,
    )


class FakeLLM:
    """Returns a fixed reply and remembers the prompt it was given."""

    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.prompt = ""

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        self.prompt = prompt
        return LLMAnswer(text=self.reply, model=self.model, input_tokens=100, output_tokens=10)


@pytest.fixture()
def two_hits(monkeypatch):
    hits = [hit("E-214: battery sensor"), hit("Party Mode: 4 speakers", source="faq/general-faq.md", page=None)]
    monkeypatch.setattr(generate, "retrieve", lambda conn, q, k, audience, rewriter=None: hits)
    return hits


def test_prompt_numbers_sources_with_location():
    prompt = build_prompt("What is E-214?", [hit("A"), hit("B", source="faq/general-faq.md", page=None)])
    assert "[1] (manuals/pulse-user-manual.pdf, page 3)\nA" in prompt
    assert "[2] (faq/general-faq.md)\nB" in prompt
    assert prompt.endswith("Question: What is E-214?")


def test_prompt_shows_when_a_source_took_effect():
    dated = replace(hit("14 days"), effective_date=date(2026, 7, 1))
    prompt = build_prompt("Return window?", [dated, hit("no date")])
    assert "[1] (manuals/pulse-user-manual.pdf, page 3, effective 2026-07-01)\n14 days" in prompt
    assert "[2] (manuals/pulse-user-manual.pdf, page 3)\nno date" in prompt


def test_citation_formats():
    assert parse_citations("Sensor fault [2]. Update firmware [1][2]. Rest it [3, 1].", 3) == [2, 1, 3]


def test_citation_to_missing_source_is_dropped():
    assert parse_citations("Fact [1]. Invented [7].", 2) == [1]


def test_not_found_detection_tolerates_small_differences():
    assert is_not_found(NOT_FOUND)
    assert is_not_found("  i couldn't find this in the documents  ")
    assert not is_not_found("E-214 is a sensor fault [1]. " + NOT_FOUND)


def test_answer_links_citations_to_sources(two_hits):
    llm = FakeLLM("It is a battery sensor fault [1].")
    result = generate.answer(conn=None, question="What is E-214?", llm=llm)
    assert result.found
    assert result.cited == [1]
    assert result.cited_sources[0][1].location == "manuals/pulse-user-manual.pdf, page 3"
    assert "Party Mode" in llm.prompt  # the LLM saw every retrieved chunk


def test_not_found_answer_has_no_sources_cited(two_hits):
    result = generate.answer(conn=None, question="Who is the CEO?", llm=FakeLLM(NOT_FOUND))
    assert not result.found
    assert result.cited == []


def test_empty_search_skips_the_llm(monkeypatch):
    monkeypatch.setattr(generate, "retrieve", lambda conn, q, k, audience, rewriter=None: [])
    llm = FakeLLM("should not be used")
    result = generate.answer(conn=None, question="anything", llm=llm)
    assert not result.found and llm.prompt == ""


class ScriptedLLM(FakeLLM):
    """Gives its replies in order: first try, second try."""

    def __init__(self, *replies: str) -> None:
        super().__init__(replies[0])
        self.replies = list(replies)

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        self.reply = self.replies.pop(0)
        return super().generate(system, prompt)


@pytest.fixture()
def searches(monkeypatch):
    """Records the rewriter of every retrieve() call."""
    seen: list = []
    monkeypatch.setattr(
        generate, "retrieve", lambda conn, q, k, audience, rewriter=None: seen.append(rewriter) or [hit("x")]
    )
    return seen


def test_found_on_the_first_try_means_no_rewrite(searches):
    result = generate.answer(conn=None, question="q", llm=FakeLLM("x [1]."), rewrite=True)
    assert searches == [None] and not result.retried


def test_not_found_tries_once_more_with_rewritten_queries(searches):
    llm = ScriptedLLM(NOT_FOUND, "Found it [1].")
    result = generate.answer(conn=None, question="q", llm=llm, rewrite=True)
    assert searches == [None, llm]  # second search got the LLM as its rewriter
    assert result.found and result.retried and result.cited == [1]


def test_second_try_can_still_say_not_found(searches):
    result = generate.answer(conn=None, question="q", llm=ScriptedLLM(NOT_FOUND, NOT_FOUND), rewrite=True)
    assert not result.found and result.retried and len(searches) == 2


def test_rewrite_switched_off_keeps_one_try(searches):
    result = generate.answer(conn=None, question="q", llm=FakeLLM(NOT_FOUND), rewrite=False)
    assert searches == [None] and not result.found and not result.retried
