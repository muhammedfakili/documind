"""A10: query rewriting. A fake LLM and a fake search: no database, no API key."""

from __future__ import annotations

from app import retrieve as retrieve_module
from app.llm import LLMAnswer
from app.retrieve import retrieve
from app.rewrite import MAX_QUERIES, parse_queries, rewrite_query
from app.store import Hit


class FakeLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        self.calls += 1
        return LLMAnswer(text=self.reply, model=self.model, input_tokens=10, output_tokens=10)


def hit(text: str) -> Hit:
    return Hit(text=text, source="doc.md", title="Doc", section=None, page=None, row=None,
               audience="public", effective_date=None, score=0.0)


def test_one_query_per_line_list_markers_removed():
    text = "1. Pulse price US\n- Pulse Pro price US\n* \"Pulse noise cancellation\"\n"
    assert parse_queries(text, "q") == ["Pulse price US", "Pulse Pro price US", "Pulse noise cancellation"]


def test_numbers_inside_a_query_are_kept():
    # Only list markers are stripped: a query may start with a year or a model number.
    assert parse_queries("2025 return policy\n2) E-214 meaning", "q") == ["2025 return policy", "E-214 meaning"]


def test_blanks_repeats_and_the_question_itself_are_dropped():
    text = "\nshipping countries\nShipping countries\nCan I order to Mars?\n\n"
    assert parse_queries(text, "Can I order to Mars?") == ["shipping countries"]


def test_at_most_max_queries():
    text = "\n".join(f"query {i}" for i in range(10))
    assert len(parse_queries(text, "q")) == MAX_QUERIES


def test_rewrite_query_asks_the_llm_once():
    llm = FakeLLM("warranty replacement\nservice policy")
    assert rewrite_query("Can you fix it?", llm) == ["warranty replacement", "service policy"]
    assert llm.calls == 1


def test_every_query_is_searched_and_the_reranker_reads_the_original_question(monkeypatch):
    searched: list[str] = []
    found = {"Can you fix it?": [hit("general faq")], "warranty replacement": [hit("we do not repair units")]}

    def fake_search(conn, q, k, audience, mode):
        searched.append(q)
        return found[q]

    reranked: list[str] = []
    monkeypatch.setattr(retrieve_module, "_search", fake_search)
    monkeypatch.setattr(retrieve_module, "cross_encoder_rerank", lambda q, hits: reranked.append(q) or hits)

    hits = retrieve(None, "Can you fix it?", k=5, rewriter=FakeLLM("warranty replacement"))
    assert searched == ["Can you fix it?", "warranty replacement"]
    assert {h.text for h in hits} == {"general faq", "we do not repair units"}  # the rewrite's find got in
    assert reranked == ["Can you fix it?"]


def test_without_a_rewriter_no_llm_is_needed(monkeypatch):
    monkeypatch.setattr(retrieve_module, "_search", lambda conn, q, k, audience, mode: [hit("a")])
    monkeypatch.setattr(retrieve_module, "cross_encoder_rerank", lambda q, hits: hits)
    assert [h.text for h in retrieve(None, "q")] == ["a"]
