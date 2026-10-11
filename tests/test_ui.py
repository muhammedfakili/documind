"""A12: the chat screen. No browser, no API server, no database.

Three layers:
  format   pure functions that turn the API's JSON into Markdown
  client   the HTTP calls, against a fake API (httpx.MockTransport)
  app      the whole Streamlit page, run by Streamlit's own test runner (AppTest)
           with ApiClient's methods replaced by fakes
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from ui import format as fmt
from ui.client import ApiClient, ApiError

APP = str(Path(__file__).resolve().parents[1] / "ui" / "app.py")


def source(
    n: int, text: str, cited: bool = True, audience: str = "public", ranks: dict | None = None
) -> dict:
    return {
        "n": n,
        "location": f"policies/doc-{n}.md",
        "title": f"Doc {n}",
        "section": "Returns",
        "audience": audience,
        "effective_date": "2026-07-01",
        "score": 0.03,
        "cited": cited,
        "ranks": ranks if ranks is not None else {"vector": 2, "keyword": 1, "hybrid": 1, "reranker": 1},
        "text": text,
    }


def response(question: str = "Can I return opened earbuds?", **overrides) -> dict:
    body = {
        "question": question,
        "answer": "No. Opened earbuds: 14 days [1].",
        "found": True,
        "retried": False,
        "cited": [1],
        "sources": [source(1, "Opened earbuds: 14 days."), source(2, "Speakers: 30 days.", cited=False)],
        "model": "fake-1",
        "input_tokens": 100,
        "output_tokens": 10,
        "seconds": 1.23,
    }
    return body | overrides


# --- format ---------------------------------------------------------------------------


def test_citations_become_badges_and_prices_stay_prices():
    md = fmt.answer_markdown("It costs $169 [1] or $199 [2, 3].")
    assert ":blue-badge[1]" in md and ":blue-badge[2] :blue-badge[3]" in md
    assert r"\$169" in md  # two bare $ would render the text between them as a formula


def test_question_terms_drop_filler_and_keep_codes():
    assert fmt.question_terms("What does E-214 mean on my Pulse Pro?") == ["e-214", "mean", "pulse", "pro"]


def test_highlight_marks_other_word_forms():
    md = fmt.highlight("## Returns\nReturn opened earbuds within 14 days.", ["returns", "opened"])
    assert md.startswith("**:orange-background[Returns]**")  # a heading in a card is bold, not a giant title
    assert ":orange-background[Return] :orange-background[opened]" in md
    assert "  \n" in md  # line breaks survive


def test_chunk_tables_stay_tables_and_prices_stay_prices():
    md = fmt.chunk_markdown(
        "Return policy > Window\n| Product | Opened |\n|---|---|\n| Earbuds | 14 days |\nPrice $129 or $169"
    )
    lines = md.split("\n")
    assert lines[1] == ""  # an empty line before the table, or Markdown does not draw it
    assert lines[2:5] == ["| Product | Opened |", "|---|---|", "| Earbuds | 14 days |"]  # untouched
    assert lines[5] == "" and r"\$129" in lines[6]


def test_highlight_keeps_codes_whole():
    md = fmt.highlight("E-214: battery calibration. E-410: pairing.", ["e-214"])
    assert ":orange-background[E-214]" in md and "orange-background[E-410]" not in md


def test_rank_lines_say_when_a_search_missed():
    lines = fmt.rank_lines({"vector": 4, "hybrid": 2, "reranker": 1})
    assert lines[0] == "Meaning search (vector): #4"
    assert lines[1] == "Word search (BM25): not in its top results"
    assert lines[-1].endswith("#1")
    assert fmt.rank_lines({}) == []


def test_excerpt_keeps_whole_lines():
    assert fmt.excerpt("short") == "short"
    rows = "\n".join(f"| row {i} | 30 days |" for i in range(30))
    cut = fmt.excerpt(rows, limit=60)
    assert cut.split("\n")[:-1] == [f"| row {i} | 30 days |" for i in range(3)]  # no half rows
    assert cut.endswith("\n…")
    assert fmt.excerpt("word " * 100, limit=22) == "word word word word\n…"  # one long line: cut at a word


def test_status_line():
    assert fmt.status_line(response()) == "1 source cited · 1.2 s · fake-1"
    retried = fmt.status_line(response(found=False, retried=True, cited=[]))
    assert "cited" not in retried and "second try" in retried


# --- client ---------------------------------------------------------------------------


def fake_api(status: int, body: dict, seen: list | None = None) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if seen is not None:
            seen.append(request)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler)


def test_client_sends_question_and_agent_key():
    seen: list[httpx.Request] = []
    client = ApiClient("http://api", api_key="agent-key", transport=fake_api(200, response(), seen))
    assert client.ask("q?", audience="all", k=3)["found"]
    assert seen[0].url.path == "/ask"
    assert seen[0].headers["X-API-Key"] == "agent-key"
    assert b'"audience":"all"' in seen[0].content and b'"k":3' in seen[0].content


def test_customer_sends_no_key_header():
    seen: list[httpx.Request] = []
    ApiClient("http://api", transport=fake_api(200, response(), seen)).ask("q?")
    assert "X-API-Key" not in seen[0].headers


def test_api_errors_keep_status_and_message():
    with pytest.raises(ApiError) as err:
        ApiClient(
            "http://api", transport=fake_api(401, {"detail": "Missing or wrong X-API-Key header."})
        ).ask("q")
    assert err.value.status == 401 and "X-API-Key" in err.value.message


def test_validation_errors_are_readable():
    body = {"detail": [{"loc": ["body", "question"], "msg": "String should have at least 1 character"}]}
    with pytest.raises(ApiError) as err:
        ApiClient("http://api", transport=fake_api(422, body)).ask("")
    assert err.value.message == "question: String should have at least 1 character"


def test_unreachable_api_has_no_status():
    def refuse(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError) as err:
        ApiClient("http://api", transport=httpx.MockTransport(refuse)).health()
    assert err.value.status is None and "Is it running?" in err.value.message


# --- the page -------------------------------------------------------------------------


@pytest.fixture()
def calls(monkeypatch) -> list[dict]:
    """ApiClient.ask/health replaced by fakes; every ask is recorded."""
    recorded: list[dict] = []

    def fake_ask(self, question, audience="public", k=5):
        recorded.append({"question": question, "audience": audience, "k": k, "key": self.api_key})
        if question == "boom":
            raise ApiError(502, "The language model is not answering right now (TimeoutError).")
        if question == "Who is the CEO?":
            return response(
                question,
                answer="I couldn't find this in the documents.",
                found=False,
                cited=[],
                sources=[source(1, "Unrelated.", cited=False)],
            )
        return response(question)

    monkeypatch.setattr(ApiClient, "ask", fake_ask)
    monkeypatch.setattr(
        ApiClient, "health", lambda self: {"status": "ok", "chunks": {"public": 120, "internal": 37}}
    )
    st.cache_data.clear()  # the sidebar's health check is cached between runs
    return recorded


def run_app() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def all_markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def test_start_screen_shows_examples_and_api_status(calls):
    at = run_app()
    assert not at.exception
    assert any("Pulse Pro" in b.label for b in at.button)
    assert "157 chunks" in at.sidebar.success[0].value
    assert calls == []  # nothing is asked before the user asks


def test_answer_with_badges_and_source_cards(calls):
    at = run_app()
    at.chat_input[0].set_value("Can I return opened earbuds?").run()
    assert not at.exception
    md = all_markdown(at)
    assert "No. Opened earbuds: 14 days :blue-badge[1]." in md
    assert ":orange-background[earbuds]" in md  # the question's words are marked in the source
    assert [t.label for t in at.tabs] == ["Sources used (1)", "Everything retrieved (2)"]
    assert any(e.label == "Why this source?" for e in at.expander)
    assert "Word search (BM25): #1" in md
    assert calls == [{"question": "Can I return opened earbuds?", "audience": "public", "k": 5, "key": None}]


def test_example_button_asks_that_question(calls):
    at = run_app()
    next(b for b in at.button if "Japan" in b.label).click().run()
    assert calls[0]["question"] == "Can I order Lumora products to Japan?"
    assert not at.button or not any("Japan" in b.label for b in at.button)  # start screen is gone


def test_not_found_is_shown_differently(calls):
    at = run_app()
    at.chat_input[0].set_value("Who is the CEO?").run()
    assert "couldn't find" in at.info[0].value
    assert [t.label for t in at.tabs] == ["Sources used (0)", "Everything retrieved (1)"]


def test_agent_mode_sends_key_and_audience_all(calls):
    at = run_app()
    at.sidebar.radio[0].set_value("Support agent").run()
    at.sidebar.text_input[0].input("agent-key").run()
    at.chat_input[0].set_value("Who approves a $320 refund?").run()
    assert calls[-1]["audience"] == "all" and calls[-1]["key"] == "agent-key"


def test_api_error_is_explained_and_chat_keeps_going(calls):
    at = run_app()
    at.chat_input[0].set_value("boom").run()
    assert "not answering right now" in at.error[0].value
    at.chat_input[0].set_value("Can I return opened earbuds?").run()
    assert not at.exception
    assert len(at.chat_message) == 4  # question, error, question, answer -- history is kept
