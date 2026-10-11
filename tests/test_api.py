"""A11: the HTTP layer. Fake database connection, fake search, fake LLM: no Docker, no API key.

The real generate.answer() runs inside these tests (only its search and LLM are fakes),
so they check the whole path from JSON request to JSON response.
"""

from __future__ import annotations

from datetime import date

import psycopg
import pytest
from fastapi.testclient import TestClient

from app import api, generate, store
from app.config import settings
from app.generate import NOT_FOUND
from app.llm import LLMAnswer
from app.store import Hit

ADMIN = "test-admin-key"


class FakeLLM:
    name = "fake"
    model = "fake-1"

    def __init__(self, reply: str = "Opened earbuds: 14 days [1].") -> None:
        self.reply = reply
        self.calls = 0

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        self.calls += 1
        if isinstance(self.reply, Exception):
            raise self.reply
        return LLMAnswer(text=self.reply, model=self.model, input_tokens=120, output_tokens=12)


def hit(text: str, source: str, audience: str = "public") -> Hit:
    return Hit(text=text, source=source, title="Return policy", section="Return window", page=None, row=None,
               audience=audience, effective_date=date(2026, 7, 1), score=0.81234)


@pytest.fixture()
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture()
def searched(monkeypatch) -> list[str]:
    """Fake search: records the audience of every call, returns two public chunks."""
    audiences: list[str] = []

    def fake_retrieve(conn, q, k, audience, rewriter=None):
        audiences.append(audience)
        return [hit("Opened earbuds: 14 days.", "policies/return-policy-2026.md"),
                hit("Speakers: 30 days.", "faq/general-faq.md")]

    monkeypatch.setattr(generate, "retrieve", fake_retrieve)
    return audiences


@pytest.fixture()
def client(llm, monkeypatch) -> TestClient:
    monkeypatch.setattr(settings, "admin_api_key", ADMIN)
    monkeypatch.setattr(settings, "query_rewrite", False)
    api.app.dependency_overrides[api.get_conn] = lambda: None  # no database
    api.app.dependency_overrides[api.get_answer_llm] = lambda: llm
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_ask_returns_answer_with_numbered_sources(client, searched):
    r = client.post("/ask", json={"question": "How long for opened earbuds?"})
    assert r.status_code == 200
    body = r.json()
    assert body["found"] and body["cited"] == [1]
    assert [s["n"] for s in body["sources"]] == [1, 2]
    assert body["sources"][0]["cited"] and not body["sources"][1]["cited"]  # UI can grey out unused ones
    assert body["sources"][0]["location"] == "policies/return-policy-2026.md"
    assert body["sources"][0]["effective_date"] == "2026-07-01"
    assert searched == ["public"]  # customer mode unless asked otherwise


def test_not_found_answer(client, searched, llm):
    llm.reply = NOT_FOUND
    body = client.post("/ask", json={"question": "Who is the CEO?"}).json()
    assert not body["found"] and body["cited"] == [] and body["answer"] == NOT_FOUND


@pytest.mark.parametrize("payload", [{}, {"question": ""}, {"question": "x" * 1001}, {"question": "q", "k": 0},
                                     {"question": "q", "audience": "internal"}])
def test_bad_requests_never_reach_the_llm(client, searched, llm, payload):
    assert client.post("/ask", json=payload).status_code == 422
    assert llm.calls == 0 and searched == []


def test_agent_mode_needs_the_admin_key(client, searched):
    q = {"question": "Refund limit?", "audience": "all"}
    assert client.post("/ask", json=q).status_code == 401
    assert client.post("/ask", json=q, headers={"X-API-Key": "wrong"}).status_code == 401
    assert searched == []  # internal documents were never searched
    assert client.post("/ask", json=q, headers={"X-API-Key": ADMIN}).status_code == 200
    assert searched == ["all"]


def test_agent_mode_is_off_without_a_configured_key(client, searched, monkeypatch):
    monkeypatch.setattr(settings, "admin_api_key", None)
    r = client.post("/ask", json={"question": "q", "audience": "all"}, headers={"X-API-Key": ""})
    assert r.status_code == 403 and searched == []


def test_llm_failure_is_a_502_not_a_crash(client, searched, llm):
    llm.reply = TimeoutError("provider timed out")
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 502 and "TimeoutError" in r.json()["detail"]


def test_database_down_is_a_503(llm, monkeypatch):
    def refuse():
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(store, "connect", refuse)
    api.app.dependency_overrides[api.get_answer_llm] = lambda: llm
    try:
        r = TestClient(api.app).post("/ask", json={"question": "q"})
    finally:
        api.app.dependency_overrides.clear()
    assert r.status_code == 503 and llm.calls == 0


def test_ingest_needs_the_admin_key(client, monkeypatch):
    runs: list[str] = []
    monkeypatch.setattr(api, "ingest_directory", lambda conn, root: runs.append(root.name) or 157)
    monkeypatch.setattr(store, "count", lambda conn: {"public": 120, "internal": 37})
    assert client.post("/ingest").status_code == 401 and runs == []
    r = client.post("/ingest", headers={"X-API-Key": ADMIN})
    assert r.status_code == 200 and runs == ["raw"]
    assert r.json()["chunks"] == 157 and r.json()["by_audience"]["internal"] == 37


def test_health(client, monkeypatch):
    monkeypatch.setattr(store, "count", lambda conn: {"public": 120, "internal": 37})
    assert client.get("/health").json() == {"status": "ok", "chunks": {"public": 120, "internal": 37}}

    def no_table(conn):
        raise psycopg.errors.UndefinedTable("relation chunks does not exist")

    monkeypatch.setattr(store, "count", no_table)
    assert client.get("/health").status_code == 503


def test_sources_carry_audience_and_ranks_for_the_ui(client, monkeypatch):
    ranked = Hit(text="Opened earbuds: 14 days.", source="policies/return-policy-2026.md", title="Return policy",
                 section=None, page=None, row=None, audience="public", effective_date=None, score=0.03,
                 ranks={"vector": 4, "keyword": 1, "hybrid": 1, "reranker": 1})
    monkeypatch.setattr(generate, "retrieve", lambda conn, q, k, audience, rewriter=None: [ranked])
    source = client.post("/ask", json={"question": "opened earbuds?"}).json()["sources"][0]
    assert source["audience"] == "public"
    assert source["ranks"] == {"vector": 4, "keyword": 1, "hybrid": 1, "reranker": 1}


# --- A13: a fresh database fills itself on the first start -----------------------------


class _FakeConn:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


@pytest.fixture()
def ingests(monkeypatch) -> list[bool]:
    calls: list[bool] = []
    monkeypatch.setattr(store, "connect", lambda: _FakeConn())
    monkeypatch.setattr(api, "ingest_directory", lambda conn, root: calls.append(True) or 157)
    return calls


def test_empty_database_is_filled_on_start(ingests, monkeypatch):
    monkeypatch.setattr(store, "count", lambda conn: {})
    api.ingest_if_empty()
    assert ingests == [True]


def test_missing_table_is_created_on_start(ingests, monkeypatch):
    def no_table(conn):
        raise psycopg.errors.UndefinedTable("relation chunks does not exist")

    monkeypatch.setattr(store, "count", no_table)
    api.ingest_if_empty()
    assert ingests == [True]


def test_filled_database_is_left_alone(ingests, monkeypatch):
    monkeypatch.setattr(store, "count", lambda conn: {"public": 95, "internal": 62})
    api.ingest_if_empty()
    assert ingests == []


def test_start_survives_a_database_that_is_down(monkeypatch):
    def down():
        raise psycopg.OperationalError("connection refused")

    monkeypatch.setattr(store, "connect", down)
    api.ingest_if_empty()  # no exception: the server starts, /health reports the problem
