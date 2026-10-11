"""Database tests. They use their own table and are skipped when Postgres is not running."""

from __future__ import annotations

import numpy as np
import psycopg
import pytest

from app import store
from app.ingest.chunking import Chunk
from app.ingest.documents import Document

TABLE = "chunks_test"


_db_down = False  # remembered: a skipped module fixture is retried for every test, and each
# refused connection on Windows takes ~20 s -- 12 tests waited 4 minutes before this flag.


@pytest.fixture(scope="module")
def conn():
    global _db_down
    if _db_down:
        pytest.skip("database not running (docker compose up -d)")
    try:
        connection = store.connect()
    except psycopg.OperationalError:
        _db_down = True
        pytest.skip("database not running (docker compose up -d)")
    store.create_table(connection, table=TABLE)
    yield connection
    connection.execute(f"DROP TABLE IF EXISTS {TABLE}")
    connection.close()


def unit(i: int) -> np.ndarray:
    """A vector pointing along axis i: easy to reason about distances."""
    v = np.zeros(store.DIM, dtype=np.float32)
    v[i] = 1.0
    return v


def make_chunk(text: str, audience: str = "public", page: int | None = None) -> Chunk:
    doc = Document(text=text, source="test.pdf", title="Test", doc_type="manual", audience=audience, page=page)
    return Chunk(text=text, doc=doc, section="S", index=0)


@pytest.fixture()
def filled(conn):
    chunks = [
        make_chunk("public A", page=3),
        make_chunk("public B"),
        make_chunk("internal C", audience="internal"),
    ]
    store.replace_chunks(conn, chunks, np.stack([unit(0), unit(1), unit(2)]), table=TABLE)
    return conn


def test_nearest_vector_comes_first_with_its_labels(filled):
    hits = store.search(filled, unit(0), k=2, audience="all", table=TABLE)
    assert hits[0].text == "public A"
    assert hits[0].location == "test.pdf, page 3"
    assert hits[0].score == pytest.approx(1.0)  # same direction
    assert hits[1].score == pytest.approx(0.0)  # perpendicular


def test_customer_never_sees_internal(filled):
    hits = store.search(filled, unit(2), k=3, audience="public", table=TABLE)
    assert {h.text for h in hits} == {"public A", "public B"}  # asked for 3, only 2 are allowed


def test_public_is_the_default(filled):
    hits = store.search(filled, unit(2), k=3, table=TABLE)
    assert "internal C" not in [h.text for h in hits]


def test_agent_sees_everything(filled):
    hits = store.search(filled, unit(2), k=1, audience="all", table=TABLE)
    assert hits[0].text == "internal C"


def test_replace_removes_old_rows(filled):
    store.replace_chunks(filled, [make_chunk("only one")], np.stack([unit(5)]), table=TABLE)
    assert store.count(filled, table=TABLE) == {"public": 1}


def test_rejects_vectors_of_wrong_size(conn):
    with pytest.raises(ValueError):
        store.replace_chunks(conn, [make_chunk("x")], np.zeros((1, 10), dtype=np.float32), table=TABLE)


# --- A8: keyword search (BM25) ---------------------------------------------------------


@pytest.fixture()
def words(conn):
    chunks = [
        make_chunk("Pulse error codes. E-214: battery calibration needed."),
        make_chunk("Pulse Pro error codes. E-410: pairing memory full."),
        make_chunk("Pulse battery lasts 5 hours. Pulse charges by USB-C."),
        make_chunk("Internal: Pulse E-214 known issue on firmware 3.0", audience="internal"),
    ]
    store.replace_chunks(conn, chunks, np.stack([unit(i) for i in range(4)]), table=TABLE)
    return conn


def test_keyword_finds_the_exact_code(words):
    hits = store.keyword_search(words, "What does E-214 mean?", k=1, table=TABLE)
    assert "E-214" in hits[0].text


def test_rare_word_beats_common_word(words):
    # "pulse" is in every chunk, "-214" in one: BM25 gives the rare word the weight.
    hits = store.keyword_search(words, "pulse pulse E-214", k=1, table=TABLE)
    assert "E-214: battery" in hits[0].text


def test_stemming_matches_other_word_forms(words):
    hits = store.keyword_search(words, "charging", k=1, table=TABLE)  # charges ~ charging
    assert "charges" in hits[0].text


def test_one_shared_word_is_enough(words):
    # Words are OR-ed: no chunk contains "pairing" AND "refund", one still matches.
    hits = store.keyword_search(words, "pairing refund", k=5, table=TABLE)
    assert [h.text for h in hits] == ["Pulse Pro error codes. E-410: pairing memory full."]


def test_keyword_search_respects_audience(words):
    public = store.keyword_search(words, "firmware known issue", k=5, table=TABLE)
    assert public == []
    agent = store.keyword_search(words, "firmware known issue", k=5, audience="all", table=TABLE)
    assert agent[0].text.startswith("Internal")


def test_question_of_only_stop_words_returns_nothing(words):
    assert store.keyword_search(words, "what does it do", k=5, table=TABLE) == []
