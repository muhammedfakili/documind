"""Chunking tests: small hand-made Documents for the rules, the real dataset for the outcome."""

from __future__ import annotations

import pytest

from app.config import DATA_DIR
from app.ingest.chunking import (
    chunk_documents,
    fixed_size_chunks,
    heading_chunks,
    split_sections,
    with_context,
)
from app.ingest.documents import Document
from app.ingest.pipeline import load_directory


def make_doc(text: str, title: str = "Test Doc") -> Document:
    return Document(text=text, source="test.md", title=title, doc_type="other", audience="public")


@pytest.fixture(scope="module")
def chunks():
    return chunk_documents(load_directory(DATA_DIR / "raw"))


# --- fixed size ------------------------------------------------------------


def test_fixed_size_overlap_repeats_words():
    doc = make_doc(" ".join(f"w{i}" for i in range(25)))
    pieces = fixed_size_chunks(doc, size=10, overlap=3)
    # step = size - overlap = 7, so chunks start at w0, w7, w14, w21
    assert [p.text.split()[0] for p in pieces] == ["w0", "w7", "w14", "w21"]
    assert pieces[0].text.split()[-3:] == pieces[1].text.split()[:3]
    assert pieces[-1].text.split()[-1] == "w24"  # nothing lost at the end


def test_fixed_size_rejects_overlap_not_smaller_than_size():
    with pytest.raises(ValueError):
        fixed_size_chunks(make_doc("a b c"), size=5, overlap=5)


# --- heading ---------------------------------------------------------------


def test_split_sections_drops_title_and_keeps_headings():
    text = "# Title\n\nIntro line.\n\n## First\nOne.\n\n## Second\nTwo."
    assert split_sections(text) == [(None, "Intro line."), ("First", "One."), ("Second", "Two.")]


def test_question_stays_with_its_answer():
    faq = "## FAQ\n" + "\n\n".join(f"**Question {i}?**\nAnswer {i} " + "word " * 20 for i in range(5))
    for chunk in heading_chunks(make_doc(faq), max_words=60):
        for i in range(5):
            assert (f"Question {i}?" in chunk.text) == (f"Answer {i} " in chunk.text)


def test_split_table_repeats_its_header():
    rows = "\n".join(f"| Product {i} | {i} days | {i + 1} days |" for i in range(12))
    doc = make_doc(f"## Window\n| Product | Unopened | Opened |\n|---|---|---|\n{rows}")
    pieces = heading_chunks(doc, max_words=30)
    assert len(pieces) > 1
    for piece in pieces:
        assert "| Product | Unopened | Opened |" in piece.text
    assert all(f"| Product {i} |" in "".join(p.text for p in pieces) for i in range(12))


def test_chunk_keeps_page_for_citation(chunks):
    e214 = next(c for c in chunks if c.doc.source == "manuals/pulse-user-manual.pdf" and "E-214" in c.text)
    assert e214.doc.page == 3
    assert e214.section == "6. Error codes"
    assert e214.location == "manuals/pulse-user-manual.pdf, page 3"


def test_csv_row_is_one_chunk(chunks):
    tickets = [c for c in chunks if c.doc.source == "support/tickets-2026-q3.csv"]
    assert len(tickets) == 30


# --- context header ----------------------------------------------------------


def test_with_context_names_document_and_section():
    assert with_context("Pulse Manual", "6. Error codes", "Code: E-214") == (
        "Pulse Manual > 6. Error codes\nCode: E-214"
    )
    assert with_context("Pulse Manual", None, "Model: LA-PL-100") == "Pulse Manual\nModel: LA-PL-100"


def test_same_error_code_is_told_apart(chunks):
    """E-214 means different things on Pulse and Pulse Pro: each chunk must say which."""
    e214 = [c for c in chunks if "Code: E-214" in c.text]
    assert any("Lumora Pulse - User Manual" in c.text and "Battery temperature" in c.text for c in e214)
    assert any("Lumora Pulse Pro - User Manual" in c.text and "ANC microphone" in c.text for c in e214)
