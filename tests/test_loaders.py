"""Loader tests against the real (synthetic) dataset in data/raw.

The dataset is deterministic and versioned with the code, so asserting on its
actual content is the most direct check that each format is read correctly.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.config import DATA_DIR
from app.ingest.documents import extract_effective_date
from app.ingest.loaders import table_to_lines
from app.ingest.pipeline import load_directory


@pytest.fixture(scope="module")
def docs():
    return load_directory(DATA_DIR / "raw")


def find(docs, source, **where):
    return [d for d in docs if d.source == source and all(getattr(d, k) == v for k, v in where.items())]


def test_every_file_produces_documents(docs):
    raw_files = {p.relative_to(DATA_DIR / "raw").as_posix() for p in (DATA_DIR / "raw").rglob("*") if p.is_file()}
    assert {d.source for d in docs} == raw_files


def test_pdf_table_row_keeps_code_and_meaning_together(docs):
    page = find(docs, "manuals/pulse-user-manual.pdf", page=3)[0]
    line = next(ln for ln in page.text.splitlines() if "E-214" in ln)
    assert "Battery temperature sensor fault in the RIGHT earbud" in line
    assert "firmware 3.2.1" in line


def test_pdf_wrapped_lines_are_joined_into_paragraphs(docs):
    page = find(docs, "manuals/pulse-user-manual.pdf", page=2)[0]
    assert "wireless (Qi) charging is not supported on Pulse." in page.text


def test_pdf_bold_lines_become_headings(docs):
    page = find(docs, "manuals/pulse-user-manual.pdf", page=3)[0]
    assert "## 6. Error codes" in page.text.splitlines()


def test_xlsx_row_keeps_column_names(docs):
    row = next(d for d in find(docs, "catalog/warranty-matrix.xlsx") if "Halo Lite" in d.text)
    assert "Warranty US region (months): 12" in row.text
    assert "Warranty EU region (months): 24" in row.text


def test_conflicting_policies_carry_their_own_dates(docs):
    old = find(docs, "policies/return-policy-2025.md")[0]
    new = find(docs, "policies/return-policy-2026.md")[0]
    assert old.effective_date == date(2025, 1, 15)
    assert new.effective_date == date(2026, 7, 1)


def test_internal_documents_are_marked(docs):
    assert find(docs, "internal/support-handbook.md")[0].audience == "internal"
    assert all(d.audience == "internal" for d in find(docs, "support/tickets-2026-q3.csv"))
    assert find(docs, "policies/shipping-policy.md")[0].audience == "public"


def test_ticket_row_date_and_citation(docs):
    ticket = find(docs, "support/tickets-2026-q3.csv", row=1)[0]
    assert ticket.effective_date == date(2026, 7, 2)
    assert ticket.location == "support/tickets-2026-q3.csv, row 1"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("**Effective date:** July 1, 2026", date(2026, 7, 1)),
        ("Version 4.2, last reviewed August 18, 2026", date(2026, 8, 18)),
        ("Model: LA-PL-100. Document version 3.1, February 2026.", date(2026, 2, 1)),
        ("## 3.2.1 — August 27, 2026\n## 3.2.0 — June 4, 2026", date(2026, 8, 27)),
        ("Halo Lite was discontinued in March 2026.", None),
    ],
)
def test_extract_effective_date(text, expected):
    assert extract_effective_date(text) == expected


def test_two_column_tables_render_as_key_value():
    assert table_to_lines([["Item", "Value"], ["Driver", "10 mm"]]) == ["Driver: 10 mm"]
