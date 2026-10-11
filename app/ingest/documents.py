"""The one shape every loader produces.

A client's knowledge base arrives as PDFs, Markdown, CSV exports and Excel
sheets. Everything downstream (chunking, embedding, search, citations) should
not care which one a piece of text came from, so each loader turns its format
into `Document`s: plain text plus the metadata we will need later.

Why the metadata matters, stage by stage:
  source / page / row  -> citations: "Pulse manual, page 3"
  effective_date       -> picking the newer of two conflicting policies (A10)
  audience             -> hiding internal documents from customers
  doc_type             -> filtering ("only search manuals")
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Literal

Audience = Literal["public", "internal"]

# data/raw/<folder>/... -> doc_type. The folder is how the client organized
# their files, and it is usually the most reliable type signal we get.
DOC_TYPES = {
    "manuals": "manual",
    "policies": "policy",
    "faq": "faq",
    "internal": "internal",
    "release-notes": "release_notes",
    "support": "support",
    "catalog": "catalog",
}

# Folders whose content must never be shown to a customer, whatever it says.
INTERNAL_FOLDERS = {"internal", "support"}
INTERNAL_MARKERS = ("(INTERNAL)", "Confidential", "Do not share")


@dataclass(frozen=True)
class Document:
    text: str
    source: str  # path relative to data/raw, forward slashes
    title: str
    doc_type: str
    audience: Audience
    effective_date: date | None = None
    page: int | None = None  # PDFs: 1-based page number
    row: int | None = None  # CSV / Excel: 1-based data row (header excluded)

    @property
    def location(self) -> str:
        """Human-readable pointer for citations."""
        if self.page is not None:
            return f"{self.source}, page {self.page}"
        if self.row is not None:
            return f"{self.source}, row {self.row}"
        return self.source

    def to_dict(self) -> dict:
        data = asdict(self)
        data["effective_date"] = self.effective_date.isoformat() if self.effective_date else None
        return data


def classify(rel_path: Path, text: str) -> tuple[str, Audience]:
    """doc_type and audience from where the file lives and what it says."""
    folder = rel_path.parts[0] if len(rel_path.parts) > 1 else ""
    doc_type = DOC_TYPES.get(folder, "other")
    internal = folder in INTERNAL_FOLDERS or any(m in text for m in INTERNAL_MARKERS)
    return doc_type, "internal" if internal else "public"


# ---------------------------------------------------------------------------
# Dates
# ---------------------------------------------------------------------------

_MONTHS = {
    m: i
    for i, m in enumerate(
        ["january", "february", "march", "april", "may", "june", "july",
         "august", "september", "october", "november", "december"],
        start=1,
    )
}
_MONTH = r"(January|February|March|April|May|June|July|August|September|October|November|December)"
_FULL_DATE = re.compile(_MONTH + r"\s+(\d{1,2}),\s+(\d{4})")
_MONTH_YEAR = re.compile(_MONTH + r"\s+(\d{4})")
_ISO_DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")

# A date right after one of these labels is the document's own date. Anything
# else ("fixed on June 4", "discontinued in March 2026") is about something else.
_LABELS = re.compile(
    r"(effective date|last updated|last reviewed|on or after|document version [\d.]+,)\s*:?\s*",
    re.IGNORECASE,
)


def _to_date(match: re.Match) -> date:
    month = _MONTHS[match.group(1).lower()]
    if match.re is _FULL_DATE:
        return date(int(match.group(3)), month, int(match.group(2)))
    return date(int(match.group(2)), month, 1)


def extract_effective_date(text: str) -> date | None:
    """The date a document itself carries, or None.

    Order of trust: a date right after a label like "Effective date:", then
    (for changelogs with no label) the newest full date in the text.
    """
    for label in _LABELS.finditer(text):
        after = text[label.end() : label.end() + 40]
        for pattern in (_FULL_DATE, _MONTH_YEAR):
            m = pattern.match(after)
            if m:
                return _to_date(m)
    full_dates = [_to_date(m) for m in _FULL_DATE.finditer(text)]
    return max(full_dates) if full_dates else None


def parse_iso_date(value: str) -> date | None:
    m = _ISO_DATE.search(value or "")
    return date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
