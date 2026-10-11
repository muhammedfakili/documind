"""A12: turn the API's JSON into what the screen shows. Pure functions, no Streamlit.

Kept apart from app.py so they can be tested without a browser (tests/test_ui.py).
Streamlit draws Markdown, which is why much of this is careful escaping: "$129 ... $169"
would otherwise render as a math formula, a "## Heading" line inside a card as a giant title.
"""

from __future__ import annotations

import re
from typing import Any

# Words too common to say anything about why a chunk was picked.
_STOP_WORDS = (
    "the and for are was were can could does did has have had how what when where which who why will "
    "would should you your yours they them their this that these those with from into about after "
    "before not but any all our ours its it's i'm i've is am be my me do of to in on at or an a "
    "there here than then also only just still lumora"
)
STOP_WORDS = frozenset(_STOP_WORDS.split())

_MARKDOWN_SPECIAL = re.compile(r"([\\`*_{}\[\]<>#+|$~])")
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1]  [1, 3] -- the same forms generate.py accepts
_WORD = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")  # keeps codes like e-214 and usb-c whole

# What each step of the search pipeline is called on screen (keys = Hit.ranks in app/store.py).
RANK_LABELS = {
    "vector": "Meaning search (vector)",
    "keyword": "Word search (BM25)",
    "hybrid": "Both searches fused",
    "reranker": "Reranker (reads question and text together)",
}


def escape_markdown(text: str) -> str:
    return _MARKDOWN_SPECIAL.sub(r"\\\1", text)


def answer_markdown(answer: str) -> str:
    """The LLM's answer with [n] drawn as small blue badges. Its own lists and bold text stay."""
    text = answer.replace("$", r"\$")  # prices, not math
    return _CITATION.sub(lambda m: " ".join(f":blue-badge[{n.strip()}]" for n in m.group(1).split(",")), text)


def question_terms(question: str) -> list[str]:
    """The question's meaningful words, in order: what to look for in a source."""
    terms: list[str] = []
    for word in _WORD.findall(question.lower()):
        if len(word) >= 3 and word not in STOP_WORDS and word not in terms:
            terms.append(word)
    return terms


def _stem(word: str) -> str:
    """A crude stem so 'returns' also marks 'return' and 'charging' marks 'charge'."""
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            return word[: -len(suffix)]
    return word


def chunk_markdown(text: str) -> str:
    """A chunk as Markdown that fits inside a small card.

    The chunks are Markdown already (tables, bold, lists from our documents), so they are
    drawn, not escaped -- escaped, a table shows up as a mess of | and ** (first live run).
    Only three things are fixed: $ is a price not math, "## Heading" becomes bold instead
    of a giant title, and every line stays its own line.
    """
    lines = text.replace("$", r"\$").splitlines()
    out: list[str] = []
    for i, line in enumerate(lines):
        row = line.lstrip().startswith("|")
        after_row = i > 0 and lines[i - 1].lstrip().startswith("|")
        if row != after_row and i > 0:
            out.append("")  # a table must stand apart from the text around it
        heading = re.match(r"\s*#{1,6}\s+(.*)", line)
        if heading:
            line = f"**{heading.group(1)}**"
        out.append(line if row else line + "  ")  # two spaces = line break in Markdown
    return "\n".join(out)


def highlight(text: str, terms: list[str]) -> str:
    """chunk_markdown(text) with the question's words marked.

    A visual hint only, not the real reason a chunk was chosen: the vector search
    matches meaning without shared words. The ranks panel shows the real reasons.
    """
    stems = sorted({_stem(t) for t in terms}, key=len, reverse=True)  # longest first: "e-214" before "214"
    if stems:
        pattern = re.compile(
            r"(?<![\w-])(" + "|".join(re.escape(s) for s in stems) + r")[\w-]*", re.IGNORECASE
        )
        text = pattern.sub(lambda m: f":orange-background[{m.group(0)}]", text)
    return chunk_markdown(text)


def excerpt(text: str, limit: int = 320) -> str:
    """The first lines of a chunk, about `limit` characters. Whole lines, so table rows stay intact."""
    if len(text) <= limit:
        return text
    kept: list[str] = []
    size = 0
    for line in text.splitlines():
        if kept and size + len(line) > limit:
            break
        kept.append(line)
        size += len(line) + 1
    cut = "\n".join(kept)
    if len(cut) > limit:  # one very long line: cut it at a word
        cut = cut[:limit].rsplit(" ", 1)[0]
    return cut + "\n…"


def source_meta(source: dict[str, Any]) -> str:
    """'manuals/pulse-user-manual.pdf, page 3 · Troubleshooting · effective 2026-07-01'"""
    parts = [source["location"]]
    if source.get("section"):
        parts.append(source["section"])
    if source.get("effective_date"):
        parts.append(f"effective {source['effective_date']}")
    return " · ".join(parts)


def rank_lines(ranks: dict[str, int]) -> list[str]:
    """One line per search step: where this source stood in it, or that it missed it."""
    if not ranks:
        return []
    lines = []
    for key, label in RANK_LABELS.items():
        if key in ranks:
            lines.append(f"{label}: #{ranks[key]}")
        elif key in ("vector", "keyword"):  # a search that ran but did not list this chunk
            lines.append(f"{label}: not in its top results")
    return lines


def status_line(response: dict[str, Any]) -> str:
    """The small grey line under an answer."""
    parts = []
    if response["found"]:
        n = len(response["cited"])
        parts.append(f"{n} source{'s' if n != 1 else ''} cited")
    parts.append(f"{response['seconds']:.1f} s")
    if response.get("retried"):
        parts.append("second try with rewritten search")
    if response.get("model"):
        parts.append(response["model"])
    return " · ".join(parts)
