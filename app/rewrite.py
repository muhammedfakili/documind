"""A10: query rewriting -- the customer's words in, search queries in the documents' words out.

Search (A8) and reranking (A9) can only find what the question's words point at.
Two kinds of questions miss:
  - the customer asks in other words than the document uses:
      "Can Lumora repair my broken headphones?"  vs  "Lumora does not repair units"
      "Can I order to Japan?"                    vs  "We ship to the United States, Canada, ..."
  - one question asks two things, and one search finds only the louder half:
      "Which is cheaper, Pulse or Pulse Pro, and which has noise cancellation?"

One short LLM call writes a few search queries for the question. Each query is
searched on its own, and the lists are fused with RRF, together with the original
question (retrieve.py). The rewrites only steer the SEARCH: the answer is still
written for the customer's own question.

When: only as a second try, after the first answer was "not found" (generate.answer).
Rewriting every question fixed the two misses but pushed already-found evidence out of
the top 5 for four others (scripts/a10_hard.py) -- extra queries also bring extra noise.

The prompt has no example taken from the exam (eval/questions.jsonl). An example
like "repair -> warranty replacement" would fix s16 by telling the model the answer,
and the score would measure our prompt, not the method.
"""

from __future__ import annotations

import re

from app.llm import LLMProvider

MAX_QUERIES = 3
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")  # "- ", "* ", "1. ", "2) "

REWRITE_SYSTEM = f"""You turn a customer question into search queries for the knowledge base of
Lumora Audio (headphones, earbuds and speakers: manuals, policies, FAQs, catalog, release notes).

Write 1 to {MAX_QUERIES} short search queries, one per line, nothing else:
- If the question asks several things, write one query per thing.
- Use the words a company policy, manual or catalog would use, not the customer's everyday words.
  Ask yourself which document would answer this, and what its heading would say.
- Keep product names and codes exactly as written (Pulse is not Pulse Pro, E-214 stays E-214).
- No numbering, no quotes, no explanations."""


def rewrite_query(question: str, llm: LLMProvider) -> list[str]:
    """Up to MAX_QUERIES search queries for `question` (the question itself is not included)."""
    reply = llm.generate(REWRITE_SYSTEM, question)
    return parse_queries(reply.text, question)


def parse_queries(text: str, question: str) -> list[str]:
    """One query per line; drop list markers, blanks, repeats and copies of the question."""
    queries: list[str] = []
    seen = {question.strip().lower()}
    for line in text.splitlines():
        q = _LIST_MARKER.sub("", line).strip().strip('"').strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            queries.append(q)
    return queries[:MAX_QUERIES]
