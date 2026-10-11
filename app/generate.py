"""A6: the last step of the query pipeline -- retrieved chunks in, cited answer out.

    question -> retrieve() -> numbered sources -> prompt -> LLM -> Answer

Three rules make the LLM a reader instead of a guesser ("grounding"):
  1. answer ONLY from the numbered sources, never from general knowledge
  2. put the source number [n] after every claim, so each sentence can be checked
  3. if the sources do not contain the answer, reply with one fixed sentence

The fixed sentence (NOT_FOUND) is something code can detect: the API and the UI
can show "not found" differently from a real answer, and the evaluation (A7)
can count how often the system correctly says it does not know.

A10 added what the measurements asked for:
  - every source shows the date it took effect, and when two versions disagree the
    newest wins (the 2025 vs 2026 return policy, an older manual vs newer release notes)
  - yes/no questions start with yes or no; conditions and exceptions are kept
  - a closed list ("only", "we do not ... other") answers "is X included?" with no.
    That is reading, not guessing -- before A10 the prompt refused these as gaps.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import psycopg

from app.config import settings
from app.llm import LLMProvider, get_llm
from app.retrieve import retrieve
from app.store import Audience, Hit

NOT_FOUND = "I couldn't find this in the documents."

SYSTEM_PROMPT = f"""You are the customer support assistant for Lumora Audio.

You answer questions using ONLY the numbered sources you are given. Rules:
- Use only facts that are written in the sources. Do not use general knowledge, do not guess,
  do not fill gaps with what is "usually" true.
- After every sentence that states a fact, cite the source it came from, like [1] or [1][3].
  Only cite numbers that appear in the sources.
- Sources can be about similar products (for example Pulse and Pulse Pro). Only use a source
  if it is about exactly what the question asks.
- If the sources do not contain the answer, reply with exactly this sentence and nothing else:
  {NOT_FOUND}
- If the sources answer only part of the question, answer that part and say which part
  you could not find.
- Each source shows the date it took effect. If sources disagree (an old and a new version
  of a policy, a manual and later release notes), the newest one is the current one: answer
  with it. You may add what the older version said.
- If a source gives a complete list or says "only" (for example "available only in A and B"),
  then anything not on that list is not included: answer "No" and cite that source.
  This is reading the source, not guessing.
- Keep the conditions and exceptions a source attaches to the answer (only for some models,
  regions, versions, serial ranges or dates). Leaving them out makes the answer wrong.
- If the question can be answered with yes or no, start with "Yes" or "No", then explain.
- Be short and direct: a few sentences or a short list."""

_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")  # [1]  [1][3]  [1, 3]


@dataclass(frozen=True)
class Answer:
    """What the system says, plus everything needed to check it."""

    question: str
    text: str
    sources: list[Hit]  # what retrieval found, in prompt order: sources[0] is [1]
    cited: list[int] = field(default_factory=list)  # source numbers used in the text, valid ones only
    found: bool = True  # False = the system said NOT_FOUND
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    retried: bool = False  # A10: first try found nothing, this answer comes from rewritten queries

    @property
    def cited_sources(self) -> list[tuple[int, Hit]]:
        """The [n] -> source pairs the UI shows under the answer."""
        return [(n, self.sources[n - 1]) for n in self.cited]


def format_sources(hits: list[Hit]) -> str:
    """Number the chunks so the LLM can point at them: [1], [2], ..."""
    blocks = []
    for n, hit in enumerate(hits, start=1):
        dated = f", effective {hit.effective_date:%Y-%m-%d}" if hit.effective_date else ""
        blocks.append(f"[{n}] ({hit.location}{dated})\n{hit.text}")
    return "\n\n".join(blocks)


def build_prompt(question: str, hits: list[Hit]) -> str:
    return f"Sources:\n\n{format_sources(hits)}\n\nQuestion: {question}"


def parse_citations(text: str, n_sources: int) -> list[int]:
    """Source numbers the answer cites, in order of first use.

    Numbers outside 1..n_sources are dropped: a citation to a source that does
    not exist is itself a small hallucination and must never reach the user.
    """
    seen: list[int] = []
    for group in _CITATION.findall(text):
        for part in group.split(","):
            n = int(part)
            if 1 <= n <= n_sources and n not in seen:
                seen.append(n)
    return seen


def is_not_found(text: str) -> bool:
    return text.strip().rstrip(".").lower() == NOT_FOUND.rstrip(".").lower()


def answer(
    conn: psycopg.Connection,
    question: str,
    k: int = 5,
    audience: Audience = "public",
    llm: LLMProvider | None = None,
    rewrite: bool | None = None,
) -> Answer:
    """The whole query pipeline: search, then let the LLM write a cited answer.

    rewrite: the second try (A10). If the first answer is NOT_FOUND, the LLM rewrites the
          question into the documents' words and we search and answer once more.
          None = settings.query_rewrite. Only misses pay for it: rewriting EVERY question
          also pushed already-found evidence out of the top 5 (scripts/a10_hard.py).
    """
    llm = llm or get_llm()
    first = answer_from_hits(question, retrieve(conn, question, k=k, audience=audience), llm)
    if first.found or not (settings.query_rewrite if rewrite is None else rewrite):
        return first
    hits = retrieve(conn, question, k=k, audience=audience, rewriter=llm)
    return replace(answer_from_hits(question, hits, llm), retried=True)


def answer_from_hits(question: str, hits: list[Hit], llm: LLMProvider | None = None) -> Answer:
    """The writing half on its own, for callers that already searched (evaluation, A9 reranking)."""
    if not hits:  # empty database: nothing to read, no reason to call the LLM
        return Answer(question=question, text=NOT_FOUND, sources=[], found=False)

    llm = llm or get_llm()
    reply = llm.generate(SYSTEM_PROMPT, build_prompt(question, hits))
    text = reply.text.strip()
    found = not is_not_found(text)
    return Answer(
        question=question,
        text=text,
        sources=hits,
        cited=parse_citations(text, len(hits)) if found else [],
        found=found,
        model=reply.model,
        input_tokens=reply.input_tokens,
        output_tokens=reply.output_tokens,
    )
