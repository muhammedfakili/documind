"""A7: the exam -- questions with known answers, and how the system is graded on them.

Two separate grades, because a RAG answer can fail in two separate places:

  retrieval   Did search put the evidence in front of the LLM?
              Checked by code: does a retrieved chunk contain the evidence text?
              hit@5, evidence recall@5, MRR@10

  answer      Is what the LLM wrote right, and is all of it backed by the sources?
              Checked by a second LLM call, the "judge" (LLM-as-judge), which sees the
              question, the reference answer, the sources and the system's answer.

Every later stage (A8 hybrid, A9 rerank, A10 hard cases) runs the same exam, so each
improvement shows up as a change in these numbers.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from app.config import PROJECT_ROOT
from app.generate import Answer, format_sources
from app.llm import LLMAnswer, LLMProvider
from app.store import Audience, Hit

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
CACHE_DIR = PROJECT_ROOT / "eval" / "cache"

# Categories where the right behaviour is to say "not found":
#   hidden        the answer exists, but only in internal documents, and a customer asks
#   unanswerable  the answer is in no document at all
REFUSE_CATEGORIES = {"hidden", "unanswerable"}
SHOULD_REFUSE = (
    "The documents available to the assistant do not contain this information. The correct response "
    "is to say it could not be found, without guessing or adding facts from elsewhere."
)


@dataclass(frozen=True)
class Question:
    id: str
    category: str  # single | table | multi | conflict | internal | hidden | unanswerable
    audience: Audience
    question: str
    reference: str  # the right answer, written by hand from the documents
    evidence: list[list[str]]  # pieces of evidence; each piece = alternative texts, any one counts

    @property
    def should_answer(self) -> bool:
        return self.category not in REFUSE_CATEGORIES


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [Question(**json.loads(line)) for line in lines if line.strip()]


# --- retrieval grading (pure code, no LLM) -----------------------------------


def matches(text: str, piece: list[str]) -> bool:
    """Does this chunk contain one piece of evidence (any of its alternative texts)?"""
    low = text.lower()
    return any(alt.lower() in low for alt in piece)


def first_relevant_rank(hits: list[Hit], evidence: list[list[str]]) -> int | None:
    """Rank (1-based) of the first chunk that contains any evidence; None if none does."""
    for rank, hit in enumerate(hits, start=1):
        if any(matches(hit.text, piece) for piece in evidence):
            return rank
    return None


def evidence_recall(hits: list[Hit], evidence: list[list[str]]) -> float:
    """Share of the evidence pieces found somewhere in `hits` (multi-document questions need all)."""
    if not evidence:
        return 1.0
    found = sum(any(matches(h.text, piece) for h in hits) for piece in evidence)
    return found / len(evidence)


# --- answer grading (LLM-as-judge) ---------------------------------------------

JUDGE_SYSTEM = """You grade answers of a customer-support assistant that must answer only from its sources.

You get: the question, a reference answer written by an expert, the sources the assistant saw,
and the assistant's answer. Return ONLY a JSON object:

{"correctness": "correct" | "partial" | "wrong", "faithful": true | false, "reason": "<one short sentence>"}

correctness -- compare with the reference answer. Key facts are the ones needed to answer what
the question actually asks; side details in the reference are not key.
  correct  every key fact of the reference is there and nothing contradicts it. Extra details
           are fine if they are true according to the sources. Wording does not matter.
  partial  some key facts are there, but an important one is missing.
  wrong    the main fact is missing or contradicts the reference, or the assistant says it
           could not find an answer that the reference gives.

faithful -- ignore the reference here, look only at the sources:
  true   every factual claim in the answer is supported by the sources.
  false  at least one claim is not in the sources (invented or from general knowledge).
"""

Verdict = dict  # {"correctness": str, "faithful": bool, "reason": str}

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


def judge_prompt(question: Question, answer: Answer) -> str:
    reference = question.reference if question.should_answer else SHOULD_REFUSE
    return (
        f"Question: {question.question}\n\n"
        f"Reference answer: {reference}\n\n"
        f"Sources the assistant saw:\n\n{format_sources(answer.sources)}\n\n"
        f"Assistant's answer: {answer.text}"
    )


def parse_verdict(text: str) -> Verdict:
    """Read the judge's JSON, tolerating code fences or text around it."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError(f"judge returned no JSON: {text[:200]}")
    data = json.loads(match.group(0))
    if data.get("correctness") not in ("correct", "partial", "wrong"):
        raise ValueError(f"bad correctness value: {data!r}")
    return {
        "correctness": data["correctness"],
        "faithful": bool(data.get("faithful")),
        "reason": str(data.get("reason", "")),
    }


def judge(llm: LLMProvider, question: Question, answer: Answer) -> Verdict:
    return parse_verdict(llm.generate(JUDGE_SYSTEM, judge_prompt(question, answer)).text)


# --- one graded question ---------------------------------------------------------

POINTS = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


@dataclass
class Result:
    question: Question
    hits: list[Hit]  # top 10 from search; the LLM saw the first 5
    answer: Answer
    verdict: Verdict | None = None  # None when the system said "not found"
    seconds: float = 0.0

    @property
    def rank(self) -> int | None:
        return first_relevant_rank(self.hits, self.question.evidence)

    @property
    def recall5(self) -> float:
        return evidence_recall(self.hits[:5], self.question.evidence)

    @property
    def score(self) -> float:
        """1 = right, 0.5 = partly right, 0 = wrong. Same scale for every category."""
        if not self.answer.found:
            return 0.0 if self.question.should_answer else 1.0
        return POINTS[self.verdict["correctness"]] if self.verdict else 0.0

    @property
    def faithful(self) -> bool | None:
        return None if self.verdict is None else self.verdict["faithful"]

    @property
    def failure(self) -> str | None:
        """Where it went wrong, so we know which stage has to fix it."""
        if self.score == 1.0:
            return None
        if not self.question.should_answer:
            return "uydurma"  # answered something that is not in the documents
        if self.recall5 < 1.0:
            return "arama"  # some evidence never reached the LLM
        if not self.answer.found:
            return "yanlis-ret"  # evidence was there, it still said "not found"
        return "cevap"  # evidence was there, the answer is wrong or incomplete


def summarize(results: list[Result]) -> dict:
    """The headline numbers of one exam run."""
    answerable = [r for r in results if r.question.should_answer]
    refuse = [r for r in results if not r.question.should_answer]
    judged = [r for r in results if r.faithful is not None]
    ranks = [r.rank for r in answerable]

    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "questions": len(results),
        "score": mean([r.score for r in results]),
        "answer_score": mean([r.score for r in answerable]),
        "refusal_score": mean([r.score for r in refuse]),
        "hit5": mean([1.0 if rank and rank <= 5 else 0.0 for rank in ranks]),
        "recall5": mean([r.recall5 for r in answerable]),
        "mrr10": mean([1 / rank if rank else 0.0 for rank in ranks]),
        "faithful": mean([1.0 if r.faithful else 0.0 for r in judged]),
        "false_refusals": sum(1 for r in answerable if not r.answer.found),
        "input_tokens": mean([r.answer.input_tokens for r in results]),
        "seconds": mean([r.seconds for r in results]),
    }


# --- a cache so re-running the exam costs nothing ---------------------------------


@dataclass
class CachedLLM:
    """Wraps a provider: the same (system, prompt) is answered from disk the second time.

    The free tier allows a limited number of calls per day. With the cache, re-running
    the exam after a change only pays for the questions whose prompt actually changed.
    """

    inner: LLMProvider
    pause: float = 0.0  # seconds to wait after a real (non-cached) call
    cache_dir: Path = CACHE_DIR
    calls: int = field(default=0, init=False)
    cached: int = field(default=0, init=False)

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    def generate(self, system: str, prompt: str) -> LLMAnswer:
        key = hashlib.sha256(f"{self.model}\n{system}\n{prompt}".encode()).hexdigest()
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            self.cached += 1
            return LLMAnswer(**json.loads(path.read_text(encoding="utf-8")))
        reply = self.inner.generate(system, prompt)
        self.calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reply.__dict__), encoding="utf-8")
        time.sleep(self.pause)
        return reply
