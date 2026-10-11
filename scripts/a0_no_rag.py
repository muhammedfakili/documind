"""A0 experiment: ask the LLM about our company WITHOUT giving it any documents.

Lumora Audio is a fictional headphone/speaker brand; its documents only exist
in data/raw (written in A1). The model has never seen them, so every answer here
is either a refusal or a guess. That gap is the entire reason RAG exists.

We ask each question twice:
  1. "plain"  -- a normal support-assistant prompt
  2. "honest" -- the same prompt, plus "say you don't know if you don't know"

Usage (from the project root):  uv run python -m scripts.a0_no_rag
"""

from __future__ import annotations

import time
from datetime import datetime

from app.config import PROJECT_ROOT
from app.llm import get_llm

QUESTIONS = [
    "What is Lumora Audio's return window for headphones that have been opened?",
    "How long is the warranty on the Lumora Aria X2 speaker, and does it cover water damage?",
    "What does error code E-214 mean on Lumora Pulse earbuds, and how do I fix it?",
    "In Lumora's support team, who has to approve a refund above $500?",
]

SYSTEMS = {
    "plain": "You are the customer support assistant for Lumora Audio. Answer the customer's question.",
    "honest": (
        "You are the customer support assistant for Lumora Audio. Answer the customer's question. "
        "If you do not actually know the answer, say so plainly instead of guessing."
    ),
}


def main() -> None:
    llm = get_llm()
    lines = [f"# A0 — RAG'siz deney ({llm.name} / {llm.model}, {datetime.now().astimezone():%Y-%m-%d %H:%M})", ""]

    for question in QUESTIONS:
        lines += [f"## Soru: {question}", ""]
        for label, system in SYSTEMS.items():
            answer = llm.generate(system, question)
            lines += [f"**[{label}]**", "", answer.text.strip(), ""]
            print(f"\n=== [{label}] {question}\n{answer.text.strip()}")
            time.sleep(4)  # stay under the free tier's per-minute limit

    out = PROJECT_ROOT / "eval" / "results" / "a0_no_rag.md"
    # utf-8-sig writes a BOM so Windows viewers don't misread Turkish characters.
    out.write_text("\n".join(lines), encoding="utf-8-sig")
    print(f"\nSaved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
