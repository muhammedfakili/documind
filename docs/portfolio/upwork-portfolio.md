# Upwork portfolio entry — DocuMind

Same layout as ScrapePilot: title, role, 5 skills, one dense paragraph (600-character limit), cover + 5 images.

**Title** (62 characters)
DocuMind — Advanced RAG Assistant with Cited, Measured Answers

**Role**
AI Engineer

**Skills**
Retrieval Augmented Generation · Python · LangChain · PostgreSQL · FastAPI

**Description** (587 / 600 characters)
Ask a company's documents a question, get an answer that cites its sources, or an honest "not found". Built on a synthetic knowledge base with deliberate traps: conflicting policy versions, tables, internal-only docs. Hybrid search (pgvector + BM25 in Postgres, RRF), a local cross-encoder reranker, and a query-rewrite retry only after a miss. Measured at every stage on the same 55-question exam: answer score 90% to 98%, evidence in the top 5 for all 46 answerable questions, 9/9 correct refusals, 100% faithful. Same pipeline in LangChain: 83%. FastAPI, Streamlit, Docker, 117 tests.

**Images, in this order**
| File | Shows |
|---|---|
| `00-cover.png` | Cover: black background, robot, "ADVANCED RAG / DOCUMENT Q&A / ASSISTANT" (same style as the other covers) |
| `01-cited-answer.png` | The return-policy trap: "No", 14 days, cited [1], source card with the 2026 policy table and its effective date |
| `02-why-this-source.png` | "Why this source?": vector #1, BM25 #17, fused #5, reranker #1 for the UK price row |
| `03-customer-not-found.png` | A customer asks about reseller margins: the answer is only in an internal document, so "not found" |
| `04-agent-internal.png` | The same question from a support agent with the key: 38% from the internal partner guide, red "internal" badge |
| `05-measured-results.png` | Answer score and hit@5 per stage (90→91→93→98%) next to the LangChain version (83%) |

Every number comes from `eval/results/` (eval_baseline, eval_hybrid, eval_rerank, eval_a10, a14_langchain).
Images are reproducible: `shots.py` (Playwright against `docker compose up`) and `chart.py` in this folder.
