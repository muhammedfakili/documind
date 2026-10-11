"""A11: the HTTP door to DocuMind -- the same query pipeline, reachable from any program.

    POST /ask      question in -> cited answer + the sources behind it (JSON)
    POST /ingest   re-read data/raw and rebuild the database           (admin key)
    GET  /health   is the database reachable, how many chunks are in it

Run it (database running):
  uv run uvicorn app.api:app --reload
then open http://127.0.0.1:8000/docs -- FastAPI writes an interactive page from this file.

Nothing about RAG lives here. Every endpoint calls a function that already exists
(generate.answer, pipeline.ingest_directory, store.count) and only translates between
HTTP and Python: checks the input, picks the status code, shapes the JSON.

Security: a customer must never reach internal documents (A5's audience filter).
audience="all" (the support agent's mode) and /ingest need the X-API-Key header to
match ADMIN_API_KEY from .env. Without ADMIN_API_KEY set, both are switched off.
"""

from __future__ import annotations

import secrets
import time
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import date
from functools import lru_cache
from typing import Annotated, Literal

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from app import generate, store
from app.config import DATA_DIR, settings
from app.embedding import embed_query
from app.ingest.pipeline import ingest_directory
from app.llm import LLMProvider, LLMRefused, ProviderNotConfigured, get_llm
from app.rerank import _model as rerank_model


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load everything slow while the server starts, not during the first customer's question.

    Measured in scripts/a11_api.py: without this, the first /ask paid several seconds of
    model loading (embedding + reranker from disk) and of building the LLM client.
    """
    embed_query("warm up")
    rerank_model(settings.rerank_model)
    try:
        get_answer_llm()
    except HTTPException:  # no API key: the server still starts, /ask answers 503 with the reason
        pass
    ingest_if_empty()
    yield


def ingest_if_empty() -> None:
    """A13: a brand-new database (the first `docker compose up`) is filled once, by itself.

    A database that already has chunks is left alone: re-ingesting on every start would
    cost ~15 s for nothing. Updating documents later is still POST /ingest.
    """
    try:
        with store.connect() as conn:
            try:
                if sum(store.count(conn).values()) > 0:
                    return
            except psycopg.errors.UndefinedTable:
                pass  # no table yet: ingest_directory creates it
            ingest_directory(conn, DATA_DIR / "raw")
    except psycopg.OperationalError:
        pass  # database down: the server still starts, /health reports it


app = FastAPI(
    title="DocuMind",
    version="0.1.0",
    description="Questions in, answers with sources out -- from Lumora Audio's documents.",
    lifespan=lifespan,
)


# --- request / response shapes ------------------------------------------------------
# Pydantic checks every request against these before our code runs: a missing or
# empty question never reaches the LLM, it is answered with 422 and a clear message.


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["How long do I have to return opened earbuds?"])
    audience: Literal["public", "all"] = "public"  # "all" = support agent, needs the admin key
    k: int = Field(default=5, ge=1, le=10, description="how many sources the LLM reads")


class Source(BaseModel):
    n: int  # the [n] the answer uses
    location: str  # "manuals/pulse-user-manual.pdf, page 3"
    title: str
    section: str | None
    audience: str  # "public" or "internal" (internal only reaches support agents)
    effective_date: date | None
    score: float
    cited: bool  # did the answer actually use it?
    ranks: dict[str, int]  # A12: {"vector": 4, "keyword": 1, "hybrid": 2, "reranker": 1} -- why it was picked
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool  # False = "I couldn't find this in the documents."
    retried: bool  # A10: the first try found nothing, this came from rewritten queries
    cited: list[int]
    sources: list[Source]
    model: str
    input_tokens: int
    output_tokens: int
    seconds: float


class IngestResponse(BaseModel):
    chunks: int
    by_audience: dict[str, int]
    seconds: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    chunks: dict[str, int]


# --- dependencies: what an endpoint needs, built once per request ------------------


def get_conn() -> Iterator[psycopg.Connection]:
    """One database connection per request, always closed afterwards."""
    try:
        conn = store.connect()
    except psycopg.OperationalError as exc:
        raise HTTPException(503, "Database unavailable. Is the db container running?") from exc
    try:
        yield conn
    finally:
        conn.close()


@lru_cache(maxsize=1)
def get_answer_llm() -> LLMProvider:
    """The answer model, built once: a new API client per request would be wasted work."""
    try:
        return get_llm()
    except ProviderNotConfigured as exc:
        raise HTTPException(503, str(exc)) from exc


def check_admin(x_api_key: str | None) -> None:
    if not settings.admin_api_key:
        raise HTTPException(403, "Admin features are off. Set ADMIN_API_KEY in .env to turn them on.")
    # compare_digest takes the same time whether the first or the last character is wrong,
    # so the key cannot be guessed one character at a time by measuring response times.
    if not secrets.compare_digest((x_api_key or "").encode(), settings.admin_api_key.encode()):
        raise HTTPException(401, "Missing or wrong X-API-Key header.")


def require_admin(x_api_key: Annotated[str | None, Header()] = None) -> None:
    check_admin(x_api_key)


Conn = Annotated[psycopg.Connection, Depends(get_conn)]


# --- endpoints ---------------------------------------------------------------------


@app.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    conn: Conn,
    llm: Annotated[LLMProvider, Depends(get_answer_llm)],
    x_api_key: Annotated[str | None, Header()] = None,
) -> AskResponse:
    """Answer a question from the documents, with numbered sources."""
    if req.audience == "all":
        check_admin(x_api_key)
    started = time.perf_counter()
    try:
        ans = generate.answer(conn, req.question, k=req.k, audience=req.audience, llm=llm)
    except LLMRefused as exc:  # the provider's safety filter blocked the prompt
        raise HTTPException(422, f"The language model refused this question: {exc}") from exc
    except psycopg.Error as exc:
        raise HTTPException(503, "Database error while searching.") from exc
    except Exception as exc:  # provider down, quota used up, network: not the caller's fault
        raise HTTPException(502, f"The language model is not answering right now ({type(exc).__name__}).") from exc
    return AskResponse(
        question=ans.question,
        answer=ans.text,
        found=ans.found,
        retried=ans.retried,
        cited=ans.cited,
        sources=[
            Source(
                n=n,
                location=hit.location,
                title=hit.title,
                section=hit.section,
                audience=hit.audience,
                effective_date=hit.effective_date,
                score=round(hit.score, 4),
                cited=n in ans.cited,
                ranks=hit.ranks,
                text=hit.text,
            )
            for n, hit in enumerate(ans.sources, start=1)
        ],
        model=ans.model,
        input_tokens=ans.input_tokens,
        output_tokens=ans.output_tokens,
        seconds=round(time.perf_counter() - started, 2),
    )


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_admin)])
def ingest(conn: Conn) -> IngestResponse:
    """Re-read every file in data/raw and replace the database contents (~15 s)."""
    started = time.perf_counter()
    n = ingest_directory(conn, DATA_DIR / "raw")
    return IngestResponse(chunks=n, by_audience=store.count(conn), seconds=round(time.perf_counter() - started, 2))


@app.get("/health", response_model=HealthResponse)
def health(conn: Conn) -> HealthResponse:
    """Database reachable and filled? (A monitoring tool calls this every minute.)"""
    try:
        chunks = store.count(conn)
    except psycopg.errors.UndefinedTable as exc:
        raise HTTPException(503, "Database is empty. Run POST /ingest first.") from exc
    return HealthResponse(status="ok", chunks=chunks)
