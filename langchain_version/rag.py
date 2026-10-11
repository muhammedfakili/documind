"""A14: DocuMind's query pipeline written with LangChain -- for comparison, the product does not use it.

    Documents (our A2 loaders) ─▶ chunks ─▶ FastEmbed vectors ─▶ InMemoryVectorStore ─┐
                                        └─▶ BM25Retriever (rank_bm25) ───────────────┴─ EnsembleRetriever (RRF)
                                                                                              │ top 5
                       ChatPromptTemplate (our SYSTEM_PROMPT) ◀── numbered sources ◀──────────┘
                                │
                                ▼
                ChatGoogleGenerativeAI (same Gemini model) ─▶ StrOutputParser ─▶ answer

What LangChain gives for free: the building blocks above, each one line. What it does not
give out of the box (we built them in A3-A10): heading-aware chunks with a "Title > Section"
header, BM25 inside Postgres, the cross-encoder reranker, effective dates in the prompt,
the second try after "not found". scripts/a14_langchain.py measures what that difference is worth.

Reading data/raw reuses app/ingest (A2): LangChain's Excel loader needs the heavy
`unstructured` package, and the comparison is about retrieval and answering, not file parsing.
"""

from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable

from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document as LCDocument
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.generate import SYSTEM_PROMPT
from app.ingest.chunking import Chunk
from app.ingest.documents import Document

with warnings.catch_warnings():  # "langchain-community is being sunset" -- true, and part of the lesson
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.embeddings import FastEmbedEmbeddings
    from langchain_community.retrievers import BM25Retriever

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# --- turning our data into LangChain Documents -----------------------------------------


def _metadata(doc: Document, section: str | None = None) -> dict:
    return {
        "source": doc.source,
        "title": doc.title,
        "section": section,
        "page": doc.page,
        "row": doc.row,
        "audience": doc.audience,
        "effective_date": doc.effective_date,
    }


def from_chunks(chunks: list[Chunk]) -> list[LCDocument]:
    """Our A3 chunks (heading-aware, with the "Title > Section" header) as LangChain Documents."""
    return [LCDocument(page_content=c.text, metadata=_metadata(c.doc, c.section)) for c in chunks]


def default_split(documents: list[Document]) -> list[LCDocument]:
    """What a LangChain tutorial does: RecursiveCharacterTextSplitter, 1000 characters, 200 overlap."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(
        [LCDocument(page_content=d.text, metadata=_metadata(d)) for d in documents]
    )


# --- retrieval --------------------------------------------------------------------------


def embeddings() -> FastEmbedEmbeddings:
    # The same model as app/embedding.py. LangChain does not add bge's query instruction.
    return FastEmbedEmbeddings(model_name=settings.embed_model, cache_dir=str(settings.embed_cache_dir))


def _visible(audience: str) -> Callable[[LCDocument], bool]:
    return lambda d: audience == "all" or d.metadata["audience"] == "public"


def vector_retriever(store: InMemoryVectorStore, audience: str, k: int) -> Runnable:
    return store.as_retriever(search_kwargs={"k": k, "filter": _visible(audience)})


def words(text: str) -> list[str]:
    """Lowercase words, punctuation removed; codes like e-214 stay whole."""
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


def hybrid_retriever(
    store: InMemoryVectorStore, docs: list[LCDocument], audience: str, k: int, tokenize: bool = False
) -> Runnable:
    """Vector + BM25, merged by EnsembleRetriever (weighted Reciprocal Rank Fusion, c=60 like ours).

    tokenize=False is LangChain's default: BM25Retriever splits on spaces only, so "E-214:" in
    a chunk and "E-214?" in a question are different words, and so are "Pulse" and "pulse".
    scripts/a14_langchain.py measures both.
    """
    visible = [d for d in docs if _visible(audience)(d)]
    extra = {"preprocess_func": words} if tokenize else {}
    bm25 = BM25Retriever.from_documents(visible, k=30, **extra)
    vector = vector_retriever(store, audience, k=30)
    ensemble = EnsembleRetriever(retrievers=[vector, bm25], weights=[0.5, 0.5])
    return ensemble | RunnableLambda(lambda found: found[:k])


# --- the chain --------------------------------------------------------------------------


def format_sources(docs: list[LCDocument]) -> str:
    """The same numbered source format as app/generate.py, so the prompts are identical."""
    blocks = []
    for n, d in enumerate(docs, start=1):
        m = d.metadata
        where = (
            m["source"]
            + (f", page {m['page']}" if m["page"] else "")
            + (f", row {m['row']}" if m["row"] else "")
        )
        dated = f", effective {m['effective_date']:%Y-%m-%d}" if m["effective_date"] else ""
        blocks.append(f"[{n}] ({where}{dated})\n{d.page_content}")
    return "\n\n".join(blocks)


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT.replace("{", "{{").replace("}", "}}")),
        ("human", "Sources:\n\n{sources}\n\nQuestion: {question}"),
    ]
)


def rag_chain(retriever: Runnable, llm: Runnable) -> Runnable:
    """question -> {"question", "docs", "answer"}. LCEL: every `|` passes one step's output to the next."""
    write = (
        RunnableLambda(lambda x: {"sources": format_sources(x["docs"]), "question": x["question"]})
        | PROMPT
        | llm
        | StrOutputParser()
    )
    return RunnableParallel(docs=retriever, question=RunnablePassthrough()) | RunnablePassthrough.assign(
        answer=write
    )
