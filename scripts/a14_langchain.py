"""A14: the same exam for the LangChain version (langchain_version/rag.py) and for ours.

Outputs:
  eval/results/a14_langchain.md     explained report (Turkish)

Two parts:
  1. retrieval, no LLM (deterministic): six searches on the 46 answerable questions --
     LangChain's tutorial default, LangChain on our chunks (vector, then hybrid), and our
     A7 / A8 / A9 searches. Each step changes one thing, so the table says what each is worth.
  2. answers: the LangChain chain (hybrid on our chunks, our prompt, the same Gemini model)
     on all 55 questions, graded by the same judge as eval/run_eval.py. Compared with our
     saved runs: hybrid (A8), rerank (A9) and a10 (the product).

Gemini calls are cached by LangChain itself (eval/cache/langchain.sqlite), judge calls by
CachedLLM, so a re-run is free.

Usage (database running):
  uv run python -m scripts.a14_langchain
"""

from __future__ import annotations

import json
import time
import warnings
from collections.abc import Callable

from langchain_core.globals import set_llm_cache
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI

from app import store
from app.config import DATA_DIR, PROJECT_ROOT, settings
from app.evaluation import (
    CachedLLM,
    Result,
    evidence_recall,
    first_relevant_rank,
    judge,
    load_questions,
    summarize,
)
from app.generate import Answer, is_not_found, parse_citations
from app.ingest.chunking import chunk_documents
from app.ingest.pipeline import load_directory
from app.llm import get_judge
from app.retrieve import retrieve
from app.store import Hit
from langchain_version import rag

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from langchain_community.cache import SQLiteCache

REPORT = PROJECT_ROOT / "eval" / "results" / "a14_langchain.md"
RUNS = PROJECT_ROOT / "eval" / "results" / "runs"
PAUSE = 6  # after a real Gemini call (free tier, ~10 per minute)


def to_hit(d) -> Hit:
    m = d.metadata
    return Hit(
        text=d.page_content,
        source=m["source"],
        title=m["title"],
        section=m["section"],
        page=m["page"],
        row=m["row"],
        audience=m["audience"],
        effective_date=m["effective_date"],
        score=0.0,
    )


def retrieval_scores(search: Callable[[str, str], list[Hit]], questions) -> dict:
    ranks, recalls, ms, chars = [], [], [], []
    for q in questions:
        started = time.perf_counter()
        hits = search(q.question, q.audience)
        ms.append((time.perf_counter() - started) * 1000)
        ranks.append(first_relevant_rank(hits[:10], q.evidence))
        recalls.append(evidence_recall(hits[:5], q.evidence))
        chars.append(sum(len(h.text) for h in hits[:5]))  # what the LLM has to read (and we pay for)
    n = len(questions)
    return {
        "hit5": sum(1 for r in ranks if r and r <= 5) / n,
        "recall5": sum(recalls) / n,
        "mrr10": sum(1 / r for r in ranks if r) / n,
        "ms": sorted(ms)[n // 2],  # median
        "chars": sum(chars) / n,
    }


def main() -> None:
    questions = load_questions()
    answerable = [q for q in questions if q.should_answer]

    documents = load_directory(DATA_DIR / "raw")
    ours = rag.from_chunks(chunk_documents(documents))
    default = rag.default_split(documents)
    emb = rag.embeddings()
    started = time.perf_counter()
    store_ours = InMemoryVectorStore.from_documents(ours, emb)
    store_default = InMemoryVectorStore.from_documents(default, emb)
    print(f"LangChain index: {len(ours)} + {len(default)} chunks in {time.perf_counter() - started:.1f} s")

    hybrid = {aud: rag.hybrid_retriever(store_ours, ours, aud, k=10) for aud in ("public", "all")}
    fixed = {
        aud: rag.hybrid_retriever(store_ours, ours, aud, k=10, tokenize=True) for aud in ("public", "all")
    }
    conn = store.connect()

    def lc(retriever_for: Callable[[str], object]) -> Callable[[str, str], list[Hit]]:
        return lambda question, audience: [to_hit(d) for d in retriever_for(audience).invoke(question)]

    variants = [
        (
            "LangChain varsayılanı: 1000 karakter parça, vektör",
            "A3 öncesi parçalar, sorgu talimatı yok",
            lc(lambda a: rag.vector_retriever(store_default, a, 10)),
        ),
        (
            "LangChain, bizim parçalar, vektör",
            "sadece parçalar değişti",
            lc(lambda a: rag.vector_retriever(store_ours, a, 10)),
        ),
        (
            "Bizim vektör (A7)",
            "aynı parçalar, Postgres'te (sonuç aynı)",
            lambda question, a: retrieve(conn, question, k=10, audience=a, mode="vector", rerank=False),
        ),
        ("LangChain hybrid (EnsembleRetriever)", "vektör + rank_bm25, RRF", lc(lambda a: hybrid[a])),
        (
            "LangChain hybrid, BM25 düzeltildi",
            "kelimeler küçük harf, noktalama ayrıldı",
            lc(lambda a: fixed[a]),
        ),
        (
            "Bizim hybrid (A8)",
            "vektör + Postgres BM25, RRF",
            lambda question, a: retrieve(conn, question, k=10, audience=a, rerank=False),
        ),
        (
            "Bizim hybrid + reranker (A9)",
            "+ cross-encoder",
            lambda question, a: retrieve(conn, question, k=10, audience=a),
        ),
    ]
    retrieval = []
    for name, change, search in variants:
        scores = retrieval_scores(search, answerable)
        retrieval.append((name, change, scores))
        print(
            f"{name}: hit@5 {scores['hit5']:.0%} recall@5 {scores['recall5']:.0%} MRR {scores['mrr10']:.2f}"
        )

    # --- part 2: the LangChain chain answers the whole exam ---------------------------
    set_llm_cache(SQLiteCache(database_path=str(PROJECT_ROOT / "eval" / "cache" / "langchain.sqlite")))
    llm = ChatGoogleGenerativeAI(model=settings.llm_model_resolved, google_api_key=settings.google_api_key)
    chains = {
        aud: rag.rag_chain(rag.hybrid_retriever(store_ours, ours, aud, k=5), llm) for aud in ("public", "all")
    }
    grader = CachedLLM(get_judge())
    results = []
    for i, q in enumerate(questions, start=1):
        for attempt in range(1, 5):
            try:
                started = time.perf_counter()
                out = chains[q.audience].invoke(q.question)
                seconds = time.perf_counter() - started
                hits = [to_hit(d) for d in out["docs"]]
                text = out["answer"].strip()
                found = not is_not_found(text)
                ans = Answer(
                    question=q.question,
                    text=text,
                    sources=hits,
                    found=found,
                    model=llm.model,
                    cited=parse_citations(text, len(hits)) if found else [],
                )
                verdict = judge(grader, q, ans) if found else None
                break
            except Exception as exc:  # free-tier hiccups: wait and retry, finished calls are cached
                if attempt == 4:
                    raise
                print(f"  {q.id}: {type(exc).__name__}, retrying in {attempt * 60} s", flush=True)
                time.sleep(attempt * 60)
        results.append(Result(q, hits, ans, verdict, seconds))
        print(f"[{i}/{len(questions)}] {q.id} score={results[-1].score} {text[:60]!r}", flush=True)
        if seconds > 0.5:  # a real call, not the cache
            time.sleep(PAUSE)

    REPORT.write_text("\n".join(report(retrieval, results, summarize(results))), encoding="utf-8-sig")
    print(f"Saved to {REPORT.relative_to(PROJECT_ROOT)}")


def code_lines(*paths: str) -> int:
    """Lines that are code: no blank lines, comments or docstring lines (rough, same rule for both)."""
    count = 0
    for path in paths:
        in_doc = False
        for line in (PROJECT_ROOT / path).read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s.startswith(('"""', "'''")):
                if not (len(s) > 3 and s.endswith(('"""', "'''"))):
                    in_doc = not in_doc
                continue
            if in_doc or not s or s.startswith("#"):
                continue
            count += 1
    return count


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def report(retrieval, results, s) -> list[str]:
    saved = {
        name: json.loads((RUNS / f"{name}.json").read_text(encoding="utf-8"))["summary"]
        for name in ("hybrid", "rerank", "a10")
    }
    lc_lines = code_lines("langchain_version/rag.py")
    our_lines = code_lines(
        "app/retrieve.py", "app/rerank.py", "app/rewrite.py", "app/generate.py", "app/store.py"
    )
    misses = [r for r in results if r.score < 1]
    L = [
        "# A14: LangChain ile aynı sistem — karşılaştırma",
        "",
        "> **Kısaca sonuç:** (rakamlar aşağıdaki tablolardan)",
        f"> - LangChain'in hazır parçalarıyla aynı hattı **~{lc_lines} satırda** kurduk. Bizim soru hattımız",
        f">   ~{our_lines} satır (Postgres, BM25, reranker, ikinci deneme dahil).",
        "> - **Kutudan çıktığı gibi hybrid aramayı bozdu:** LangChain'in BM25Retriever'ı kelimeleri sadece",
        ">   boşluktan ayırıyor (`E-214:` ≠ `E-214?`, `Pulse` ≠ `pulse`). Vektöre eklenince hit@5 düştü;",
        ">   kelime ayırmayı düzeltmek bir kısmını kurtardı; Postgres BM25 (kök bulma + stop words) ile 96%.",
        "> - **Sürpriz:** LangChain'in varsayılan 1000 karakterlik parçaları aramada bizim parçalardan iyi çıktı,",
        ">   ama LLM'e her soruda ~2 kat metin gidiyor ve doğru parça ortalamada daha aşağıda (MRR). Cevap",
        ">   puanına etkisini ölçmedik.",
        f"> - LangChain zinciri sınavda **{pct(s['score'])}** aldı. Aynı aşamadaki sistemimiz (A8 hybrid)",
        f">   {pct(saved['hybrid']['score'])}, ürünümüz (A10) {pct(saved['a10']['score'])}. Kaybedilen 10 sorunun",
        '>   yarısı tablo sorusu; kanıtın gelmediği sorularda LLM doğru şekilde "bulamadım" dedi.',
        ">",
        "> Aynı 55 soru, aynı cevap modeli, aynı prompt, aynı hakem. Betik: `scripts/a14_langchain.py`,",
        "> LangChain kodu: `langchain_version/rag.py`. Ders: `dersler/14-langchain.md`.",
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        "| **LangChain** | RAG ve LLM uygulamaları için hazır parçalar kütüphanesi: okuyucu, bölücü, vektör deposu, retriever, zincir. |",
        '| **Retriever** | LangChain\'de "soru gir, belge listesi al" arayüzü. Bizim `retrieve()` fonksiyonunun karşılığı. |',
        "| **EnsembleRetriever** | Birden fazla retriever'ın sonucunu RRF ile birleştirir. Bizim A8'deki `rrf_fuse`'un karşılığı. |",
        "| **LCEL** | LangChain'in `a \\| b \\| c` yazımı: bir adımın çıktısı bir sonrakine girer. |",
        "",
        "## 1. Arama (LLM yok, 46 cevaplanabilir soru)",
        "",
        '> 👀 **Neye bak:** Her satır bir öncekinden **tek bir şeyi** değiştiriyor ("Ne değişti" sütunu).',
        "> Hangi değişiklik en çok puan getiriyor? Süre = soru başına ortanca arama süresi.",
        "> LLM'e giden metin = ilk 5 parçanın toplam uzunluğu: büyük parça daha çok kanıt taşır ama her soruda",
        "> daha çok token (para, süre) demek.",
        "",
        "| Arama | Ne değişti | hit@5 | recall@5 | MRR@10 | LLM'e giden metin | Süre |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, change, sc in retrieval:
        L.append(
            f"| {name} | {change} | {pct(sc['hit5'])} | {pct(sc['recall5'])} | {sc['mrr10']:.2f} "
            f"| {sc['chars']:.0f} karakter | {sc['ms']:.0f} ms |"
        )
    L += [
        "",
        "## 2. Cevaplar (55 soru, hakem: aynı NVIDIA modeli)",
        "",
        "> 👀 **Neye bak:** LangChain zinciri bizim A8 sistemimizle aynı seviyede mi? A9-A10'un eklediği",
        "> parçalar (reranker, ikinci deneme) LangChain'de hazır gelmiyor.",
        "",
        '| Sistem | Genel | Cevaplanabilir | "Bulamadım" soruları | hit@5 | Sadakat |',
        "|---|---:|---:|---:|---:|---:|",
        (
            f"| **LangChain zinciri** (hybrid, bizim parçalar ve prompt) | **{pct(s['score'])}** | "
            f"{pct(s['answer_score'])} | {pct(s['refusal_score'])} | {pct(s['hit5'])} | {pct(s['faithful'])} |"
        ),
    ]
    for name, label in (
        ("hybrid", "Bizim A8 (hybrid)"),
        ("rerank", "Bizim A9 (+ reranker)"),
        ("a10", "Bizim A10 (ürün: + ikinci deneme)"),
    ):
        x = saved[name]
        L.append(
            f"| {label} | {pct(x['score'])} | {pct(x['answer_score'])} | {pct(x['refusal_score'])} | "
            f"{pct(x['hit5'])} | {pct(x['faithful'])} |"
        )
    L += [
        "",
        "Not: hit@5 burada LLM'e verilen 5 parçaya bakıyor. Bölüm 1 ile aynı arama, aynı rakam olmalı.",
        "",
        f"**LangChain zincirinin tam puan alamadığı {len(misses)} soru:**",
        "",
        "| Soru | Puan | Cevap |",
        "|---|---:|---|",
    ]
    for r in misses:
        text = " ".join(r.answer.text.split()).replace("|", "/")
        L.append(
            f"| {r.question.id}: {r.question.question} | {r.score} | {text[:150]}{' …' if len(text) > 150 else ''} |"
        )
    L.append("")
    return L


if __name__ == "__main__":
    main()
