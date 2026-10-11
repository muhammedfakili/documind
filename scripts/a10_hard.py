"""A10: the hard cases -- query rewriting, and old vs new versions of a document.

Outputs:
  eval/results/a10_hard.md     explained report (Turkish)

Experiments:
  1. query rewriting, retrieval only (the 46 answerable exam questions, same evidence
     check as A7-A9). Four ways to use the LLM's rewrites:
       none        A9's search, the question alone
       equal       question + every rewrite, one vote each in RRF
       question x2 the same, but the customer's own question votes twice
       rewrites    only the rewrites, the question itself is not searched
     Every variant ends with A9's reranker reading the top 10 against the ORIGINAL question.
  2. what the rewrites look like, and how slow the extra LLM call is
  3. version stress test: the old document is put FIRST in the sources on purpose,
     and the A9 prompt (no dates) is compared with the A10 prompt (dates + "newest wins").
     Answers are checked by code (does the right number appear?), 3 tries each,
     because the answer model is not deterministic.
The answer-level effect of everything: eval/run_eval.py --name a10 --rewrite --compare rerank.

Usage (from the project root, database running, uses the Gemini free tier):
  uv run python -m scripts.a10_hard
"""

from __future__ import annotations

import json
import re
import time
from collections import defaultdict
from statistics import mean, median

from app import store
from app.config import PROJECT_ROOT
from app.evaluation import CachedLLM, Question, evidence_recall, first_relevant_rank, load_questions
from app.generate import SYSTEM_PROMPT, build_prompt
from app.llm import get_llm
from app.rerank import rerank
from app.retrieve import CANDIDATES, RERANK_CANDIDATES, _search, retrieve, rrf_fuse
from app.rewrite import rewrite_query
from app.store import Hit

REPORT = PROJECT_ROOT / "eval" / "results" / "a10_hard.md"

CATEGORY_LABEL = {
    "single": "Tek belge",
    "table": "Tablo / rakam",
    "multi": "Çok belge",
    "conflict": "Tuzak",
    "internal": "İç belge (destek)",
}
VARIANTS = {
    "none": "Yeniden yazma yok (A9)",
    "equal": "Soru + yeniden yazılanlar, eşit oy",
    "double": "Soru ×2 + yeniden yazılanlar",
    "only": "Sadece yeniden yazılanlar",
}

# A9's answer prompt, copied here unchanged, so experiment 3 can compare old and new.
A9_SYSTEM_PROMPT = SYSTEM_PROMPT.split("- Each source shows the date")[0] + (
    "- Be short and direct: a few sentences or a short list."
)

# Experiment 3: (question, old-version source, text in its outdated chunk, the right answer as a regex)
VERSION_CASES = [
    (
        "I opened my Pulse Pro earbuds and it's been 20 days since delivery. Can I still return them?",
        "policies/return-policy-2025.md",
        "30 days from the delivery date",
        r"\b14\b",
    ),
    (
        "How many days do I have to return opened Halo ANC headphones?",
        "policies/return-policy-2025.md",
        "30 days from the delivery date",
        r"\b14\b",
    ),
    (
        "How many Aria X2 speakers can I connect in Party Mode?",
        "manuals/aria-x2-user-manual.pdf",
        "Up to 4 Aria X2",
        r"\b8\b",
    ),
]
TRIES = 3
REFUSE = {"hidden", "unanswerable"}


def old_prompt(question: str, hits: list[Hit]) -> str:
    """A9's prompt format: location only, no dates."""
    blocks = [f"[{n}] ({h.location})\n{h.text}" for n, h in enumerate(hits, start=1)]
    return "Sources:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def search_variant(conn, q: Question, rewrites: list[str], variant: str) -> list[Hit]:
    if variant == "none":
        return retrieve(conn, q.question, k=10, audience=q.audience)
    depth = max(RERANK_CANDIDATES, CANDIDATES)
    original = _search(conn, q.question, depth, q.audience, "hybrid")
    extra = [_search(conn, r, depth, q.audience, "hybrid") for r in rewrites]
    lists = {"equal": [original, *extra], "double": [original, original, *extra], "only": extra or [original]}
    hits = rrf_fuse(lists[variant])[:RERANK_CANDIDATES]
    return rrf_fuse([hits, rerank(q.question, hits)])[:10]


def grade(questions: list[Question], ranked: dict[str, list[Hit]]) -> dict[str, float]:
    ranks = [first_relevant_rank(ranked[q.id], q.evidence) for q in questions]
    return {
        "hit5": mean(1.0 if r and r <= 5 else 0.0 for r in ranks),
        "recall5": mean(evidence_recall(ranked[q.id][:5], q.evidence) for q in questions),
        "mrr": mean(1 / r if r else 0.0 for r in ranks),
    }


def timed_ms(fn, runs: int = 3) -> float:
    best = float("inf")
    for _ in range(runs):
        started = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - started) * 1000)
    return best


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def main() -> None:
    conn = store.connect()
    questions = [q for q in load_questions() if q.should_answer]
    llm = CachedLLM(get_llm(), pause=4)  # same cache as run_eval: the exam reuses these rewrites

    # 1-2. rewriting
    rewrites = {}
    for i, q in enumerate(questions, start=1):
        rewrites[q.id] = rewrite_query(q.question, llm)
        print(f"[{i}/{len(questions)}] {q.id} {rewrites[q.id]}", flush=True)
    ranked = {v: {q.id: search_variant(conn, q, rewrites[q.id], v) for q in questions} for v in VARIANTS}
    scores = {v: grade(questions, ranked[v]) for v in VARIANTS}

    timings = []
    for q in questions[:5]:  # real calls, not from the cache
        started = time.perf_counter()
        rewrite_query(q.question, llm.inner)
        timings.append((time.perf_counter() - started) * 1000)
        time.sleep(4)
    for v in VARIANTS:  # warm-up: the first call loads the models, that is not search time
        search_variant(conn, questions[0], rewrites[questions[0].id], v)
    # Fastest of 3 runs per question: other programs on the laptop (browser, Docker) make
    # single timings jump between ~100 and ~500 ms for the same reranker call.
    search_ms = [timed_ms(lambda q=q: search_variant(conn, q, rewrites[q.id], "equal")) for q in questions[:10]]
    base_ms = [timed_ms(lambda q=q: search_variant(conn, q, rewrites[q.id], "none")) for q in questions[:10]]

    # 3. version stress test
    answer_llm = get_llm()
    version_rows = []
    for question, old_source, old_marker, right_answer in VERSION_CASES:
        deep = retrieve(conn, question, k=30, rerank=False)
        old = next(h for h in deep if h.source == old_source and old_marker in h.text)
        hits = [old, *[h for h in retrieve(conn, question, k=5) if h.text != old.text]][:5]
        print("stress sources:", [h.location for h in hits], flush=True)
        for label, system, prompt in [
            ("A9 prompt (tarih yok)", A9_SYSTEM_PROMPT, old_prompt(question, hits)),
            ("A10 prompt (tarih + en yeni kazanır)", SYSTEM_PROMPT, build_prompt(question, hits)),
        ]:
            right = 0
            sample = ""
            for _ in range(TRIES):
                text = answer_llm.generate(system, prompt).text.strip()
                right += bool(re.search(right_answer, text))
                sample = sample or text
                time.sleep(4)
            version_rows.append((question, old_source, label, right, sample))
            print(f"{label}: {right}/{TRIES}  {question[:50]}", flush=True)

    REPORT.write_text(
        "\n".join(report(questions, rewrites, ranked, scores, timings, search_ms, base_ms, version_rows)),
        encoding="utf-8-sig",
    )
    print(f"Saved to {REPORT.relative_to(PROJECT_ROOT)}")


def report(questions, rewrites, ranked, scores, timings, search_ms, base_ms, version_rows) -> list[str]:
    L = [
        "# A10: Zor durumlar — sorguyu yeniden yazma ve eski/yeni sürüm",
        "",
        "> **Kısaca sonuç:**",
        "> - Soruyu **her seferinde** yeniden yazmak iki ıskayı (s16, s17) düzeltti ama doğru bulunan dört",
        ">   sorunun kanıtını ilk 5'ten itti (t03, m01, m05, m06). Bu yüzden ürün yeniden yazmayı sadece",
        ">   **ikinci deneme** olarak kullanıyor: ilk cevap \"bulamadım\" ise.",
        "> - Tarihli kaynak + \"en yeni kazanır\" kuralı stres testinde fark yaratmadı: eski prompt da 9/9",
        ">   doğru. Belgelerimiz zaten kendini anlatıyor (\"Replaces ...\", \"previously 4\"). Kural ucuz bir",
        ">   sigorta olarak kaldı.",
        "> - Tam sınav (bölüm 5): **%93 → %98**. 9 \"bulamadım\" sorusu ikinci denemeden sonra da 9/9 doğru.",
        ">",
        "> Ders anlatımı: `dersler/10-zor-durumlar.md`. Bu dosyayı `scripts/a10_hard.py` üretiyor.",
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        ("| **Query rewriting (sorguyu yeniden yazma)** | LLM, müşterinin sorusunu belgelerin kullanacağı"
         " kelimelerle 1-3 arama sorgusuna çeviriyor. Her sorgu ayrı aranıyor, sonuçlar RRF ile"
         " birleşiyor. |"),
        ("| **İkinci deneme (fallback)** | Önce normal arama + cevap. Cevap \"bulamadım\" ise yeniden yazıp"
         " bir kez daha. Bulunan sorulara dokunulmuyor, ek maliyet sadece ıskalarda. |"),
        ("| **Sürüm çelişkisi** | Aynı konuda eski ve yeni belge (2025 / 2026 iade politikası). Kaynakların"
         " yanında yürürlük tarihi gösteriliyor, yeni olan kazanıyor. |"),
        "",
        "## 1. Her soruda yeniden yazma aramayı nasıl etkiliyor? (46 cevaplanabilir soru)",
        "",
        (
            "> 👀 **Neye bak:** MRR ve hit@5. Her satırın sonunda A9'un reranker'ı orijinal soruyla son sırayı"
            " belirliyor."
        ),
        "",
        "| Yöntem | hit@5 | recall@5 | MRR@10 |",
        "|---|---:|---:|---:|",
    ]
    for v, label in VARIANTS.items():
        s = scores[v]
        L.append(f"| {label} | {pct(s['hit5'])} | {pct(s['recall5'])} | {s['mrr']:.3f} |")

    L += ["", "### Kategoriye göre (MRR@10)", "", "| Kategori | Soru | " + " | ".join(VARIANTS) + " |",
          "|---|---:|" + "---:|" * len(VARIANTS)]
    by_cat = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)
    for cat, label in CATEGORY_LABEL.items():
        group = by_cat[cat]
        cells = [f"{grade(group, ranked[v])['mrr']:.2f}" for v in VARIANTS]
        L.append(f"| {label} | {len(group)} | " + " | ".join(cells) + " |")

    L += ["", "## 2. Soru soru: sırası değişenler (A9 → soru + yeniden yazılanlar, eşit oy)", "",
          "| Soru | Kanıtın sırası | recall@5 | Yeniden yazılan sorgular |", "|---|---|---|---|"]
    for q in questions:
        a, b = ranked["none"][q.id], ranked["equal"][q.id]
        ra, rb = first_relevant_rank(a, q.evidence), first_relevant_rank(b, q.evidence)
        ca, cb = evidence_recall(a[:5], q.evidence), evidence_recall(b[:5], q.evidence)
        if ra != rb or ca != cb:
            arrow = "📈" if (rb or 99) < (ra or 99) or cb > ca else "📉"
            L.append(f"| {arrow} {q.id} | {ra or '—'} → {rb or '—'} | {pct(ca)} → {pct(cb)} |"
                     f" {' / '.join(rewrites[q.id])} |")

    L += ["", "### Hedef sorular", ""]
    for qid in ("s16", "s17", "m01"):
        q = next(q for q in questions if q.id == qid)
        L.append(f"- **{qid}** \"{q.question}\" → {' / '.join(rewrites[qid])}")
    L += [
        "",
        "## 3. Hız",
        "",
        "| Adım | Süre |",
        "|---|---:|",
        f"| Yeniden yazma (LLM çağrısı, 5 sorunun ortancası) | {median(timings):.0f} ms |",
        f"| Arama + rerank, A9 (10 soru ort., 3 denemenin en hızlısı) | {mean(base_ms):.0f} ms |",
        f"| Arama + rerank, soru + yeniden yazılanlar (aynı ölçüm) | {mean(search_ms):.0f} ms |",
        "",
        "## 4. Sürüm stres testi: eski belge kaynak listesinde 1. sırada",
        "",
        (
            f"> Her satır {TRIES} deneme. Eski sürümün parçası bilerek 1. kaynak yapıldı; yeni sürüm de"
            " listede. Kod kontrol ediyor: güncel rakam (14 gün / 8 hoparlör) cevapta geçiyor mu?"
        ),
        "",
        "| Soru | Öne konan eski belge | Prompt | Doğru | Örnek cevap |",
        "|---|---|---|---:|---|",
    ]
    for question, old_source, label, right, sample in version_rows:
        one = " ".join(sample.split()).replace("|", "/")[:160]
        L.append(f"| {question} | `{old_source}` | {label} | {right}/{TRIES} | {one} |")
    L += [
        "",
        "> 👀 **Neye bak:** iki prompt da tam puan. Fark cevabın içinde: A10 promptu eski sürümü de",
        "> \"(The older 2025 policy allowed 30 days ...)\" diye anıyor. Müşteri için daha açıklayıcı.",
        "",
        *exam_section(),
    ]
    return [*L, ""]


def exam_section() -> list[str]:
    """Section 5: the three exam runs side by side (written by eval/run_eval.py)."""
    runs = {}
    for name in ("rerank", "a10-prompt", "a10"):
        path = PROJECT_ROOT / "eval" / "results" / "runs" / f"{name}.json"
        if path.exists():
            runs[name] = json.loads(path.read_text(encoding="utf-8"))
    if len(runs) < 3:
        return ["## 5. Tam sınav", "", "Önce `eval/run_eval.py` ile rerank, a10-prompt ve a10 çalıştırılmalı."]
    rows = [
        ("Genel puan", "score", pct),
        ("Cevaplanabilir sorular", "answer_score", pct),
        ('"Bulamadım" soruları', "refusal_score", pct),
        ("hit@5 (son kullanılan kaynaklar)", "hit5", pct),
        ("recall@5", "recall5", pct),
        ("Sadakat", "faithful", pct),
        ("Token / soru", "input_tokens", lambda x: f"{x:.0f}"),
    ]
    L = [
        "## 5. Tam sınav: 55 soru, cevap + hakem",
        "",
        "| Metrik | A9 (`rerank`) | + yeni prompt (`a10-prompt`) | + ikinci deneme (`a10`) |",
        "|---|---:|---:|---:|",
    ]
    for label, key, fmt in rows:
        L.append(f"| {label} | " + " | ".join(fmt(runs[n]["summary"][key]) for n in runs) + " |")
    a10 = runs["a10"]["questions"]
    retried = [q for q in a10 if q.get("retried")]
    rescued = [q["id"] for q in retried if q["found"]]
    still = [q["id"] for q in retried if not q["found"] and q["category"] not in REFUSE]
    L += [
        "",
        f"- İkinci denemeye giren soru: **{len(retried)}** (55'ten). Kurtarılan: {', '.join(rescued) or 'yok'}.",
        f"- İkinci denemeden sonra hâlâ \"bulamadım\" diyen cevaplanabilir soru: {', '.join(still) or 'yok'}.",
        "- \"Bulamadım\" sorularının hepsi (gizli + belgelerde yok) iki denemeye de giriyor: maliyetin çoğu orada.",
        "- `hit@5` ikinci denemede, yeniden yazılmış aramanın kaynaklarıyla ölçülüyor (LLM'in son gördüğü).",
    ]
    return L


if __name__ == "__main__":
    main()
