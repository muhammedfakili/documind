"""Run the exam (eval/questions.jsonl) against the current system and write a report.

For every question:
  1. search the top 10 chunks (as the customer or the support agent, per question)
  2. the LLM answers from the top 5 (exactly what the product does)
  3. code grades the search, an LLM judge grades the answer

Outputs:
  eval/results/eval_<name>.md        explained report (Turkish)
  eval/results/runs/<name>.json      every question in detail, for comparing runs later

LLM calls are cached in eval/cache/, so a re-run only pays for prompts that changed.

Usage (from the project root, database running):
  uv run python -m eval.run_eval --name a10 --rewrite --compare rerank  # the current system (A10)
  uv run python -m eval.run_eval --name rerank                         # A9's search (with today's prompt)
  uv run python -m eval.run_eval --name hybrid --no-rerank             # A8's system
  uv run python -m eval.run_eval --name baseline --mode vector --no-rerank  # A7's system
  uv run python -m eval.run_eval --limit 5                     # quick try on 5 questions

Every run gets a before/after section against an earlier run: --compare (default: baseline).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import replace
from datetime import datetime

from app import store
from app.config import PROJECT_ROOT
from app.evaluation import JUDGE_SYSTEM, CachedLLM, Result, judge, load_questions, summarize
from app.generate import answer_from_hits
from app.llm import get_judge, get_llm
from app.retrieve import Mode, retrieve

RESULTS = PROJECT_ROOT / "eval" / "results"
PAUSE = 6  # seconds after each real LLM call: the Gemini free tier allows ~10 per minute
RETRIES = 4  # per question, on top of the provider's own retries

CATEGORIES = {
    "single": ("Tek belge", "Cevap tek bir yerde, düz metinde."),
    "table": ("Tablo / rakam", "Cevap bir tablo satırında ya da Excel hücresinde."),
    "multi": ("Çok belge", "Cevap için 2-3 farklı yeri birleştirmek gerekiyor."),
    "conflict": ("Tuzak", "Eski/yeni sürüm, benzer ürün (Pulse / Pulse Pro), kılavuz ve güncelleme çelişkisi."),
    "internal": ("İç belge (destek)", "Destek çalışanı soruyor, cevap iç belgede."),
    "hidden": ("İç belge (müşteri)", "Müşteri soruyor, cevap sadece iç belgede: \"bulamadım\" demeli."),
    "unanswerable": ("Belgelerde yok", "Cevap hiçbir belgede yok: \"bulamadım\" demeli."),
}
FAILURES = {
    "arama": "🔎 Arama: kanıt LLM'e verilen 5 parçaya girmedi",
    "cevap": "✍️ Cevap: kanıt vardı ama cevap yanlış/eksik",
    "yanlis-ret": "🙊 Yanlış ret: kanıt vardı ama \"bulamadım\" dedi",
    "uydurma": "🤥 Uydurma: belgelerde olmayan bir şeye cevap verdi",
}
MODE = {"public": "🙋", "all": "🧑‍💼"}


def run(limit: int | None, mode: Mode, rerank: bool, rewrite: bool = False) -> list[Result]:
    questions = load_questions()[:limit]
    llm = CachedLLM(get_llm(), pause=PAUSE)  # writes the answers, exactly like the product
    grader = CachedLLM(get_judge())  # the judge: another provider, its own quota
    conn = store.connect()
    results = []
    for i, q in enumerate(questions, start=1):
        started = time.perf_counter()
        first_hits = retrieve(conn, q.question, k=10, audience=q.audience, mode=mode, rerank=rerank)
        for attempt in range(1, RETRIES + 1):
            # Free-tier servers fail now and then (500/503) even after the provider's own
            # retries. Finished questions are in the cache, so waiting and retrying is cheap.
            try:
                hits = first_hits
                ans = answer_from_hits(q.question, hits[:5], llm)
                if rewrite and not ans.found:  # A10's second try, exactly like generate.answer()
                    hits = retrieve(conn, q.question, k=10, audience=q.audience, mode=mode, rerank=rerank,
                                    rewriter=llm)
                    ans = replace(answer_from_hits(q.question, hits[:5], llm), retried=True)
                verdict = judge(grader, q, ans) if ans.found else None
                break
            except Exception as exc:  # any API failure: wait, then retry
                if attempt == RETRIES:
                    raise
                print(f"  {q.id}: {type(exc).__name__}, retrying in {attempt * 60} s", flush=True)
                time.sleep(attempt * 60)
        results.append(Result(q, hits, ans, verdict, time.perf_counter() - started))
        r = results[-1]
        print(f"[{i}/{len(questions)}] {q.id} score={r.score} rank={r.rank} {ans.text[:70]!r}", flush=True)
    print(f"answer LLM: {llm.calls} new, {llm.cached} cached; judge: {grader.calls} new, {grader.cached} cached")
    return results


# --- report ------------------------------------------------------------------------


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def short(text: str, n: int = 120) -> str:
    one = " ".join(text.split()).replace("|", "/")
    return one if len(one) <= n else one[:n] + " …"


def score_mark(score: float) -> str:
    return {1.0: "✅", 0.5: "🟡"}.get(score, "❌")


def save_run(name: str, results: list[Result], summary: dict, model: str) -> None:
    runs = RESULTS / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    data = {
        "name": name,
        "model": model,
        "date": datetime.now().astimezone().isoformat(timespec="minutes"),
        "summary": summary,
        "questions": [
            {
                "id": r.question.id,
                "category": r.question.category,
                "score": r.score,
                "rank": r.rank,
                "recall5": r.recall5,
                "found": r.answer.found,
                "retried": r.answer.retried,
                "failure": r.failure,
                "answer": r.answer.text,
                "cited": [h.location for _, h in r.answer.cited_sources],
                "top5": [h.location for h in r.hits[:5]],
                "verdict": r.verdict,
            }
            for r in results
        ],
    }
    (runs / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def compare_with(name: str, results: list[Result], s: dict, other: str) -> list[str]:
    """Before/after table against runs/<other>.json -- the point of keeping every run."""
    path = RESULTS / "runs" / f"{other}.json"
    if name == other or not path.exists():
        return []
    base = json.loads(path.read_text(encoding="utf-8"))
    b = base["summary"]
    rows = [
        ("Genel puan", "score", pct),
        ("Cevaplanabilir sorular", "answer_score", pct),
        ("\"Bulamadım\" soruları", "refusal_score", pct),
        ("hit@5", "hit5", pct),
        ("recall@5", "recall5", pct),
        ("MRR@10", "mrr10", lambda x: f"{x:.2f}"),
        ("Sadakat", "faithful", pct),
        ("Token / soru", "input_tokens", lambda x: f"{x:.0f}"),
    ]
    L = [
        f"## ⚖️ `{other}` ile karşılaştırma",
        "",
        (f"> Aynı 55 soru, aynı cevap modeli ve hakem. `{other}`: {base['date']},"
         f" {base['model']}."),
        "",
        f"| Metrik | {other} | Bu çalışma | Fark |",
        "|---|---:|---:|---:|",
    ]
    for label, key, fmt in rows:
        diff = s[key] - b[key]
        sign = "+" if diff > 0 else ""
        shown = f"{sign}{diff * 100:.0f} puan" if fmt is pct else f"{sign}{fmt(diff)}"
        L.append(f"| {label} | {fmt(b[key])} | **{fmt(s[key])}** | {shown if diff else '='} |")
    before = {q["id"]: q for q in base["questions"]}
    changed = [r for r in results if r.question.id in before and r.score != before[r.question.id]["score"]]
    L += ["", f"**Puanı değişen sorular ({len(changed)}):**", ""]
    if changed:
        L += [f"| Soru | {other} | Şimdi | Kanıtın sırası (önce → şimdi) | Soru metni |", "|---|---:|---:|---|---|"]
        for r in changed:
            old = before[r.question.id]
            arrow = "📈" if r.score > old["score"] else "📉"
            L.append(
                f"| {arrow} {r.question.id} | {old['score']} | {r.score} | {old['rank'] or '—'} → {r.rank or '—'}"
                f" | {short(r.question.question, 80)} |"
            )
    else:
        L.append("Hiçbir sorunun puanı değişmedi.")
    return [*L, ""]


def report(name: str, results: list[Result], s: dict, model: str, compare: str = "baseline") -> list[str]:
    answerable = [r for r in results if r.question.should_answer]
    n_ans = len(answerable)
    n_ref = len(results) - n_ans
    by_cat: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_cat[r.question.category].append(r)
    failures = [r for r in results if r.failure]
    fail_count = {key: sum(1 for r in failures if r.failure == key) for key in FAILURES}

    L = [
        f"# Değerlendirme raporu: `{name}`",
        "",
        ("> **Bu rapor ne?** Sisteme cevaplarını önceden bildiğimiz "
        f"**{len(results)} soruluk bir sınav** yaptık. Her soruda iki ayrı not verdik:"),
        "> **arama notu** (doğru bilgi LLM'in önüne geldi mi? kod kontrol ediyor) ve",
        "> **cevap notu** (cevap doğru mu, kaynaklara sadık mı? ikinci bir LLM, \"hakem\", kontrol ediyor).",
        f"> Bu dosyayı `eval/run_eval.py` üretiyor ({model}, {datetime.now().astimezone():%Y-%m-%d %H:%M}).",
        ("> Ders anlatımı: `dersler/07-degerlendirme.md`. Soru-soru ayrıntı: `eval/results/runs/"
        f"{name}.json`."),
        "",
        "## ✅ Kısaca sonuç",
        "",
        (f"- **Genel puan: {pct(s['score'])}** ({len(results)} sorunun ortalaması; doğru = 1, kısmen = 0.5,"
        " yanlış = 0)."),
        f"- **Cevabı belgelerde olan {n_ans} soru: {pct(s['answer_score'])}.**",
        f"- **\"Bulamadım\" demesi gereken {n_ref} soru: {pct(s['refusal_score'])}.**",
        (f"- **Arama:** kanıt {pct(s['hit5'])} soruda ilk 5'te (hit@5). Çok belgeli sorular dahil, gereken"
        f" kanıtın {pct(s['recall5'])}'i LLM'e ulaştı (recall@5). MRR@10 = {s['mrr10']:.2f}."),
        (f"- **Sadakat:** Verilen cevapların {pct(s['faithful'])}'inde her cümle kaynaklarda var"
        " (hakemin kararı)."),
        "- **Hatalar nereden:** "
        + ", ".join(f"{FAILURES[k].split(':')[0]} {v}" for k, v in fail_count.items())
        + f" (toplam {len(failures)} soru tam puan alamadı).",
        (f"- **Maliyet:** soru başına ortalama {s['input_tokens']:.0f} token, {s['seconds']:.1f} sn"
        " (önbellekten gelenler hariç değil)."),
        "",
        *compare_with(name, results, s, compare),
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        ("| **Kanıt (evidence)** | Doğru cevabın geçtiği metin parçası. Her soru için elle yazdık"
        " (ör. `Price GBP: 169`). Kod, bulunan parçalarda bu metni arıyor. |"),
        "| **hit@5** | Kanıt içeren bir parça ilk 5 sonucun içinde mi? (LLM sadece ilk 5'i görüyor.) |",
        ("| **recall@5** | Çok belgeli sorularda gereken kanıtların yüzde kaçı ilk 5'te? 3 kanıttan 2'si"
        " geldiyse 0.67. |"),
        ("| **MRR@10** | İlk doğru parçanın sırasının tersi, ortalaması: 1. sıra = 1, 2. sıra = 0.5,"
        " 4. sıra = 0.25, ilk 10'da yok = 0. 1'e ne kadar yakınsa doğru parça o kadar üstte. |"),
        ("| **Hakem (LLM-as-judge)** | Cevabı notlandıran ikinci LLM çağrısı. Soruyu, bizim yazdığımız"
        " doğru cevabı, kaynakları ve sistemin cevabını görüyor. |"),
        ("| **Doğru / Kısmen / Yanlış** | Hakemin notu: anahtar bilgilerin hepsi / bir kısmı / hiçbiri"
        " (ya da çelişen bilgi). |"),
        ("| **Sadakat (faithfulness)** | Cevaptaki her iddia kaynaklarda yazıyor mu? Doğru ama kaynakta"
        " olmayan bilgi de sadakatsiz sayılır: o bilgi şans eseri doğru. |"),
        "| **Yanlış ret** | Kanıt LLM'in önündeydi ama \"bulamadım\" dedi. |",
        "",
        "## 1. Sınavın içeriği",
        "",
        "| Kategori | Soru | Ne ölçüyor | Örnek |",
        "|---|---:|---|---|",
    ]
    for cat, (label, what) in CATEGORIES.items():
        group = by_cat.get(cat, [])
        if group:
            L.append(f"| **{label}** | {len(group)} | {what} | {group[0].question.question} |")
    L += [
        "",
        "## 2. Kategoriye göre sonuçlar",
        "",
        "| Kategori | Soru | Arama hit@5 | MRR@10 | Cevap puanı | Sadakat |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cat, (label, _) in CATEGORIES.items():
        group = by_cat.get(cat, [])
        if not group:
            continue
        g = summarize(group)
        retrieval = (f"{pct(g['hit5'])}", f"{g['mrr10']:.2f}") if group[0].question.should_answer else ("—", "—")
        faithful = pct(g["faithful"]) if any(r.faithful is not None for r in group) else "—"
        L.append(f"| {label} | {len(group)} | {retrieval[0]} | {retrieval[1]} | **{pct(g['score'])}** | {faithful} |")
    L += [
        (f"| **Toplam** | **{len(results)}** | **{pct(s['hit5'])}** | **{s['mrr10']:.2f}** |"
        f" **{pct(s['score'])}** | **{pct(s['faithful'])}** |"),
        "",
        "Arama sütunları \"bulamadım\" kategorilerinde `—`: orada bulunacak bir kanıt yok.",
        "",
        "## 3. Tam puan alamayan sorular",
        "",
    ]
    if not failures:
        L += ["Hepsi tam puan aldı.", ""]
    else:
        L += [
            "| # | Kategori | Soru | Hata türü | Kanıt sırası | Puan | Hakemin gerekçesi |",
            "|---|---|---|---|---:|:-:|---|",
        ]
        for r in failures:
            q = r.question
            rank = r.rank if r.rank else "yok"
            reason = short(r.verdict["reason"], 140) if r.verdict else "Sistem \"bulamadım\" dedi."
            L.append(
                f"| {q.id} | {CATEGORIES[q.category][0]} | {MODE[q.audience]} {q.question} |"
                f" {FAILURES[r.failure]} | {rank} | {score_mark(r.score)} | {reason} |"
            )
        L += ["", "### Cevapların tam metni", ""]
        for r in failures:
            q = r.question
            L += [f"**{q.id} — {q.question}**", ""]
            L += [f"- Doğru cevap: {q.reference or '*(belgelerde yok, bulamadım demeli)*'}"]
            L += [f"- Sistemin cevabı: {short(r.answer.text, 600)}"]
            L += [f"- LLM'e verilen 5 parça: {', '.join(f'`{h.location}`' for h in r.hits[:5])}", ""]
    L += [
        "## 4. Bütün sorular",
        "",
        "| # | Soru | Kanıt sırası | Puan | Sadık mı |",
        "|---|---|---:|:-:|:-:|",
    ]
    for r in results:
        rank = "—" if not r.question.should_answer else (r.rank or "yok")
        faithful = "—" if r.faithful is None else ("✅" if r.faithful else "❌")
        L.append(f"| {r.question.id} | {MODE[r.question.audience]} {r.question.question} | {rank} |"
                 f" {score_mark(r.score)} | {faithful} |")
    L += [
        "",
        "## 5. Hakeme giden talimat",
        "",
        "```",
        JUDGE_SYSTEM.strip(),
        "```",
        "",
        ("Hakem LLM'e soru, bizim yazdığımız doğru cevap, sistemin gördüğü 5 kaynak ve sistemin cevabı"
        " gidiyor. \"Bulamadım\" diyen cevaplar hakeme gitmiyor, onları kod notluyor."),
        "",
        "## 🧪 Testler ne kontrol ediyor (`tests/test_evaluation.py`)",
        "",
        "| Test | Neyi garanti ediyor |",
        "|---|---|",
        ("| `test_question_file_is_valid` | Her soruda gerekli alanlar var, kategoriler doğru, kimlikler tekrar"
        " etmiyor, \"bulamadım\" sorularında kanıt yok. |"),
        ("| `test_every_evidence_exists_in_the_documents` | Elle yazdığımız her kanıt metni gerçekten bir"
        " parçada geçiyor (yazım hatası olsa sınav haksız olurdu). Müşteri sorularının kanıtı herkese"
        " açık belgede. |"),
        "| `test_rank_and_recall` | Kanıt sırası ve çok belgeli kanıt oranı doğru hesaplanıyor. |",
        ("| `test_judge_json_is_read_even_with_extra_text` | Hakemin cevabı kod bloğu içinde gelse de okunuyor,"
        " geçersiz not reddediliyor. |"),
        ("| `test_scores_and_failure_types` | Puanlama ve hata türü (arama / cevap / yanlış ret / uydurma)"
        " doğru. |"),
        "| `test_cache_answers_the_same_prompt_from_disk` | Aynı prompt ikinci kez LLM'e gitmiyor. |",
        "",
    ]
    return L


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--mode", choices=["vector", "keyword", "hybrid"], default="hybrid")
    parser.add_argument("--no-rerank", dest="rerank", action="store_false", help="skip the A9 reranker")
    parser.add_argument("--rewrite", action="store_true", help="A10: on \"not found\", rewrite the question and try once more")
    parser.add_argument("--compare", default="baseline", help="earlier run to compare with (runs/<name>.json)")
    args = parser.parse_args()

    results = run(args.limit, args.mode, args.rerank, args.rewrite)
    summary = summarize(results)
    search_desc = ("rewrite + " if args.rewrite else "") + args.mode + (" + rerank" if args.rerank else "")
    model = f"arama: {search_desc}, cevap: {get_llm().model}, hakem: {get_judge().model}"
    save_run(args.name, results, summary, model)
    out = RESULTS / f"eval_{args.name}.md"
    out.write_text("\n".join(report(args.name, results, summary, model, args.compare)), encoding="utf-8-sig")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {out.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
