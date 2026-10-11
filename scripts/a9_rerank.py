"""A9: re-sort the hybrid search results with a cross-encoder -- which model, how, how deep, how slow?

Outputs:
  eval/results/a9_rerank.md     explained report (Turkish)

Retrieval only, no LLM: the 46 answerable exam questions are searched with A8's
hybrid search (30 deep), then re-sorted by each reranker. Same evidence check as
A7/A8. The answer-level effect: eval/run_eval.py --name rerank --compare hybrid.

Experiments:
  1. four small and one large reranker side by side, each used two ways:
       alone  the reranker's order replaces the search order
       fused  the reranker's order and the search order are merged with RRF (A8)
  2. depth: how many hybrid results the reranker reads (10 / 20 / 30)
  3. per category and question by question for the chosen setup
  4. the E-214 journey: Pulse vs Pulse Pro
  5. speed

Usage (from the project root, database running; first run downloads ~1.5 GB of models):
  uv run python -m scripts.a9_rerank
"""

from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean

from app import store
from app.config import PROJECT_ROOT, settings
from app.evaluation import Question, evidence_recall, first_relevant_rank, load_questions
from app.rerank import _model, rerank
from app.retrieve import RERANK_CANDIDATES, retrieve, rrf_fuse
from app.store import Hit

REPORT = PROJECT_ROOT / "eval" / "results" / "a9_rerank.md"

# (model, size on disk, one-line description)
MODELS = [
    ("Xenova/ms-marco-MiniLM-L-6-v2", "80 MB", "6 katman, web arama verisiyle (MS MARCO) eğitilmiş"),
    ("Xenova/ms-marco-MiniLM-L-12-v2", "120 MB", "aynısının 12 katmanlısı"),
    ("jinaai/jina-reranker-v1-tiny-en", "130 MB", "Jina'nın en küçüğü"),
    ("jinaai/jina-reranker-v1-turbo-en", "150 MB", "Jina'nın hızlı orta boyu"),
    ("BAAI/bge-reranker-base", "1 GB", "embedding modelimizin (bge) ailesinden, en büyüğü"),
]
DEPTHS = (10, 20, 30)
BIG = "BAAI/bge-reranker-base"

CATEGORY_LABEL = {
    "single": "Tek belge",
    "table": "Tablo / rakam",
    "multi": "Çok belge",
    "conflict": "Tuzak",
    "internal": "İç belge (destek)",
}

E214 = ("What does error code E-214 mean on the Pulse earbuds?", "Battery temperature sensor fault")


def grade(questions: list[Question], ranked: dict[str, list[Hit]]) -> dict[str, float]:
    ranks = [first_relevant_rank(ranked[q.id][:10], q.evidence) for q in questions]
    return {
        "hit5": mean(r is not None and r <= 5 for r in ranks),
        "recall5": mean(evidence_recall(ranked[q.id][:5], q.evidence) for q in questions),
        "mrr10": mean(1 / r if r else 0.0 for r in ranks),
    }


def pct(x: float) -> str:
    return f"{x * 100:.0f}%"


def rank_str(r: int | None) -> str:
    return str(r) if r else "30+"


def short_name(model: str) -> str:
    return model.split("/")[-1]


def triple(r: dict) -> str:
    return f"{pct(r['hit5'])} / {pct(r['recall5'])} / {r['mrr10']:.3f}"


def timed_ms(name: str, questions: list[Question], candidates: dict[str, list[Hit]], depth: int) -> float:
    """Reranker time per question; the faster of two runs (a busy CPU only ever slows a run down)."""
    runs = []
    for _ in range(2):
        started = time.perf_counter()
        for q in questions:
            rerank(q.question, candidates[q.id][:depth], model=name)
        runs.append((time.perf_counter() - started) / len(questions) * 1000)
    return min(runs)


def main() -> None:
    conn = store.connect()
    questions = [q for q in load_questions() if q.should_answer]
    candidates = {q.id: retrieve(conn, q.question, k=30, audience=q.audience, rerank=False) for q in questions}
    base = grade(questions, candidates)

    # --- 1 + 2. every model at every depth, alone and fused ----------------------------
    # The reranker re-sorts the top `depth`; below that the search order stays, exactly
    # what retrieve() returns. "fused" is what the product does since A9.
    results: dict[tuple[str, int, str], dict] = {}
    ranked_by: dict[tuple[str, int, str], dict[str, list[Hit]]] = {}
    for name, _, _ in MODELS:
        _model(name)  # load (and download) once, before any timing
        for depth in DEPTHS:
            alone, fused = {}, {}
            for q in questions:
                top, rest = candidates[q.id][:depth], candidates[q.id][depth:]
                order = rerank(q.question, top, model=name)
                alone[q.id] = order + rest
                fused[q.id] = rrf_fuse([top, order]) + rest
            for kind, ranked in (("alone", alone), ("fused", fused)):
                results[(name, depth, kind)] = grade(questions, ranked)
                ranked_by[(name, depth, kind)] = ranked
            print(f"{short_name(name):32} d={depth} alone {triple(results[(name, depth, 'alone')])}"
                  f" | fused {triple(results[(name, depth, 'fused')])}", flush=True)

    # --- 5. speed, measured separately so loading/downloading never counts --------------
    speed = {name: timed_ms(name, questions, candidates, RERANK_CANDIDATES) for name, _, _ in MODELS}

    chosen = (settings.rerank_model, RERANK_CANDIDATES, "fused")
    best = results[chosen]
    best_alone = results[(settings.rerank_model, RERANK_CANDIDATES, "alone")]
    big_alone = results[(BIG, 20, "alone")]
    big_fused = results[(BIG, 20, "fused")]
    final = ranked_by[chosen]

    # --- 3. per category, question by question -------------------------------------------
    by_cat: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)
    per_cat = {c: (grade(qs, candidates), grade(qs, final)) for c, qs in by_cat.items()}
    changes = []
    for q in questions:
        before = first_relevant_rank(candidates[q.id], q.evidence)
        after = first_relevant_rank(final[q.id], q.evidence)
        rec_b = evidence_recall(candidates[q.id][:5], q.evidence)
        rec_a = evidence_recall(final[q.id][:5], q.evidence)
        if before != after or rec_b != rec_a:
            better = (rec_a, -(after or 99)) > (rec_b, -(before or 99))
            changes.append((q, before, after, rec_b, rec_a, better))
    won = sum(1 for c in changes if c[5])

    # --- 4. E-214 ---------------------------------------------------------------------
    question, needle = E214
    e214 = {}
    for label, hits in (
        ("Vektör (A5)", retrieve(conn, question, k=10, mode="vector", rerank=False)),
        ("Hybrid (A8)", retrieve(conn, question, k=10, rerank=False)),
        ("Hybrid + rerank (A9)", retrieve(conn, question, k=10)),
    ):
        e214[label] = (first_relevant_rank(hits, [[needle]]), hits[0].location, " ".join(hits[0].text.split())[:90])

    # --- report -------------------------------------------------------------------------
    m = short_name(chosen[0])
    L = [
        "# A9 — Reranking: soruyla parçayı birlikte okumak",
        "",
        (f"> **Bu rapor ne?** A8'in hybrid aramasının bulduğu adayları {len(MODELS)} farklı reranker modeliyle"
         f" yeniden sıraladık. Cevabı belgelerde olan **{len(questions)} soru**, A7'deki kanıt kontrolü."),
        "> Burada LLM yok, sadece arama ölçülüyor. Cevaplara etkisi: `eval/results/eval_rerank.md`.",
        "> Bu dosyayı `scripts/a9_rerank.py` üretiyor. Ders anlatımı: `dersler/09-reranking.md`.",
        "",
        "## ✅ Kısaca sonuç",
        "",
        (f"- **Seçilen: `{m}`, ilk {chosen[1]} adayı okuyor, sırası aramanın sırasıyla RRF ile birleşiyor.**"
         f" MRR {base['mrr10']:.3f} → **{best['mrr10']:.3f}**, recall@5 {pct(base['recall5'])} →"
         f" **{pct(best['recall5'])}**, hit@5 {pct(base['hit5'])} → {pct(best['hit5'])}."),
        ("- **Kazanç küçük.** A8'in hybrid'i zaten güçlüydü. Reranker'ın asıl katkısı **tablo satırlarında**:"
         " doğru satırı 4-5. sıradan 1. sıraya çıkarıyor."),
        (f"- **Reranker tek başına tehlikeli:** en büyük model (`bge-reranker-base`, 1 GB) 20 aday okuyup tek"
         f" başına karar verince hit@5 {pct(base['hit5'])} → **{pct(big_alone['hit5'])}**'e düştü (birleşikte"
         f" {pct(big_fused['hit5'])}). Büyük model ≠ iyi sonuç. Bu yüzden reranker'ın sırası aramanın yerine"
         " geçmiyor, onunla **oylanıyor**."),
        ("- **Çözemediği:** s16 (tamir) ve s17 (Japonya). Reranker da \"Japonya = diğer ülkeler\" çıkarımını"
         " yapamıyor. Bu soruların çaresi aramada değil (A10)."),
        f"- **Bedeli:** soru başına **{speed[settings.rerank_model]:.0f} ms** ek süre (CPU'da, GPU yok).",
        f"- **{len(changes)} sorunun sırası değişti:** {won} iyileşti, {len(changes) - won} kötüleşti.",
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        ("| **Bi-encoder** | Soru ve parça **ayrı ayrı** vektöre çevrilip karşılaştırılıyor (A4-A8'deki vektör"
         " araması). Parça vektörleri önceden hesaplandığı için hızlı. |"),
        ("| **Cross-encoder (reranker)** | Soru ve parça **birlikte**, tek girdi olarak okunuyor; model tek bir"
         " \"ne kadar alakalı\" puanı veriyor. Daha dikkatli ama her soru-parça çifti için ayrı hesap. |"),
        "| **Derinlik** | Reranker'ın okuduğu aday sayısı. Fazlası daha çok şans ama daha yavaş. |",
        "| **Tek başına / birleşik** | Reranker'ın sırası aramanın yerine mi geçiyor, yoksa onunla RRF ile mi oylanıyor. |",
        "| **MS MARCO** | Bing arama kayıtlarından yapılmış büyük bir eğitim seti. Çoğu reranker bununla eğitiliyor. |",
        "",
        "## 1. Modeller yan yana",
        "",
        (f"Hepsi ilk {RERANK_CANDIDATES} hybrid adayını yeniden sıralıyor. Hücrelerde: hit@5 / recall@5 / MRR@10."
         " İlk satır rerank'sız A8."),
        "",
        "| Model | Boyut | Tek başına | Birleşik | Soru başına | Not |",
        "|---|---:|---:|---:|---:|---|",
        f"| Rerank yok (A8 hybrid) | — | {triple(base)} | — | 0 ms | |",
    ]
    for name, size, note in MODELS:
        a = results[(name, RERANK_CANDIDATES, "alone")]
        f = results[(name, RERANK_CANDIDATES, "fused")]
        b = "**" if name == chosen[0] else ""
        L.append(f"| {b}{short_name(name)}{b} | {size} | {triple(a)} | {b}{triple(f)}{b} | {speed[name]:.0f} ms | {note} |")
    L += [
        "",
        ("👀 **Neye bak:** (1) MS MARCO ile eğitilmiş MiniLM modelleri, bizim kısa teknik metinlerde ve tablo"
         " satırlarında hybrid'in **gerisinde** kalıyor. 12 katmanlısı 6 katmanlısından bile kötü: büyük olan"
         " her zaman iyi değil, eğitildiği veri önemli. (2) Birleşik sütun neredeyse her modelde tek başınadan"
         " iyi ya da ona eşit: arama ve reranker farklı hatalar yapıyor, oylama ikisini dengeliyor."
         f" (3) Seçilen model tek başına da iyi ({triple(best_alone)}), birleşik daha iyi."
         " (4) MiniLM-L-12 birleşikte recall@5'te 1 soru önde: s16'yı 4. sıraya taşıyor. Ama tek başına en kötü"
         " model o; jina-tiny'nin üstünlüğü ise 4 soruya yayılıyor (s14, t02, t03, t08) ve iki kat hızlı."
         " 46 soruda tek sorunun farkı şans olabilir, o yüzden jina-tiny."),
        "",
        "## 2. Kaç aday okunsun?",
        "",
        "| Model | Derinlik | Tek başına | Birleşik |",
        "|---|---:|---:|---:|",
    ]
    for name, _, _ in MODELS:
        for depth in DEPTHS:
            mark = " ✔" if (name, depth, "fused") == chosen else ""
            L.append(
                f"| {short_name(name)} | {depth}{mark} | {triple(results[(name, depth, 'alone')])} |"
                f" {triple(results[(name, depth, 'fused')])} |"
            )
    L += [
        "",
        ("👀 **Neye bak:** Daha derin okumak neredeyse hiç kazandırmıyor, bazen kaybettiriyor: 20-30. sıradaki"
         " adaylar arasında reranker'ı kandıran \"benzer ama yanlış\" parçalar var. 10 aday hem yeterli hem"
         " en hızlısı: süre aday sayısıyla doğru orantılı artıyor (her aday bir model çalıştırması)."),
        "",
        f"## 3. Kategori kategori (`{m}`, derinlik {chosen[1]}, birleşik)",
        "",
        "| Kategori | Soru | Hybrid MRR / recall@5 | + Rerank MRR / recall@5 |",
        "|---|---:|---:|---:|",
    ]
    for cat, label in CATEGORY_LABEL.items():
        if cat in per_cat:
            b, a = per_cat[cat]
            L.append(
                f"| {label} | {len(by_cat[cat])} | {b['mrr10']:.2f} / {pct(b['recall5'])} |"
                f" **{a['mrr10']:.2f} / {pct(a['recall5'])}** |"
            )
    L += [
        "",
        "## 4. Soru soru",
        "",
        "Kanıtın sırası (30+ = ilk 30'da yok) ve ilk 5'e giren kanıt oranı:",
        "",
        "| | Soru | Hybrid | + Rerank | recall@5 | Soru metni |",
        "|---|---|---:|---:|---|---|",
    ]
    for q, before, after, rec_b, rec_a, better in changes:
        L.append(
            f"| {'📈' if better else '📉'} | {q.id} | {rank_str(before)} | **{rank_str(after)}** |"
            f" {rec_b:.2f} → {rec_a:.2f} | {q.question} |"
        )
    L += [
        "",
        ("👀 **Neye bak:** Kazançlar tablo (t) ve çok belgeli (m) sorularda: reranker \"hangi satır bu soruya"
         " cevap veriyor\" kararını aramadan iyi veriyor. s16 ve s17'de ise yardımcı olamıyor: \"Lumora does"
         " not repair units\" ve \"other countries\" cümlelerini tamir ve Japonya sorularıyla ilişkilendiremiyor."),
        "",
        "## 5. E-214 yolculuğu: Pulse mu, Pulse Pro mu?",
        "",
        f"Soru: *{question}* · Doğru satır: `{needle}` (Pulse kılavuzu).",
        "",
        "| Arama | Doğru satırın sırası | 1. sıradaki parça |",
        "|---|---:|---|",
        *[f"| {label} | **{rank_str(r)}** | `{loc}`: {text} |" for label, (r, loc, text) in e214.items()],
        "",
        ("👀 **Neye bak:** Reranker da Pulse ile Pulse Pro'yu ayıramadı. Pulse Pro'nun parçası \"Note: Pulse Pro uses its own"
         " error code list\" diye başlıyor ve E-214 satırını içeriyor; soruyla çok benzer. Doğru satır hybrid'in"
         " koyduğu 2. sırada kaldı. LLM ilk 5'in hepsini gördüğü ve kaynakları ürün adıyla okuduğu için A6'dan"
         " beri doğru cevabı veriyor; ama bu ayrımı aramaya yaptıramadık."),
        "",
        "## 🧪 Bu aşamanın testleri",
        "",
        "| Test | Neyi garanti ediyor |",
        "|---|---|",
        "| `test_best_scored_chunk_comes_first` | Reranker'ın en yüksek puan verdiği parça en üste çıkar. |",
        "| `test_score_becomes_the_reranker_score` | `Hit.score` artık RRF puanı değil, reranker puanı. |",
        "| `test_k_cuts_after_sorting` | Önce sıralanır, sonra kesilir: 3. sıradaki parça 1. olabilir. |",
        "| `test_ties_keep_the_search_order` | Eşit puanlıların sırasını aramanın sırası belirler. |",
        "| `test_question_and_every_chunk_go_to_the_model_together` | Model soruyu her parçayla birlikte görür. |",
        "| `test_no_hits_no_model_call` | Aday yoksa model hiç çalışmaz. |",
        "",
        ("Testler sahte bir modelle çalışıyor (gerçeği yüzlerce MB). Sıralama mantığını test ediyorlar; modelin"
         " kalitesini bu rapor ölçüyor."),
        "",
    ]
    REPORT.write_text("\n".join(L), encoding="utf-8-sig")
    print(f"Saved to {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
