"""A8: keyword search next to vector search, fused with RRF -- does it find more?

Outputs:
  eval/results/a8_hybrid.md     explained report (Turkish)

Retrieval only, no LLM: the 46 answerable exam questions (eval/questions.jsonl)
are searched five ways and graded with the same evidence check as A7.
The answer-level effect is measured separately by eval/run_eval.py --name hybrid.

Experiments:
  1. what Postgres does with a sentence (stems, stop words, E-214)
  2. five searches side by side: vector, keyword (ts_rank), keyword (BM25), two hybrids
  3. per category: where keyword search helps and where it hurts
  4. question by question: which ranks changed
  5. the RRF constant k: does tuning it matter?
  6. speed

Usage (from the project root, database running):
  uv run python -m scripts.a8_hybrid
"""

from __future__ import annotations

import time
from collections import defaultdict
from statistics import mean

from app import store
from app.config import PROJECT_ROOT
from app.embedding import embed_query
from app.evaluation import Question, evidence_recall, first_relevant_rank, load_questions
from app.retrieve import CANDIDATES, RRF_K, retrieve, rrf_fuse
from app.store import Hit

REPORT = PROJECT_ROOT / "eval" / "results" / "a8_hybrid.md"

# Postgres' built-in ranking, for comparison: same OR-ed words, but no word rarity.
TS_RANK_SQL = """
    WITH q AS (
        SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' | ')) AS tq
        FROM unnest(to_tsvector('english', %(q)s))
    )
    SELECT text, source, title, section, page, row_num, audience, effective_date, ts_rank_cd(tsv, q.tq)
    FROM chunks, q
    WHERE (%(all)s OR audience = 'public') AND tsv @@ q.tq
    ORDER BY 9 DESC, id
    LIMIT %(k)s
"""

TOKENIZE = [
    "What does error E-214 mean on my Pulse earbuds?",
    "How long does the battery last when charging over Wi-Fi?",
    "Can Lumora repair my broken headphones?",
]

CATEGORY_LABEL = {
    "single": "Tek belge",
    "table": "Tablo / rakam",
    "multi": "Çok belge",
    "conflict": "Tuzak",
    "internal": "İç belge (destek)",
}

METHODS = {
    "vector": "Vektör (A7)",
    "ts_rank": "Kelime, ts_rank",
    "bm25": "Kelime, BM25",
    "hybrid_ts": "Hybrid (vektör + ts_rank)",
    "hybrid": "**Hybrid (vektör + BM25)**",
}


def ts_rank_search(conn, question: str, k: int, audience: str) -> list[Hit]:
    rows = conn.execute(TS_RANK_SQL, {"q": question, "k": k, "all": audience == "all"}).fetchall()
    return [Hit(*row[:8], score=float(row[8])) for row in rows]


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


def main() -> None:
    conn = store.connect()
    store.create_table(conn)  # adds the tsv column + GIN index to an A5-era table
    questions = [q for q in load_questions() if q.should_answer]

    # --- 1. tokenizer -------------------------------------------------------------
    stems = [
        (s, conn.execute("SELECT array_agg(lexeme) FROM unnest(to_tsvector('english', %s))", [s]).fetchone()[0])
        for s in TOKENIZE
    ]

    # --- 2. five searches, 30 deep each ---------------------------------------------
    runs: dict[str, dict[str, list[Hit]]] = defaultdict(dict)
    for q in questions:
        vec = store.search(conn, embed_query(q.question), k=CANDIDATES, audience=q.audience)
        tsr = ts_rank_search(conn, q.question, CANDIDATES, q.audience)
        bm = store.keyword_search(conn, q.question, k=CANDIDATES, audience=q.audience)
        runs["vector"][q.id] = vec
        runs["ts_rank"][q.id] = tsr
        runs["bm25"][q.id] = bm
        runs["hybrid_ts"][q.id] = rrf_fuse([vec, tsr])
        runs["hybrid"][q.id] = rrf_fuse([vec, bm])
    overall = {m: grade(questions, runs[m]) for m in METHODS}

    # --- 3. per category --------------------------------------------------------------
    by_cat: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_cat[q.category].append(q)
    per_cat = {c: {m: grade(qs, runs[m]) for m in ("vector", "bm25", "hybrid")} for c, qs in by_cat.items()}

    # --- 4. question by question ------------------------------------------------------
    changes = []
    for q in questions:
        before = first_relevant_rank(runs["vector"][q.id], q.evidence)
        after = first_relevant_rank(runs["hybrid"][q.id], q.evidence)
        kw = first_relevant_rank(runs["bm25"][q.id], q.evidence)
        rec_b = evidence_recall(runs["vector"][q.id][:5], q.evidence)
        rec_a = evidence_recall(runs["hybrid"][q.id][:5], q.evidence)
        if before != after or rec_b != rec_a:
            better = (rec_a, -(after or 99)) > (rec_b, -(before or 99))
            changes.append((q, before, kw, after, rec_b, rec_a, better))

    # --- 5. RRF constant -----------------------------------------------------------------
    rrf_ks = {
        k: grade(questions, {q.id: rrf_fuse([runs["vector"][q.id], runs["bm25"][q.id]], k=k) for q in questions})
        for k in (1, 10, 30, 60, 100)
    }

    # --- 6. speed ----------------------------------------------------------------------
    def timed(mode: str) -> float:
        started = time.perf_counter()
        for q in questions:
            retrieve(conn, q.question, k=5, audience=q.audience, mode=mode, rerank=False)
        return (time.perf_counter() - started) / len(questions) * 1000

    timed("hybrid")  # warm-up: model loaded, caches filled
    speed = {m: timed(m) for m in ("vector", "keyword", "hybrid")}

    # --- report -------------------------------------------------------------------------
    v, h, bm, ts = overall["vector"], overall["hybrid"], overall["bm25"], overall["ts_rank"]
    won = sum(1 for c in changes if c[6])
    lost = len(changes) - won
    L = [
        "# A8 — Hybrid search: anlam + kelime",
        "",
        ("> **Bu rapor ne?** A7'deki sınavın cevabı belgelerde olan "
         f"**{len(questions)} sorusunu** beş farklı aramayla aradık ve aynı kanıt kontrolüyle notladık."),
        "> Burada LLM yok, sadece arama ölçülüyor. Cevaplara etkisi: `eval/results/eval_hybrid.md`.",
        "> Bu dosyayı `scripts/a8_hybrid.py` üretiyor. Ders anlatımı: `dersler/08-hybrid-search.md`.",
        "",
        "## ✅ Kısaca sonuç",
        "",
        (f"- **Hybrid (vektör + BM25) aramayı iyileştirdi:** hit@5 {pct(v['hit5'])} → **{pct(h['hit5'])}**,"
         f" recall@5 {pct(v['recall5'])} → **{pct(h['recall5'])}**, MRR {v['mrr10']:.2f} → **{h['mrr10']:.2f}**."),
        (f"- **Kelime puanlaması çok önemli:** Postgres'in hazır `ts_rank`'ı tek başına hit@5 {pct(ts['hit5'])},"
         f" BM25 {pct(bm['hit5'])}. `ts_rank`'lı hybrid vektörden bile kötü"
         f" ({pct(overall['hybrid_ts']['hit5'])}). Fark: BM25 kelimenin ne kadar **nadir** olduğunu biliyor."),
        (f"- **{len(changes)} sorunun sırası değişti:** {won} iyileşti (s16 ve s17 yaklaştı ama ilk 5'e giremedi),"
         f" {lost} kötüleşti. Hybrid her soruda"
         " kazanmıyor; toplamda kazanıyor."),
        ("- **Çözülenler:** m05 ve m06'da eksik kanıt tamamlandı (A7'de LLM bu yüzden \"bulamadım\" demişti),"
         " t08'in kanıtı 10. sıradan 5'e çıktı."),
        ("- **Çözülemeyenler:** s16, s17 ve m01. Sebebi kelime uyuşmazlığı: soruda \"Japan\" var, belgede"
         " \"other countries\" yazıyor. Bunu A9 (reranking) ve A10 (soruyu yeniden yazma) çözecek."),
        (f"- **Maliyet:** soru başına arama {speed['vector']:.0f} ms → {speed['hybrid']:.0f} ms."
         " Yeni bir servis yok, aynı Postgres."),
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        ("| **Vektör araması** | A4-A5'ten beri yaptığımız: anlamca yakın parçalar. \"Too many devices\" ile"
         " \"pairing memory full\" birbirini bulur. |"),
        ("| **Kelime araması (full-text)** | Aynı kelimeleri içeren parçalar. `E-214`, `LA-PL-200`, `£9.99` gibi"
         " birebir ifadeleri kaçırmaz. |"),
        "| **Kök (stem / lexeme)** | Kelimenin eki atılmış hali: `earbuds` → `earbud`, `charging` → `charg`. |",
        "| **Durak kelime (stop word)** | Anlam taşımayan sık kelimeler (`the`, `does`, `my`). Atılıyor. |",
        "| **tsvector** | Postgres'in bir metni kök listesine çevirmiş hali. Her parça için ayrı sütunda tutuluyor. |",
        "| **GIN indeksi** | Kitabın arkasındaki dizin: her kök için içinde geçtiği satırların listesi. |",
        ("| **IDF (nadirlik)** | Bir kelime ne kadar az parçada geçiyorsa o kadar değerli. `pulse` 157 parçanın 56'sında,"
         " `-214` sadece 7'sinde: `-214` eşleşmesi çok daha anlamlı. |"),
        ("| **BM25** | Arama motorlarının standart puanı: nadirlik × kelimenin parçada kaç kez geçtiği (doyumlu)"
         " ÷ parça uzunluğu. |"),
        ("| **RRF** | İki sıralı listeyi birleştirme: her liste her parçaya `1 / (60 + sıra)` puan veriyor,"
         " puanlar toplanıyor. Ham skorlara bakmıyor, sadece sıraya. |"),
        "",
        "## 1. Postgres bir cümleyi nasıl görüyor?",
        "",
        "| Cümle | Postgres'in sakladığı kökler |",
        "|---|---|",
        *[f"| {s} | `{' '.join(sorted(lex))}` |" for s, lex in stems],
        "",
        ("👀 **Neye bak:** `what`, `does`, `my`, `on` gitti. `earbuds` → `earbud`, `battery` → `batteri`."
         " `E-214` iki parçaya bölündü: `e` ve `-214`. Önemli olan `-214`, çünkü çok nadir."
         " Kelime araması `charging` arayan kişiye `charges` geçen parçayı da bulur (ikisi de `charg`)."),
        "",
        "## 2. Beş arama yan yana",
        "",
        f"Her arama 30 sonuç getirdi, hybrid ikisini RRF (k = {RRF_K}) ile birleştirdi. {len(questions)} soru.",
        "",
        "| Arama | hit@5 | recall@5 | MRR@10 |",
        "|---|---:|---:|---:|",
        *[
            f"| {label} | {pct(overall[m]['hit5'])} | {pct(overall[m]['recall5'])} | {overall[m]['mrr10']:.2f} |"
            for m, label in METHODS.items()
        ],
        "",
        ("👀 **Neye bak:** Kelime araması tek başına vektörden kötü (MRR düşük): anlamı bilmiyor. Ama"
         " vektörün kaçırdığını bulduğu için **birleşince** ikisinden de iyi. `ts_rank` ise birleşince"
         " **zarar** veriyor: soruda `pulse` geçince Pulse kelimesi geçen her parçaya puan veriyor, çöp"
         " sonuçlar RRF'de iyi parçaları aşağı itiyor."),
        "",
        "## 3. Kategori kategori",
        "",
        "| Kategori | Soru | Vektör MRR / recall@5 | BM25 MRR / recall@5 | Hybrid MRR / recall@5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for cat, label in CATEGORY_LABEL.items():
        if cat in per_cat:
            g = per_cat[cat]
            cells = " | ".join(f"{g[m]['mrr10']:.2f} / {pct(g[m]['recall5'])}" for m in ("vector", "bm25", "hybrid"))
            L.append(f"| {label} | {len(by_cat[cat])} | {cells} |")
    L += [
        "",
        ("👀 **Neye bak:** En büyük kazanç **tablo** sorularında recall'da ve **tuzak** sorularında (ürün"
         " kodu, hata kodu). Tablo sorularının MRR'ı hâlâ düşük: kanıt bulunuyor ama 1. sırada değil."
         " Hangi satırın soruya cevap verdiğine karar vermek okumayı gerektiriyor, A9'daki reranker'ın işi."),
        "",
        "## 4. Soru soru: kim kazandı, kim kaybetti",
        "",
        "Kanıtın sırası (30+ = ilk 30'da yok) ve ilk 5'e giren kanıt oranı:",
        "",
        "| | Soru | Vektör | BM25 | Hybrid | recall@5 | Soru metni |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for q, before, kw, after, rec_b, rec_a, better in changes:
        mark = "📈" if better else "📉"
        L.append(
            f"| {mark} | {q.id} | {rank_str(before)} | {rank_str(kw)} | **{rank_str(after)}** |"
            f" {rec_b:.2f} → {rec_a:.2f} | {q.question} |"
        )
    L += [
        "",
        ("👀 **Neye bak:** **s16**: BM25 kanıtı 3. sıraya koyuyor, vektör 20.'ye. RRF ortalamayı alınca 6."
         " sırada kalıyor. İki aramadan biri çok yanılırsa RRF onu tam kurtaramıyor. **s17**: belgede `Japan`"
         " geçmiyor (`We do not currently ship to other countries`), kelime araması da çaresiz. **m01**:"
         " soruda `cheaper`, tabloda `Price` yazıyor; fiyat satırları gürültü kelimelerin altında kalıyor."),
        "",
        "## 5. RRF sabiti (k) önemli mi?",
        "",
        "| k | hit@5 | recall@5 | MRR@10 |",
        "|---:|---:|---:|---:|",
        *[
            f"| {k}{' (seçilen)' if k == RRF_K else ''} | {pct(g['hit5'])} | {pct(g['recall5'])} | {g['mrr10']:.2f} |"
            for k, g in rrf_ks.items()
        ],
        "",
        ("👀 **Neye bak:** k büyüdükçe sıralar arasındaki puan farkı küçülüyor (1. ile 5. sıra neredeyse eşit"
         " sayılıyor). Bizim setimizde k = 10 bir soru daha kazandırıyor, ama **60'ta kaldık**:"
         " 46 soruda 1 soruluk fark şans olabilir. Sınav sorularına göre ayar yapmak, sınavı ezberlemek"
         " demek (overfitting). 60, RRF makalesinin değeri ve geniş bir aralıkta iyi çalışıyor."),
        "",
        "## 6. Hız",
        "",
        "| Arama | Soru başına |",
        "|---|---:|",
        *[f"| {m} | {ms:.1f} ms |" for m, ms in speed.items()],
        "",
        ("👀 **Neye bak:** Süreyi çoğunlukla sorunun vektöre çevrilmesi alıyor. Kelime araması GIN indeksiyle"
         " milisaniyeler sürüyor. Cevap üreten LLM saniyeler sürdüğü için bu fark kullanıcıya hissettirmez."),
        "",
        "## 🧪 Bu aşamanın testleri",
        "",
        "| Test | Neyi garanti ediyor |",
        "|---|---|",
        "| `test_keyword_finds_the_exact_code` | `E-214` arayan, `E-214` geçen parçayı bulur. |",
        "| `test_rare_word_beats_common_word` | Her parçada geçen `pulse`, nadir `-214`'ün önüne geçemez (IDF). |",
        "| `test_stemming_matches_other_word_forms` | `charging` arayınca `charges` geçen parça gelir. |",
        "| `test_one_shared_word_is_enough` | Tek ortak kelime yeter (OR), bütün kelimeler şart değil. |",
        "| `test_keyword_search_respects_audience` | Kelime araması da müşteriye iç belge göstermez. |",
        "| `test_question_of_only_stop_words_returns_nothing` | Anlamsız soru hata vermez, boş döner. |",
        "| `test_chunk_high_in_both_lists_wins` | İki listede de üstte olan parça, tek listede 1. olanı geçer. |",
        "| `test_found_by_one_search_still_kept` | Sadece bir aramanın bulduğu parça kaybolmaz. |",
        "| `test_raw_scores_are_ignored_only_ranks_count` | BM25'in 50 puanı, kosinüsün 0.9'unu ezemez. |",
        "| `test_score_is_the_fused_points` | Aynı parça iki listede varsa tek kez döner, puanı toplanır. |",
        "",
    ]
    REPORT.write_text("\n".join(L), encoding="utf-8-sig")
    print(f"vector {v}\nhybrid {h}\nchanged {len(changes)} (won {won}, lost {lost})\nspeed {speed}")
    print(f"Saved to {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
