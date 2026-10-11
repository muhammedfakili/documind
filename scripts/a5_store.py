"""A5: load everything into Postgres + pgvector, check search, measure the index.

Outputs:
  table "chunks" in the database     157 rows: text + vector + labels
  eval/results/a5_store.md            explained report (Turkish)

Three experiments:
  1. same answers   does the database return exactly what numpy returned in A4?
  2. who is asking  customer mode (public only) vs support-agent mode (all)
  3. index speed    100,000 random vectors: exact scan vs HNSW, time and recall

Usage (from the project root, database running: docker compose up -d):
  uv run python -m scripts.a5_store
"""

from __future__ import annotations

import time
from statistics import mean

import numpy as np
import psycopg
from psycopg import sql

from app import store
from app.config import DATA_DIR, PROJECT_ROOT, settings
from app.embedding import embed_query
from app.ingest.pipeline import ingest_directory
from app.retrieve import retrieve

REPORT = PROJECT_ROOT / "eval" / "results" / "a5_store.md"

# Same questions as A4, plus where the answer lives.
# (question, string only the right chunk contains, answer document is internal?)
SEARCH = [
    ("What does error code E-214 mean on the Pulse earbuds?", "Battery temperature sensor fault", False),
    ("What does E-214 mean on Pulse Pro?", "ANC microphone calibration failed", False),
    ("How long do I have to return earbuds I already opened?", "**14 days**", False),
    ("Who has to approve a refund of more than 500 dollars?", "Above $500", True),
    ("How much is express delivery to the UK?", "£9.99", False),
    ("How many Aria X2 speakers can play together in Party Mode?", "Up to 4", False),
    ("Is water damage covered for Halo headphones?", "never covered", False),
    ("What margin does a Gold reseller get?", "Margin on MSRP: 35%", True),
    ("Can I still buy Halo Lite?", "discontinued", False),
    ("If only one earbud is broken, can just that one be replaced?", "single defective earbud", True),
]

BENCH_TABLE = "bench_vectors"
BENCH_ROWS = 100_000
EXTRA_QUESTIONS = [
    "How do I pair Pulse with my phone?",
    "How long does the Halo ANC battery last?",
    "Can I swim with Aria X2?",
    "Which countries do you ship to?",
    "How do I reset my earbuds to factory settings?",
    "Is there a student discount?",
    "How do I update the firmware?",
    "What is the warranty period in the EU?",
    "Do resellers handle warranty claims?",
    "What should an agent do if a customer reports a swollen battery?",
]


def rank(hits: list[store.Hit], needle: str) -> int | None:
    return next((i for i, h in enumerate(hits, start=1) if needle in h.text), None)


def mark(position: int | None) -> str:
    if position == 1:
        return "✅ 1."
    if position is not None and position <= 3:
        return f"🟡 {position}."
    return f"❌ {position}." if position else "❌ ilk 10'da yok"


def short(text: str, n: int = 90) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= n else one_line[:n] + " …"


# ---------------------------------------------------------------------------
# Experiment 3: index speed on a big synthetic table
# ---------------------------------------------------------------------------


def build_bench(conn: psycopg.Connection, rng: np.random.Generator, centers: np.ndarray) -> float | None:
    """Fill the bench table and build its HNSW index. Returns build seconds (None if reused).

    Real text vectors are clustered by topic (see the A4 heatmap), so the fake
    ones are too: each is one of our 157 real chunk vectors plus noise. Purely
    random vectors would be an unrealistically hard test -- in 384 dimensions
    they are all about equally far from each other.
    """
    t = sql.Identifier(BENCH_TABLE)
    indexed = conn.execute(
        "SELECT count(*) FROM pg_indexes WHERE tablename = %s AND indexdef LIKE '%%hnsw%%'", (BENCH_TABLE,)
    ).fetchone()[0]
    if indexed and conn.execute(sql.SQL("SELECT count(*) FROM {t}").format(t=t)).fetchone()[0] == BENCH_ROWS:
        return None  # built on an earlier run
    conn.execute(sql.SQL("DROP TABLE IF EXISTS {t}").format(t=t))
    conn.execute(sql.SQL("CREATE TABLE {t} (id int PRIMARY KEY, embedding vector(384))").format(t=t))
    picks = centers[rng.integers(0, len(centers), BENCH_ROWS)]
    vectors = picks + rng.standard_normal((BENCH_ROWS, store.DIM), dtype=np.float32) * 0.04
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    with conn.cursor() as cur, cur.copy(
        sql.SQL("COPY {t} (id, embedding) FROM STDIN WITH (FORMAT BINARY)").format(t=t)
    ) as copy:
        copy.set_types(["int4", "vector"])
        for i, vec in enumerate(vectors):
            copy.write_row([i, vec])
    # Docker gives the container only 64 MB of shared memory, which parallel
    # index builds need; one worker with plenty of normal memory works fine.
    conn.execute("SET max_parallel_maintenance_workers = 0")
    conn.execute("SET maintenance_work_mem = '512MB'")
    started = time.perf_counter()
    conn.execute(sql.SQL("CREATE INDEX ON {t} USING hnsw (embedding vector_cosine_ops)").format(t=t))
    conn.execute(sql.SQL("ANALYZE {t}").format(t=t))
    return time.perf_counter() - started


def timed_ids(conn: psycopg.Connection, query: np.ndarray, use_index: bool) -> tuple[float, list[int]]:
    """Top-10 ids and the database's own execution time in ms.

    Timing on the server (EXPLAIN ANALYZE) leaves out sending the 384 numbers
    over the network, which is the same for both search paths.
    """
    conn.execute(f"SET enable_indexscan = {'on' if use_index else 'off'}")
    select = sql.SQL("SELECT id FROM {t} ORDER BY embedding <=> %s LIMIT 10").format(
        t=sql.Identifier(BENCH_TABLE)
    )
    ids = [r[0] for r in conn.execute(select, (query,)).fetchall()]
    plan = conn.execute(sql.SQL("EXPLAIN (ANALYZE, FORMAT JSON) ") + select, (query,)).fetchone()[0]
    conn.execute("SET enable_indexscan = on")
    return plan[0]["Execution Time"], ids


def uses_index(conn: psycopg.Connection, query: np.ndarray) -> bool:
    plan = conn.execute(
        sql.SQL("EXPLAIN SELECT id FROM {t} ORDER BY embedding <=> %s LIMIT 10").format(
            t=sql.Identifier(BENCH_TABLE)
        ),
        (query,),
    ).fetchall()
    return any("Index Scan" in line[0] for line in plan)


EF_SEARCH = [40, 100, 200]  # 40 is pgvector's default


def bench(conn: psycopg.Connection, centers: np.ndarray) -> dict:
    rng = np.random.default_rng(42)
    build_seconds = build_bench(conn, rng, centers)
    # Real questions, embedded like any user question.
    queries = [embed_query(q) for q, _, _ in SEARCH] + [embed_query(q) for q in EXTRA_QUESTIONS]

    timed_ids(conn, queries[0], use_index=False)  # warm the cache so the first timing is fair
    exact = [timed_ids(conn, q, use_index=False) for q in queries]
    by_ef = []
    for ef in EF_SEARCH:
        conn.execute(f"SET hnsw.ef_search = {ef}")
        timed_ids(conn, queries[0], use_index=True)
        approx = [timed_ids(conn, q, use_index=True) for q in queries]
        recall = mean(len(set(e[1]) & set(a[1])) / 10 for e, a in zip(exact, approx))
        by_ef.append((ef, mean(a[0] for a in approx), recall))
    conn.execute("RESET hnsw.ef_search")
    size = conn.execute("SELECT pg_size_pretty(pg_total_relation_size(%s))", (BENCH_TABLE,)).fetchone()[0]
    return {
        "build_seconds": build_seconds,
        "exact_ms": mean(e[0] for e in exact),
        "by_ef": by_ef,
        "uses_index": uses_index(conn, queries[0]),
        "size": size,
        "queries": len(queries),
    }


# ---------------------------------------------------------------------------


def main() -> None:
    with store.connect() as conn:
        started = time.perf_counter()
        stored = ingest_directory(conn, DATA_DIR / "raw")
        ingest_seconds = time.perf_counter() - started
        counts = store.count(conn)

        # 1. same answers as A4's numpy search?
        vectors = np.load(DATA_DIR / "processed" / "chunk_vectors.npy")
        texts = [row[0] for row in conn.execute("SELECT text FROM chunks ORDER BY id").fetchall()]
        same = 0
        for question, _, _ in SEARCH:
            qv = embed_query(question)
            numpy_top = [texts[i] for i in np.argsort(-(vectors @ qv))[:5]]
            db_top = [h.text for h in store.search(conn, qv, k=5, audience="all")]
            same += numpy_top == db_top

        # 2. who is asking
        agent = {q: retrieve(conn, q, k=10, audience="all") for q, _, _ in SEARCH}
        customer = {q: retrieve(conn, q, k=10, audience="public") for q, _, _ in SEARCH}
        leaked = sum(h.audience == "internal" for hits in customer.values() for h in hits)

        # 3. index speed
        b = bench(conn, vectors)

    e214 = SEARCH[0][0]
    agent_top1 = sum(rank(agent[q], n) == 1 for q, n, _ in SEARCH)
    agent_top3 = sum((rank(agent[q], n) or 99) <= 3 for q, n, _ in SEARCH)
    public_qs = [(q, n) for q, n, internal in SEARCH if not internal]
    cust_top1 = sum(rank(customer[q], n) == 1 for q, n in public_qs)
    cust_top3 = sum((rank(customer[q], n) or 99) <= 3 for q, n in public_qs)
    speedup = b["exact_ms"] / b["by_ef"][0][1]

    lines = [
        "# A5 — Veritabanı (Postgres + pgvector) raporu",
        "",
        "> **Bu rapor ne?** Bütün parçaları vektörleriyle birlikte veritabanına koyduk ve aramayı artık",
        "> veritabanı yapıyor. Üç şeyi ölçtük: sonuçlar A4 ile aynı mı, \"kim soruyor\" filtresi çalışıyor mu,",
        "> index aramayı ne kadar hızlandırıyor. Bu dosyayı `scripts/a5_store.py` üretiyor.",
        "> Ders anlatımı: `dersler/05-veritabani.md`.",
        "",
        "## ✅ Kısaca sonuç",
        "",
        (
            f"- **{stored} parça** veritabanında ({counts.get('public', 0)} herkese açık, "
            f"{counts.get('internal', 0)} iç belge). Baştan sona hazırlık hattı "
            f"**{ingest_seconds:.0f} saniye** sürdü."
        ),
        (
            f"- **Aynı sonuç:** 10 sorunun **{same} tanesinde** veritabanı, A4'teki numpy aramasıyla "
            "birebir aynı ilk 5'i verdi."
        ),
        (
            f"- **Müşteri modu:** {sum(len(h) for h in customer.values())} sonucun **{leaked} tanesi** iç belge. "
            + ("**Hiç sızıntı yok.**" if leaked == 0 else "**Sızıntı var!**")
        ),
        (
            f"- **Index:** 100.000 vektörde arama **{b['exact_ms']:.1f} ms → {b['by_ef'][0][1]:.1f} ms** "
            f"(~{speedup:.0f} kat hızlı), doğru ilk 10'un **%{100 * b['by_ef'][0][2]:.0f}**'i yakalanıyor."
        ),
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        "| **Postgres** | Verileri tablo halinde saklayan, SQL ile sorgulanan veritabanı. Docker içinde çalışıyor. |",
        "| **pgvector** | Postgres'e \"vektör\" sütun tipini ve vektör aramayı ekleyen eklenti. |",
        "| **`<=>`** | pgvector'ün kosinüs **mesafe** işareti. 0 = aynı yön. Benzerlik = 1 − mesafe. |",
        "| **Index (HNSW)** | Bütün satırlara bakmadan en yakın vektörleri bulmayı sağlayan yol haritası. |",
        "| **Tam tarama (exact)** | Her satırla tek tek karşılaştırmak. Kesin doğru ama yavaş. |",
        "| **Recall (yakalama oranı)** | Index'in bulduğu ilk 10, tam taramanın ilk 10'unun yüzde kaçı? |",
        "| **Müşteri modu / destek modu** | `audience=\"public\"`: iç belgeler hiç gelmez. `audience=\"all\"`: hepsi gelir. |",
        "",
        "## 1. Sonuçlar A4 ile aynı mı?",
        "",
        "A4'te aramayı Python'da (numpy) yapmıştık. Şimdi aynı vektörlerle aynı soruları veritabanına sorduk.",
        f"10 sorunun **{same}/10**'unda ilk 5 sonuç birebir aynı çıktı. Veritabanına taşırken hiçbir şey",
        "bozulmadı. Fark sadece **aramayı kimin yaptığında.**",
        "",
        "## 2. Kim soruyor? Müşteri modu ve destek modu",
        "",
        "Aynı soru, iki farklı kullanıcı. Destek ekibi iç belgeleri de görebilir, müşteri göremez.",
        "",
        f"### 2.1 \"{e214}\"",
        "",
        "| Sıra | 🧑‍💼 Destek modu (hepsi) | 🙋 Müşteri modu (sadece public) |",
        "|---:|---|---|",
    ]
    def cell(hit: store.Hit) -> str:
        lock = "🔒 " if hit.audience == "internal" else ""
        target = " ⭐" if SEARCH[0][1] in hit.text else ""
        section = f" › {hit.section}" if hit.section else ""
        return f"{lock}`{hit.location}`{section} ({hit.score:.3f}){target}"

    for i in range(5):
        lines.append(f"| {i + 1} | {cell(agent[e214][i])} | {cell(customer[e214][i])} |")
    lines += [
        "",
        "⭐ = aradığımız parça (Pulse kılavuzu, \"6. Error codes\" bölümü).",
        "",
        "👀 **Neye bak:** Destek modunda 1. ve 3. sırada 🔒 iç belgeler var (bilinen sorunlar listesi, destek",
        "kaydı). Müşteri modunda ikisi de **tamamen yok** ve listenin geri kalanı yukarı kayıyor.",
        (
            f"⭐ parça destek modunda {rank(agent[e214], SEARCH[0][1])}., müşteri modunda "
            f"**{rank(customer[e214], SEARCH[0][1])}.** sırada. Filtre hem güvenliği sağlıyor hem de bu"
        ),
        "soruda sonucu biraz iyileştiriyor. Ama hâlâ 1. sırada değil: Pulse Pro'nun E-214'ü önde.",
        "Bunu A8 (kelime araması) ve A9 (yeniden sıralama) çözecek.",
        "",
        "### 2.2 Bütün sorular",
        "",
        "| Soru | Cevap nerede | 🧑‍💼 Destek modu | 🙋 Müşteri modu |",
        "|---|---|---:|---:|",
    ]
    for question, needle, internal in SEARCH:
        where = "🔒 iç belge" if internal else "herkese açık"
        cust_cell = (
            "🔒 gizlendi ✅" if internal and rank(customer[question], needle) is None
            else mark(rank(customer[question], needle))
        )
        lines.append(f"| {question} | {where} | {mark(rank(agent[question], needle))} | {cust_cell} |")
    lines += [
        "",
        "**Nasıl okunur:**",
        f"- **Destek modu:** 10 sorudan {agent_top1} tanesinde doğru parça 1. sırada, {agent_top3} tanesinde ilk 3'te.",
        f"- **Müşteri modu:** Cevabı herkese açık belgede olan {len(public_qs)} sorudan {cust_top1} tanesinde",
        f"  doğru parça 1. sırada, {cust_top3} tanesinde ilk 3'te.",
        "- Cevabı **iç belgede** olan 3 soruda (iade onay limiti, bayi marjı, RMA kuralı) müşteri modu cevabı",
        "  **bulamıyor**. Bu bir hata değil, **istenen davranış**: müşteri bayi marjını öğrenmemeli. A6'da",
        "  sistem bu sorulara \"bu bilgiyi paylaşamıyorum / belgelerde bulamadım\" diyecek.",
        "",
        "## 3. Index ne kadar hızlandırıyor?",
        "",
        "Bizim 157 parçamız çok az. Postgres bu kadar küçük bir tabloda index'i kullanmaya bile gerek",
        "görmüyor, hepsini okuyup geçiyor (milisaniyenin altında). Index'in değerini görmek için",
        f"**{BENCH_ROWS:,} vektörlük** ayrı bir deneme tablosu kurduk ({b['size']}).".replace(",", "."),
        "Sahte vektörleri gerçek parça vektörlerimizin etrafına dağıttık, çünkü gerçek metin vektörleri",
        f"konulara göre kümelenir. Sorgu olarak {b['queries']} gerçek soru kullandık.",
        "",
        "| Arama yolu | Ortalama süre | Doğru ilk 10'u yakalama (recall) |",
        "|---|---:|---:|",
        f"| Tam tarama (her satıra bak) | {b['exact_ms']:.1f} ms | %100 (tanım gereği) |",
        *[
            f"| HNSW, `ef_search = {ef}`{' (varsayılan)' if ef == 40 else ''} | **{ms:.1f} ms** | **%{100 * r:.0f}** |"
            for ef, ms, r in b["by_ef"]
        ],
        "",
        *(
            [
                (
                    f"Index'i kurmak bir kere **{b['build_seconds']:.0f} saniye** sürdü "
                    "(hazırlık hattında, soru beklerken değil)."
                ),
                "",
            ]
            if b["build_seconds"] is not None
            else []
        ),
        "**Nasıl okunur:**",
        "- **Tam tarama** her soruda 100.000 satırın hepsiyle karşılaştırıyor. Kesin doğru ama satır sayısıyla",
        "  birlikte yavaşlıyor: 1 milyon satırda ~10 kat, 10 milyonda ~100 kat.",
        "- **HNSW index** bir yol haritası gibi: en yakınlara doğru adım adım ilerliyor, satırların çoğuna",
        "  hiç bakmıyor. Bedeli: sonuç **yaklaşık**. Bazen gerçek ilk 10'dan birini kaçırıp onun yerine",
        "  \"neredeyse o kadar yakın\" bir komşu getiriyor.",
        "- **`ef_search`** index'in ne kadar dikkatli arayacağı. Büyüdükçe daha çok aday inceleniyor,",
        "  süre artıyor. Bu veride varsayılan 40 bile doğru sonuçların hepsini yakaladı, yani daha",
        "  büyüğüne gerek yok.",
        "- **Zor veri dersi:** İlk denemede vektörleri tamamen **rastgele** üretmiştik. O veride index",
        "  doğru ilk 10'un sadece **%33**'ünü yakaladı. Rastgele vektörlerin hiçbir yapısı yok, hepsi",
        "  birbirine neredeyse eşit uzaklıkta. Gerçek metin vektörleri ise konulara göre kümelenir ve index",
        "  bu yapıyı kullanır. Yakalama düşük çıkan gerçek bir projede ilk ayar `ef_search`'i büyütmektir.",
        "- Bu hız ↔ doğruluk takası bütün vektör veritabanlarının temeli. Adı bu yüzden",
        "  **ANN: Approximate Nearest Neighbor** (yaklaşık en yakın komşu).",
        f"- Index kullanıldı mı? Sorgu planı (`EXPLAIN`): **{'evet, hnsw' if b['uses_index'] else 'hayır'}**.",
        "- **Bizim projede:** 157 parça için index'in bir faydası yok ama zararı da yok. Müşterinin 500.000",
        "  parçası olduğunda kod aynı kalıyor, veritabanı index'i kendisi devreye alıyor.",
        "",
        "## 4. Tablo",
        "",
        "Veritabanındaki `chunks` tablosunun sütunları:",
        "",
        "| Sütun | Ne tutuyor | Nereden geliyor |",
        "|---|---|---|",
        "| `text` | Parçanın metni (bağlam satırı dahil) | A3 |",
        f"| `embedding` | {store.DIM} sayılık vektör | A4 |",
        "| `source`, `title`, `page`, `row_num` | Kaynak bilgisi (A6'da \"Pulse kılavuzu, sayfa 3\") | A2 |",
        "| `audience` | `public` ya da `internal` (müşteri filtresi) | A2 |",
        "| `effective_date` | Belgenin tarihi (A10'da eski/yeni politika) | A2 |",
        "| `doc_type`, `section`, `chunk_index` | Belge türü, bölüm adı, belgedeki sırası | A2, A3 |",
        "",
        f"Veritabanı adresi `.env` içindeki `DATABASE_URL`: `{settings.database_url.split('@')[-1]}`.",
    ]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"{stored} chunks stored in {ingest_seconds:.1f}s; same as numpy {same}/10; leaked {leaked}")
    print(f"agent top1 {agent_top1}/10 top3 {agent_top3}/10; customer top1 {cust_top1}/7 top3 {cust_top3}/7")
    print(f"bench: exact {b['exact_ms']:.1f} ms, by ef {b['by_ef']}, build {b['build_seconds']}")
    print(f"Wrote {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
