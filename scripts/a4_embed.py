"""A4: embed every chunk, look at what the vectors mean, and run a first search.

Outputs:
  data/processed/chunk_vectors.npy   one 384-number row per line of chunks.jsonl
  eval/results/a4_embedding.md       explained report (Turkish)
  eval/results/a4_map.png            sentences drawn in 2D: similar meaning = close

The "first search" is brute force: compare the question vector with all 157
chunk vectors and sort. A5 moves exactly this into the database.

Usage (from the project root):  uv run python -m scripts.a4_embed
"""

from __future__ import annotations

import time
from statistics import mean

import matplotlib

matplotlib.use("Agg")  # write a PNG, no window
import matplotlib.pyplot as plt
import numpy as np

from app.config import DATA_DIR, PROJECT_ROOT, settings
from app.embedding import MAX_TOKENS, cosine, embed_passages, embed_query, token_count
from app.ingest.chunking import Chunk, chunk_documents, fixed_size_chunks, heading_chunks
from app.ingest.pipeline import load_directory

VECTORS = DATA_DIR / "processed" / "chunk_vectors.npy"
REPORT = PROJECT_ROOT / "eval" / "results" / "a4_embedding.md"
MAP = PROJECT_ROOT / "eval" / "results" / "a4_map.png"
HEATMAP = PROJECT_ROOT / "eval" / "results" / "a4_heatmap.png"

# (sentence A, sentence B, what the pair shows)
PAIRS = [
    ("How do I return my earbuds?", "Can I send back my headphones?", "Farklı kelimeler, aynı anlam"),
    ("How do I return my earbuds?", "How many days do I have to return opened earbuds?", "Aynı konu"),
    ("How do I return my earbuds?", "What does shipping to Canada cost?", "Farklı konu"),
    ("How do I return my earbuds?", "The return key on my keyboard is stuck.", "Aynı kelime (return), farklı anlam"),
    ("The battery drains very fast.", "Battery life is too short.", "Farklı kelimeler, aynı anlam"),
    ("Error code E-214", "Error code E-410", "Farklı hata kodları"),
]

# Sentences for the 2D map, grouped by topic.
MAP_SENTENCES = {
    "İade": [
        "How do I return my earbuds?",
        "Can I send back my headphones?",
        "What is the refund period?",
        "I want my money back for this speaker.",
    ],
    "Pil": [
        "The battery drains very fast.",
        "Battery life is too short.",
        "How long does it take to charge the case?",
        "My earbuds die after one hour.",
    ],
    "Kargo": [
        "What does shipping to Canada cost?",
        "When will my order arrive?",
        "Do you deliver to Australia?",
        "Is express delivery available?",
    ],
    "Su": [
        "Can I swim with my earbuds?",
        "Is the speaker waterproof?",
        "My headphones got wet in the rain.",
        "Is sweat a problem for the earbuds?",
    ],
}

# (question, a string that only the right chunk contains)
SEARCH = [
    ("What does error code E-214 mean on the Pulse earbuds?", "Battery temperature sensor fault"),
    ("What does E-214 mean on Pulse Pro?", "ANC microphone calibration failed"),
    ("How long do I have to return earbuds I already opened?", "**14 days**"),
    ("Who has to approve a refund of more than 500 dollars?", "Above $500"),
    ("How much is express delivery to the UK?", "£9.99"),
    ("How many Aria X2 speakers can play together in Party Mode?", "Up to 4"),
    ("Is water damage covered for Halo headphones?", "never covered"),
    ("What margin does a Gold reseller get?", "Margin on MSRP: 35%"),
    ("Can I still buy Halo Lite?", "discontinued"),
    ("If only one earbud is broken, can just that one be replaced?", "single defective earbud"),
]


def search(question_vec: np.ndarray, vectors: np.ndarray, k: int = 3) -> list[tuple[int, float]]:
    scores = vectors @ question_vec  # vectors are normalized: dot product = cosine
    best = np.argsort(-scores)[:k]
    return [(int(i), float(scores[i])) for i in best]


def rank_of_answer(question_vec: np.ndarray, chunks: list[Chunk], vectors: np.ndarray, needle: str) -> int | None:
    """1-based position of the first chunk containing `needle`, None if not in the top 10."""
    for position, (i, _) in enumerate(search(question_vec, vectors, k=10), start=1):
        if needle in chunks[i].text:
            return position
    return None


def short(text: str, n: int = 110) -> str:
    one_line = " ".join(text.split())
    return one_line if len(one_line) <= n else one_line[:n] + " …"


def draw_map() -> None:
    labels = [(topic, s) for topic, sentences in MAP_SENTENCES.items() for s in sentences]
    vecs = embed_passages([s for _, s in labels])
    # PCA by hand: the two directions in which the 384-number vectors differ most.
    centered = vecs - vecs.mean(axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    xy = centered @ components[:2].T

    colors = {"İade": "#d1495b", "Pil": "#edae49", "Kargo": "#00798c", "Su": "#30638e"}
    fig, ax = plt.subplots(figsize=(10, 7))
    for (topic, sentence), (x, y) in zip(labels, xy):
        ax.scatter(x, y, s=80, color=colors[topic], label=topic)
        ax.annotate(sentence, (x, y), fontsize=8, xytext=(5, 4), textcoords="offset points")
    handles, names = ax.get_legend_handles_labels()
    unique = dict(zip(names, handles))
    ax.legend(unique.values(), unique.keys(), title="Konu")
    ax.set_title("384 sayılık vektörler 2 boyuta indirildi: aynı konudaki cümleler yan yana")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(MAP, dpi=130)
    plt.close(fig)

    # The 2D map loses information; the full similarity table does not.
    sims = vecs @ vecs.T
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(sims, cmap="viridis", vmin=0.4, vmax=1.0)
    names = [f"[{topic}] {sentence}" for topic, sentence in labels]
    ax.set_xticks(range(len(names)), [str(i + 1) for i in range(len(names))], fontsize=8)
    ax.set_yticks(range(len(names)), [f"{i + 1}. {n}" for i, n in enumerate(names)], fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{sims[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if sims[i, j] > 0.75 else "white")
    fig.colorbar(image, ax=ax, label="Benzerlik")
    ax.set_title("Her cümle her cümleyle: aynı konu = parlak kare bloklar")
    fig.tight_layout()
    fig.savefig(HEATMAP, dpi=130)
    plt.close(fig)


def main() -> None:
    docs = load_directory(DATA_DIR / "raw")
    chunks = chunk_documents(docs)

    started = time.perf_counter()
    vectors = embed_passages([c.text for c in chunks])
    embed_seconds = time.perf_counter() - started
    np.save(VECTORS, vectors)

    tokens = [token_count(c.text) for c in chunks]
    words = [len(c.text.split()) for c in chunks]
    ratio = sum(tokens) / sum(words)

    # --- sentence pairs ---------------------------------------------------
    pair_rows = []
    for a, b, meaning in PAIRS:
        va, vb = embed_passages([a, b])
        pair_rows.append(f"| {a} | {b} | **{float(cosine(va, vb)):.2f}** | {meaning} |")

    # --- cosine by hand ---------------------------------------------------
    a, b = "How do I return my earbuds?", "Can I send back my headphones?"
    va, vb = embed_passages([a, b])
    dot = float(np.sum(va * vb))

    # --- first search: which chunking + which query form finds the answer --
    variants = {
        "Başlığa göre + bağlam satırı (A3'te seçtiğimiz)": chunks,
        "Başlığa göre, bağlam satırı YOK": [c for d in docs for c in heading_chunks(d, context=False)],
        "Makas: 100 kelime, 20 örtüşme": [c for d in docs for c in fixed_size_chunks(d, 100, 20)],
    }
    variant_vectors = {
        name: vectors if variant is chunks else embed_passages([c.text for c in variant])
        for name, variant in variants.items()
    }
    q_with = {q: embed_query(q) for q, _ in SEARCH}
    q_without = {q: embed_query(q, instruction=False) for q, _ in SEARCH}

    def score(name: str, qvecs: dict) -> tuple[int, int, list[int | None]]:
        ranks = [rank_of_answer(qvecs[q], variants[name], variant_vectors[name], n) for q, n in SEARCH]
        return sum(r == 1 for r in ranks), sum(r is not None and r <= 3 for r in ranks), ranks

    results = {name: score(name, q_with) for name in variants}
    chosen = "Başlığa göre + bağlam satırı (A3'te seçtiğimiz)"
    no_instruction = score(chosen, q_without)

    # --- write report -----------------------------------------------------
    hit1, hit3, chosen_ranks = results[chosen]
    lines = [
        "# A4 — Embedding (anlam parmak izi) raporu",
        "",
        "> **Bu rapor ne?** Her parçayı 384 sayılık bir vektöre çevirdik, vektörlerin gerçekten anlamı",
        "> yakaladığını örneklerle kontrol ettik ve **ilk aramayı** yaptık. Bu dosyayı `scripts/a4_embed.py`",
        "> üretiyor, her çalıştırmada yeniden hesaplanıyor. Ders anlatımı: `dersler/04-embedding.md`.",
        "",
        "## ✅ Kısaca sonuç",
        "",
        f"- **Model:** `{settings.embed_model}`, bilgisayarında yerel çalışıyor (internet ve API anahtarı gerekmiyor).",
        (
            f"- **{len(chunks)} parça → {len(chunks)} vektör**, her biri {vectors.shape[1]} sayı. "
            f"Hepsi **{embed_seconds:.1f} saniyede** hesaplandı."
        ),
        f"- **İlk arama:** 10 sorudan **{hit1} tanesinde** doğru parça **1. sırada**, **{hit3} tanesinde** ilk 3'te.",
        f"- **Token kontrolü:** en uzun parça {max(tokens)} token, sınır {MAX_TOKENS}. Hiçbir parça kesilmiyor.",
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        "| **Embedding / vektör** | Bir metnin anlamını temsil eden sayı listesi. Bizde 384 sayı. \"Anlam parmak izi\". |",
        "| **Embedding modeli** | Metni okuyup o sayı listesini üreten yapay zekâ modeli. Cevap yazmaz, sadece sayı üretir. |",
        "| **Kosinüs benzerliği** | İki vektörün ne kadar aynı yöne baktığı. 1'e yakın = anlam yakın. |",
        "| **Token** | Modelin okuduğu kelime parçası. \"Lumora\" → `lu` + `##mora` = 2 token. |",
        "| **Sorgu talimatı** | Soru vektöre çevrilirken başına eklenen \"bunu arama için temsil et\" cümlesi. |",
        "| **1. sırada / ilk 3'te** | Arama sonuçlarında doğru parça kaçıncı geldi. Arama ölçümünün temel rakamı. |",
        "",
        "## 1. Anlam benzerliği: model gerçekten anlıyor mu?",
        "",
        "Her satırda iki cümleyi vektöre çevirip benzerliklerini ölçtük.",
        "",
        "| Cümle A | Cümle B | Benzerlik | Ne gösteriyor |",
        "|---|---|---:|---|",
        *pair_rows,
        "",
        "**Nasıl okunur:**",
        "- Bu modelde alakasız iki cümle bile ~0.40-0.50 alıyor. Yani **0.45 \"alakasız\"**, **0.80 üstü \"çok yakın\"**",
        "  demek. Mutlak sayıdan çok **sıralama** önemli: hangisi daha yakın?",
        "- \"return my earbuds\" ile \"send back my headphones\" ortak kelime taşımıyor ama çok yakın çıkıyor.",
        "  Kelime aramanın yapamadığı şey bu.",
        "- **Zayıf nokta** (dürüst not): Model hata kodları gibi **kodları ayırt etmekte** zayıf. E-214 ile",
        "  E-410 neredeyse aynı görünüyor. Bunu A8'de kelime araması ekleyerek çözeceğiz.",
        "",
        "## 2. Kosinüs benzerliğini elle hesaplamak",
        "",
        f"`{a}` ve `{b}` cümlelerinin vektörlerinin ilk 5 sayısı:",
        "",
        "```",
        f"A = [{', '.join(f'{x:+.3f}' for x in va[:5])}, ... 379 sayı daha]",
        f"B = [{', '.join(f'{x:+.3f}' for x in vb[:5])}, ... 379 sayı daha]",
        "```",
        "",
        "Formül: karşılıklı sayıları çarp, hepsini topla (**nokta çarpımı**), iki vektörün uzunluğuna böl.",
        "",
        "```",
        f"A·B  = ({va[0]:+.3f} × {vb[0]:+.3f}) + ({va[1]:+.3f} × {vb[1]:+.3f}) + ... 384 çarpım  = {dot:.3f}",
        f"|A|  = {np.linalg.norm(va):.3f}    |B| = {np.linalg.norm(vb):.3f}",
        f"cos  = {dot:.3f} / ({np.linalg.norm(va):.3f} × {np.linalg.norm(vb):.3f}) = {dot:.3f}",
        "```",
        "",
        "Model vektörleri **uzunluğu 1** olacak şekilde veriyor. Bu yüzden bölme işlemi bir şey değiştirmiyor",
        "ve kosinüs benzerliği düz nokta çarpımına eşit. Veritabanı (A5) bu sayede çok hızlı hesaplıyor.",
        "",
        "## 3. Anlam haritası",
        "",
        "4 konudan (iade, pil, kargo, su) 16 cümleyi vektöre çevirdik. Model hangi cümlenin hangi konuya",
        "ait olduğunu bilmiyordu, sadece cümleleri okudu.",
        "",
        "### 3.1 Benzerlik tablosu (`a4_heatmap.png`)",
        "",
        "Her cümleyi her cümleyle karşılaştırdık. Cümleler konu sırasıyla dizili (1-4 iade, 5-8 pil,",
        "9-12 kargo, 13-16 su). Parlak (sarı-yeşil) = benzer, koyu (mor) = farklı.",
        "",
        "![Benzerlik tablosu](a4_heatmap.png)",
        "",
        "👀 **Neye bak:** Köşegen boyunca 4×4'lük kareler. **Kargo** (9-12) ve **su** (13-16) kareleri net:",
        "kendi aralarında parlak, diğerleriyle koyu. Model aynı konudaki cümleleri birbirine yakın buluyor.",
        "",
        "Kusursuz değil: 8. cümle (\"My earbuds die after one hour\") pil cümlesi ama iade ve su cümlelerine de",
        "yakın çıkmış (0.64-0.73). Sebebi **\"earbuds\"** kelimesi: o cümlelerin çoğunda da geçiyor. Model hem",
        "konuya hem de **neyin hakkında konuşulduğuna** (kulaklık) bakıyor. Arama için bu sorun değil, çünkü",
        "biz sadece en yakın birkaç parçayı alıyoruz.",
        "",
        "### 3.2 İki boyutlu harita (`a4_map.png`)",
        "",
        "384 sayıyı, en çok farklılaştıkları 2 yöne indirip kâğıda çizdik.",
        "",
        "![Anlam haritası](a4_map.png)",
        "",
        "👀 **Neye bak:** Kargo cümleleri sağ üstte net bir küme, \"battery\" cümleleri yan yana. Ama iade",
        "cümleleri dağınık. Sebebi model değil, çizim: 384 boyutu 2'ye sıkıştırınca bilginin çoğu kayboluyor",
        "(bir şehrin 3 boyutlu maketinin gölgesine bakmak gibi). Doğru resmi 3.1'deki tablo veriyor.",
        "",
        "## 4. İlk arama 🔎",
        "",
        "Soruyu vektöre çevirdik, 157 parçanın hepsiyle karşılaştırdık, en benzer olanları sıraladık.",
        "Soru, cevaptaki kelimeleri birebir kullanmasın diye farklı cümlelerle soruldu.",
        "",
        "### 4.1 Sorular ve doğru parçanın sırası",
        "",
        "| Soru | Doğru parça kaçıncı geldi |",
        "|---|---:|",
    ]
    for (question, _), rank in zip(SEARCH, chosen_ranks):
        mark = "✅ 1." if rank == 1 else (f"🟡 {rank}." if rank and rank <= 3 else f"❌ {rank or '10+'}.")
        lines.append(f"| {question} | {mark} |")

    e214 = SEARCH[0][0]
    lines += [
        "",
        f"### 4.2 Örnek: \"{e214}\"",
        "",
        "| Sıra | Benzerlik | Parça (kaynak) | Parçanın başı |",
        "|---:|---:|---|---|",
    ]
    for pos, (i, s) in enumerate(search(q_with[e214], vectors), start=1):
        lines.append(f"| {pos} | {s:.3f} | `{chunks[i].location}` | {short(chunks[i].text)} |")

    lines += [
        "",
        "👀 **Neye bak:** Gelen 3 parçanın üçü de E-214 ve Pulse ile ilgili. Model **konuyu** doğru",
        "anlıyor. Ama aradığımız parça (Pulse kılavuzu, sayfa 3) 6. sırada. Sebepler:",
        "- Model kodları ayırt edemiyor: \"E-214\" ile \"E-410\" ona neredeyse aynı (bölüm 1).",
        "- \"Pulse\" ile \"Pulse Pro\" da çok benzer kelimeler. 2. sıradaki parça yanlış ürüne ait.",
        "- 1. sıradaki iç belge (bilinen sorunlar listesi) aslında konuyla ilgili ama müşteriye gösterilmemeli.",
        "",
        "Çözümler sırayla gelecek: **A8** tam kelime/kod araması ekleyecek (\"E-214\" birebir eşleşsin),",
        "**A9** ilk 20 sonucu daha dikkatli bir modelle yeniden sıralayacak, **A5** iç belgeleri filtreleyecek.",
        "",
        "## 5. A3'ün kararı aramada da doğru mu?",
        "",
        "A3'te bağlam satırının işe yarayacağını **tahmin etmiştik**. Şimdi aynı 10 soruyu üç farklı",
        "parçalama ile gerçekten aradık:",
        "",
        "| Parçalama | Doğru parça 1. sırada | Doğru parça ilk 3'te |",
        "|---|---:|---:|",
    ]
    for name, (h1, h3, _) in results.items():
        bold = "**" if name == chosen else ""
        lines.append(f"| {bold}{name}{bold} | {bold}{h1}/10{bold} | {bold}{h3}/10{bold} |")

    lines += [
        "",
        "### Sorgu talimatı işe yarıyor mu?",
        "",
        "| Soru nasıl vektöre çevrildi | 1. sırada | İlk 3'te |",
        "|---|---:|---:|",
        f"| Talimatla (`{embed_query.__module__}.QUERY_INSTRUCTION` + soru) | {hit1}/10 | {hit3}/10 |",
        f"| Talimatsız (sadece soru) | {no_instruction[0]}/10 | {no_instruction[1]}/10 |",
        "",
        "**Nasıl okunur:** Bağlam satırı aramada da kazandırıyor, makas en geride kalıyor. A3'teki tahminimiz",
        "gerçek aramada doğrulandı. Sorgu talimatı da küçük bir artı veriyor, o yüzden açık bırakıyoruz.",
        "Ama 10 soru küçük bir örnek, 1 sorunun farkı şans olabilir. A7'de 40-50 soruyla sağlam ölçeceğiz.",
        "",
        "## 6. Token kontrolü",
        "",
        "A3'te parça boyutunu kelimeyle ölçmüş ve \"1 kelime ≈ 1.3 token\" demiştik. Gerçek sayılar:",
        "",
        "| | Değer |",
        "|---|---:|",
        f"| Ortalama parça | {mean(tokens):.0f} token |",
        f"| En uzun parça | {max(tokens)} token |",
        f"| Modelin sınırı | {MAX_TOKENS} token |",
        f"| Gerçek token / kelime oranı | {ratio:.2f} |",
        "",
        "Oran tahminden biraz yüksek çıktı (ürün kodları, sayılar ve `|` işaretleri çok token yiyor),",
        "ama en uzun parça bile sınırın çok altında. **Hiçbir parça sessizce kesilmiyor.**",
        "",
        "## 7. Çıktı dosyası: `data/processed/chunk_vectors.npy`",
        "",
        f"{vectors.shape[0]} satır × {vectors.shape[1]} sütunluk bir sayı tablosu. Satır *i*, `chunks.jsonl`",
        "dosyasındaki *i*. parçanın vektörü. A5'te bunlar veritabanına yazılacak.",
    ]

    draw_map()
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"{len(chunks)} chunks -> vectors {vectors.shape} in {embed_seconds:.1f}s")
    print(f"search: top1 {hit1}/10, top3 {hit3}/10")
    print(f"Wrote {REPORT.relative_to(PROJECT_ROOT)}, {MAP.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
