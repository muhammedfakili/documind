"""A3: compare chunking strategies on the real dataset and write a report.

Outputs:
  data/processed/chunks.jsonl     heading chunks (the strategy we keep), one per line
  eval/results/a3_chunking.md     comparison table + side-by-side samples, explained in Turkish

Before embeddings exist (A4) we cannot measure search yet, so this report uses
two proxies that predict it:
  facts kept whole   for a set of known facts, is there ONE chunk that holds
                     every piece of it (value + what the value means + which product)?
  knows its source   share of chunks whose text names the document they came from

Usage (from the project root):  uv run python -m scripts.a3_chunk
"""

from __future__ import annotations

import json
from statistics import mean

from app.config import DATA_DIR, PROJECT_ROOT
from app.ingest.chunking import Chunk, chunk_documents, fixed_size_chunks, heading_chunks, word_count
from app.ingest.documents import Document
from app.ingest.pipeline import load_directory

OUT = DATA_DIR / "processed" / "chunks.jsonl"
REPORT = PROJECT_ROOT / "eval" / "results" / "a3_chunking.md"

# (report label, strategy, settings)
CONFIGS = [
    ("Makas: 50 kelime, 10 örtüşme", "fixed", {"size": 50, "overlap": 10}),
    ("Makas: 100 kelime, örtüşme yok", "fixed", {"size": 100, "overlap": 0}),
    ("Makas: 100 kelime, 20 örtüşme", "fixed", {"size": 100, "overlap": 20}),
    ("Makas: 200 kelime, 40 örtüşme", "fixed", {"size": 200, "overlap": 40}),
    ("Başlığa göre, max 150, bağlam satırı YOK", "heading", {"max_words": 150, "context": False}),
    ("Başlığa göre, max 60, bağlam satırıyla", "heading", {"max_words": 60}),
    ("⭐ Başlığa göre, max 150, bağlam satırıyla", "heading", {"max_words": 150}),
]
CHOSEN = "⭐ Başlığa göre, max 150, bağlam satırıyla"
BASELINE = "Makas: 100 kelime, 20 örtüşme"

# A fact is kept whole when a single chunk contains every one of its strings:
# the value, what the value means, and which product/document it is about.
# (label, plain-language question, required strings)
FACTS = [
    ("Pulse E-214", "Pulse'ta E-214 hatası ne demek?",
     ["Lumora Pulse - User Manual", "E-214", "Battery temperature sensor fault"]),
    ("Pulse Pro E-214", "Pulse Pro'da E-214 hatası ne demek?",
     ["Pulse Pro", "E-214", "ANC microphone calibration failed"]),
    ("Açılmış kulaklık iadesi", "Açılmış kulaklık kaç gün içinde iade edilir?",
     ["Return Policy", "Opened", "Earbuds (Pulse, Pulse Pro)", "**14 days**"]),
    ("500$ üstü iade onayı", "500 doların üstündeki iadeyi kim onaylar?",
     ["Who can approve", "Above $500", "Customer Operations Manager"]),
    ("İngiltere ekspres kargo", "İngiltere'ye ekspres kargo ne kadar?",
     ["Express", "United Kingdom", "Next business day, £9.99"]),
    ("Aria X2 Party Mode", "Party Mode'da kaç Aria X2 bağlanır?", ["Aria X2", "Party Mode", "Up to 4"]),
    ("Halo su hasarı", "Halo kulaklıkta su hasarı garantiye girer mi?",
     ["Warranty", "Halo ANC", "water damage is never covered"]),
    ("Gold bayi marjı", "Gold seviyesindeki bayinin kâr marjı kaç?",
     ["Reseller", "Tier: Gold", "Margin on MSRP: 35%"]),
    ("Halo Lite satışta mı", "Halo Lite hâlâ satılıyor mu?",
     ["Is Halo Lite still available?", "discontinued in March 2026"]),
    ("Tek kulaklık değişimi", "Tek bir kulaklık arızalıysa sadece o değişir mi?",
     ["RMA", "single defective earbud", "`CASE`"]),
]

TERMS = [
    ("Parça (chunk)", "Belgenin küçük bir bölümü. Arama parça bulur, yapay zekâ parça okur. Sınav kartı gibi."),
    ("Fiş (Document)", "A2'de okunan belge: bir Markdown dosyası, bir PDF sayfası ya da bir Excel/CSV satırı."),
    ("Strateji", "Fişi parçalara bölme yöntemi. İki tane denedik: makas ve başlığa göre."),
    ("Makas (sabit boyut)", "Her N kelimede bir körlemesine kes. Cümleye, tabloya, başlığa bakmaz."),
    ("Örtüşme (overlap)", "Makasta, bir parçanın son birkaç kelimesini sonrakinin başına da koymak."),
    ("Başlığa göre", "Belgeyi `## ` başlıklarından böl. Uzunsa paragraf sonundan böl, cümle ortasından asla."),
    ("Bağlam satırı", "Parçanın en üstüne yazılan `Belge adı > Bölüm adı` satırı."),
    ("max 150", "Bir parçanın gövdesi en fazla yaklaşık 150 kelime olsun."),
    ("Bütün kalan bilgi", "Soruyu cevaplamak için gereken her şey TEK BİR parçada mı? (değer + anlamı + hangi ürün)"),
    ("Kaynağını bilen parça", "Parçanın kendi metninde hangi belgeden geldiği yazıyor mu?"),
]

TESTS = [
    ("test_fixed_size_overlap_repeats_words", "Makas doğru kesiyor mu: örtüşen kelimeler tekrar ediyor, sondaki kelime kaybolmuyor."),
    ("test_fixed_size_rejects_overlap_not_smaller_than_size", "Örtüşme parça boyutuna eşit ya da büyükse hata veriyor mu (yoksa sonsuz döngü olurdu)."),
    ("test_split_sections_drops_title_and_keeps_headings", "Belge `## ` başlıklarından doğru bölünüyor mu."),
    ("test_question_stays_with_its_answer", "SSS'te bir soru, cevabından ayrı parçaya düşmüyor mu."),
    ("test_split_table_repeats_its_header", "Uzun tablo bölününce her parça başlık satırıyla başlıyor mu."),
    ("test_chunk_keeps_page_for_citation", "E-214 parçası hâlâ \"Pulse kılavuzu, sayfa 3\" bilgisini taşıyor mu (A6'da kaynak göstermek için)."),
    ("test_csv_row_is_one_chunk", "30 destek kaydı 30 parça olarak kalıyor mu (satırlar zaten küçük)."),
    ("test_with_context_names_document_and_section", "Bağlam satırı doğru biçimde yazılıyor mu: `Belge > Bölüm`."),
    ("test_same_error_code_is_told_apart", "Pulse'un E-214'ü ile Pulse Pro'nun E-214'ü ayrı ve kendi ürün adlarıyla mı."),
]


def run(docs: list[Document], strategy: str, settings: dict) -> list[Chunk]:
    if strategy == "fixed":
        return [c for d in docs for c in fixed_size_chunks(d, **settings)]
    return [c for d in docs for c in heading_chunks(d, **settings)]


def facts_kept(chunks: list[Chunk]) -> list[str]:
    return [label for label, _, parts in FACTS if any(all(p in c.text for p in parts) for c in chunks)]


def knows_source(chunk: Chunk) -> bool:
    return chunk.doc.title in chunk.text


def first(chunks: list[Chunk], source: str, needle: str) -> Chunk:
    return next(c for c in chunks if c.doc.source == source and needle in c.text)


def code_block(text: str) -> list[str]:
    return ["```", text, "```"]


def main() -> None:
    docs = load_directory(DATA_DIR / "raw")
    # Row documents (CSV/Excel) are already one fact per chunk; text documents
    # are where strategies differ, so the comparison runs on those.
    text_docs = [d for d in docs if d.row is None]
    results = {label: run(text_docs, strategy, settings) for label, strategy, settings in CONFIGS}
    kept = {label: facts_kept(chunks) for label, chunks in results.items()}
    all_chunks = chunk_documents(docs)
    chosen, baseline = results[CHOSEN], results[BASELINE]

    def known_pct(label: str) -> float:
        chunks = results[label]
        return 100 * sum(knows_source(c) for c in chunks) / len(chunks)

    lines = [
        "# A3 — Parçalama (chunking) raporu",
        "",
        "> **Bu rapor ne?** Belgeleri parçalara bölmenin farklı yollarını denedik ve hangisinin daha iyi",
        "> parça ürettiğini ölçtük. Bu dosyayı `scripts/a3_chunk.py` üretiyor; elle yazılmadı, her",
        "> çalıştırmada yeniden hesaplanıyor. Ders anlatımı: `dersler/03-chunking.md`.",
        "",
        "## ✅ Kısaca sonuç",
        "",
        "- **Seçilen yöntem:** başlığa göre böl, parça başına en fazla ~150 kelime, üstüne bağlam satırı yaz.",
        (
            f"- **Skor:** 10 bilgiden **{len(kept[CHOSEN])} tanesi** tek parçada bütün kalıyor, parçaların "
            f"**%{known_pct(CHOSEN):.0f}**'i hangi belgeden geldiğini biliyor."
        ),
        (
            f"- **Karşılaştırma:** klasik makas (100 kelime) sadece **{len(kept[BASELINE])}/10** ve "
            f"**%{known_pct(BASELINE):.0f}**."
        ),
        (
            f"- **Çıktı:** {len(docs)} fiş → **{len(all_chunks)} parça**, `data/processed/chunks.jsonl` dosyasına "
            "kaydedildi. A4 bu dosyayı okuyacak."
        ),
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        *[f"| **{term}** | {meaning} |" for term, meaning in TERMS],
        "",
        "## 1. Karşılaştırma tablosu",
        "",
        (
            f"Deneme {len(text_docs)} metin fişi üzerinde yapıldı (PDF sayfaları + Markdown dosyaları). "
            f"CSV/Excel satırları ({len(docs) - len(text_docs)} adet) zaten küçük ve tek bilgi; her biri "
            "olduğu gibi 1 parça kalıyor, o yüzden karşılaştırmaya katılmadılar."
        ),
        "",
        "| Yöntem | Kaç parça | Ortalama uzunluk (kelime) | En uzun parça | Bütün kalan bilgi | Kaynağını bilen parça |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, _, _ in CONFIGS:
        chunks = results[label]
        sizes = [word_count(c.text) for c in chunks]
        bold = "**" if label == CHOSEN else ""
        lines.append(
            f"| {bold}{label}{bold} | {len(chunks)} | {mean(sizes):.0f} | {max(sizes)} | "
            f"{bold}{len(kept[label])}/10{bold} | {bold}%{known_pct(label):.0f}{bold} |"
        )
    lines += [
        "",
        "**Nasıl okunur:**",
        "- **Bütün kalan bilgi** ve **kaynağını bilen parça** ne kadar yüksekse o kadar iyi.",
        "- Makasta parça büyüdükçe skor artıyor (50 → 200). Ama büyük parça, A4'te bulanık bir anlama",
        "  dönüşür. Bu tablo bunu göremez, onu A7'deki gerçek arama ölçümü gösterecek.",
        "- Başlığa göre bölüp bağlam satırı yazmazsak skor makas 200'ün bile altında kalıyor.",
        "  **Farkı yaratan, bağlam satırı.**",
        "- Örtüşme 0'dan 20'ye çıkınca makas 100'ün skoru değişmiyor. Örtüşme, yapıyı bilmemenin yerini tutmuyor.",
        "",
        "## 2. Bilgi bilgi sonuç",
        "",
        "10 soru seçtik. Her biri için cevabın **tamamı** (değer + anlamı + hangi ürün) tek bir parçada var mı",
        "diye baktık. ✅ = var, arama bu parçayı bulursa soru cevaplanabilir. ❌ = bilgi parçalara dağılmış",
        "ya da hangi ürüne ait olduğu yazmıyor.",
        "",
        "| Soru | " + " | ".join(label for label, _, _ in CONFIGS) + " |",
        "|---|" + "---|" * len(CONFIGS),
    ]
    for label, question, _ in FACTS:
        marks = ["✅" if label in kept[name] else "❌" for name, _, _ in CONFIGS]
        lines.append(f"| {question} | " + " | ".join(marks) + " |")

    lines += [
        "",
        "## 3. Örnekler: aynı yer, iki farklı kesim",
        "",
        "### 3.1 Pulse kılavuzu, E-214 hata kodu (sayfa 3)",
        "",
        "👀 **Neye bak:** İlk satırlara bak. Makasla kesilen parçada **\"Pulse\" kelimesi hiç geçmiyor**.",
        "E-214, Pulse Pro'da başka bir anlama geliyor. Bu parçayı okuyan, hangi ürün olduğunu bilemez.",
        "Ayrıca makas parçası cümlenin ortasında (\"contact support for a\") bitiyor.",
        "",
        "**Makas (100 kelime, 20 örtüşme):**",
        "",
        *code_block(first(baseline, "manuals/pulse-user-manual.pdf", "E-214").text),
        "",
        "**⭐ Başlığa göre + bağlam satırı:**",
        "",
        *code_block(first(chosen, "manuals/pulse-user-manual.pdf", "E-214").text),
        "",
        "### 3.2 İade politikası 2026, iade süresi tablosu",
        "",
        "👀 **Neye bak:** Makas parçası tam `Earbuds (Pulse, Pulse` yazısında bitiyor. \"14 days\" bir sonraki",
        "parçaya düşüyor, orada da \"Unopened / Opened\" başlığı yok. Neyin 14 gün olduğu kayboluyor.",
        "Ayrıca makas, satır sonlarını sildiği için tablo tek uzun satıra dönüşmüş.",
        "",
        "**Makas (100 kelime, 20 örtüşme):**",
        "",
        *code_block(first(baseline, "policies/return-policy-2026.md", "Earbuds (Pulse").text),
        "",
        "**⭐ Başlığa göre + bağlam satırı:**",
        "",
        *code_block(first(chosen, "policies/return-policy-2026.md", "Earbuds (Pulse").text),
    ]

    small = results["Başlığa göre, max 60, bağlam satırıyla"]
    table_pieces = [
        c
        for c in small
        if c.doc.source == "policies/shipping-policy.md" and "| Region |" in c.text
    ]
    lines += [
        "",
        "### 3.3 Uzun bir tablo bölünmek zorunda kalırsa (kargo tablosu, max 60)",
        "",
        f"👀 **Neye bak:** Sınırı bilerek küçük tuttuk (60 kelime), tablo sığmadı ve {len(table_pieces)} parçaya",
        "bölündü. **Her parça `| Region | Standard | Express | ...` satırıyla başlıyor.** Böylece ikinci",
        "parçadaki \"€14.99\" yazısının ekspres kargo fiyatı olduğu hâlâ anlaşılıyor.",
    ]
    for i, piece in enumerate(table_pieces, start=1):
        lines += ["", f"**Parça {i}:**", "", *code_block(piece.text)]

    lines += [
        "",
        "## 4. Testler ne kontrol ediyor (`tests/test_chunking.py`)",
        "",
        "Testler kodu her değiştirdiğimizde bu kuralların hâlâ geçerli olduğunu otomatik kontrol ediyor.",
        "Çalıştırmak için: `uv run pytest`",
        "",
        "| Test | Sade anlamı |",
        "|---|---|",
        *[f"| `{name}` | {meaning} |" for name, meaning in TESTS],
        "",
        "## 5. Çıktı dosyası: `data/processed/chunks.jsonl`",
        "",
        "Her satır bir parça. İçinde parçanın metni (`text`, bağlam satırı dahil), hangi bölümden geldiği",
        "(`section`), belgedeki sırası (`index`) ve fişin arka yüzündeki bütün etiketler var (`source`, `page`,",
        "`row`, `audience`, `effective_date` ...). A4 bu dosyadaki `text` alanlarını vektöre çevirecek.",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"{len(docs)} documents -> {len(all_chunks)} chunks (heading, max 150)")
    print(f"Wrote {OUT.relative_to(PROJECT_ROOT)} and {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
