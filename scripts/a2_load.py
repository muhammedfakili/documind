"""A2: load data/raw into Documents and write an inspection report.

Outputs:
  data/processed/documents.jsonl   every Document, one JSON object per line
  eval/results/a2_loading.md       per-file summary + before/after samples

Usage (from the project root):  uv run python -m scripts.a2_load
"""

from __future__ import annotations

import json
from collections import defaultdict

from pypdf import PdfReader

from app.config import DATA_DIR, PROJECT_ROOT
from app.ingest.pipeline import load_directory

RAW = DATA_DIR / "raw"
PROCESSED = DATA_DIR / "processed" / "documents.jsonl"
REPORT = PROJECT_ROOT / "eval" / "results" / "a2_loading.md"


def main() -> None:
    docs = load_directory(RAW)

    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED.open("w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")

    by_source = defaultdict(list)
    for doc in docs:
        by_source[doc.source].append(doc)

    lines = [
        "# A2 — Belge yükleme raporu",
        "",
        f"**{len(by_source)} dosya → {len(docs)} Document**, toplam {sum(len(d.text) for d in docs):,} karakter.",
        "",
        "| Dosya | Document | Karakter | Tür | Kitle | Tarih |",
        "|---|---:|---:|---|---|---|",
    ]
    for source, group in by_source.items():
        dated = sorted({d.effective_date for d in group if d.effective_date})
        date_text = dated[0].isoformat() if len(dated) == 1 else (f"{len(dated)} farklı" if dated else "—")
        lines.append(
            f"| `{source}` | {len(group)} | {sum(len(d.text) for d in group):,} | "
            f"{group[0].doc_type} | {group[0].audience} | {date_text} |"
        )

    # Before/after: the same PDF page through plain pypdf and through our loader.
    page_before = PdfReader(RAW / "manuals/pulse-user-manual.pdf").pages[2].extract_text()
    page_after = next(d for d in docs if d.source == "manuals/pulse-user-manual.pdf" and d.page == 3).text
    xlsx_row = next(d for d in docs if d.source == "catalog/warranty-matrix.xlsx" and d.row == 1)
    ticket = next(d for d in docs if d.source == "support/tickets-2026-q3.csv" and d.row == 1)
    partner = next(d for d in docs if d.source == "support/retail-partners.csv" and d.row == 1)

    lines += [
        "",
        "## Önce / sonra: Pulse kılavuzu, sayfa 3 (hata kodu tablosu)",
        "",
        "**Düz `pypdf` çıktısı:**",
        "",
        "```",
        page_before.strip(),
        "```",
        "",
        "**Bizim loader'ın çıktısı:**",
        "",
        "```",
        page_after,
        "```",
        "",
        "## Excel satırı → metin",
        "",
        "```",
        xlsx_row.text,
        "```",
        "",
        "## Destek kaydı (özel formatlayıcı)",
        "",
        "```",
        ticket.text,
        "```",
        "",
        "## Bayi satırı (partner_row — alıştırma)",
        "",
        "```",
        partner.text,
        "```",
    ]

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8-sig")
    print(f"{len(by_source)} files -> {len(docs)} documents")
    print(f"Wrote {PROCESSED.relative_to(PROJECT_ROOT)} and {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
