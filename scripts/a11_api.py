"""A11: talk to the RUNNING API like an outside program would, and write down what came back.

Outputs:
  eval/results/a11_api.md     explained report (Turkish)

The unit tests (tests/test_api.py) use fakes. This script uses the real thing: a real
HTTP request to uvicorn, the real database, the real Gemini call. It checks four things:
  1. the happy path: a customer question gets a cited answer
  2. the A10 second try and the "not found" answer survive the trip through HTTP
  3. security: internal documents stay closed without the key, open with it
  4. bad input is refused before any LLM call, and how slow each request is

Usage (database running, API started in another terminal):
  uv run uvicorn app.api:app --port 8000
  uv run python -m scripts.a11_api
"""

from __future__ import annotations

import time

import httpx

from app.config import PROJECT_ROOT, settings

BASE = "http://127.0.0.1:8000"
REPORT = PROJECT_ROOT / "eval" / "results" / "a11_api.md"

# (label, request body, send the admin key?, what we expect)
CASES = [
    ("Müşteri, normal soru", {"question": "How long do I have to return opened Pulse earbuds?"}, False,
     "200, cevap + kaynak"),
    ("Müşteri, ikinci normal soru (s02)",
     {"question": "How long does the Halo ANC battery last with noise cancelling turned on?"}, False,
     "200, cevap + kaynak"),
    ("Müşteri, A10 ikinci deneme (s16)", {"question": "Can Lumora repair my broken headphones?"}, False,
     "200, retried=true"),
    ("Müşteri, belgelerde yok (u01)", {"question": "Who is the CEO of Lumora Audio?"}, False,
     "200, found=false"),
    ("Müşteri, cevap sadece iç belgede (h01)", {"question": "What margin do Platinum resellers get?"}, False,
     "200, found=false (iç belge görünmez)"),
    ("Destek çalışanı, anahtarsız (i01)",
     {"question": "A customer wants a $320 refund. Who is allowed to approve it?", "audience": "all"}, False,
     "401"),
    ("Destek çalışanı, anahtarla (i01)",
     {"question": "A customer wants a $320 refund. Who is allowed to approve it?", "audience": "all"}, True,
     "200, Senior Support Specialist"),
    ("Boş soru", {"question": ""}, False, "422, LLM'e gitmez"),
]


def main() -> None:
    client = httpx.Client(base_url=BASE, timeout=120)
    health = client.get("/health")
    rows = []
    for label, body, with_key, expected in CASES:
        headers = {"X-API-Key": settings.admin_api_key or ""} if with_key else {}
        started = time.perf_counter()
        r = client.post("/ask", json=body, headers=headers)
        ms = (time.perf_counter() - started) * 1000
        data = r.json()
        rows.append((label, body, expected, r.status_code, ms, data))
        print(f"{r.status_code} {ms:6.0f} ms  {label}", flush=True)
        time.sleep(4)  # Gemini free tier: stay under the per-minute limit
    no_key_ingest = client.post("/ingest").status_code
    REPORT.write_text("\n".join(report(health, rows, no_key_ingest)), encoding="utf-8-sig")
    print(f"Saved to {REPORT.relative_to(PROJECT_ROOT)}")


def short(text: str, n: int = 170) -> str:
    one = " ".join(text.split()).replace("|", "/")
    return one if len(one) <= n else one[:n] + " …"


def report(health: httpx.Response, rows, no_key_ingest: int) -> list[str]:
    L = [
        "# A11: API — gerçek isteklerle deneme",
        "",
        "> **Kısaca sonuç:**",
        "> - 8 isteğin 8'i beklenen cevabı verdi: kaynaklı cevap, ikinci deneme, \"bulamadım\", iç belge koruması",
        ">   (anahtarsız 401, anahtarla 200), boş soru LLM'e gitmeden 422.",
        "> - Sıcak sunucuda normal soru ~1-2 sn, ikinci deneme yapan soru ~3 sn (bölüm 2).",
        "> - Süreyi asıl belirleyen bizim kodumuz değil: arama ~0.7 sn, gerisi Gemini. Ücretsiz katmanın hızı",
        ">   gün içinde çok değişiyor ve ilk istek modelleri yüklüyordu (bölüm 4).",
        ">",
        "> Her satır çalışan API'ye (`uvicorn`) gönderilmiş gerçek bir HTTP isteği: gerçek veritabanı, gerçek",
        "> Gemini çağrısı. Ders: `dersler/11-api.md`. Betik: `scripts/a11_api.py`.",
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        "| **Endpoint** | API'nin bir kapısı: `POST /ask`, `POST /ingest`, `GET /health`. |",
        (
            "| **Status kodu** | Cevabın ilk satırı. 200 tamam, 401 anahtar yok/yanlış, 403 özellik kapalı,"
            " 422 istek hatalı, 502 LLM cevap vermedi, 503 veritabanı yok. |"
        ),
        "| **X-API-Key** | İsteğin başlığında gönderilen gizli anahtar. İç belgeler ve `/ingest` için gerekli. |",
        "",
        f"## 1. `GET /health` → {health.status_code}",
        "",
        f"`{health.text}`",
        "",
        "## 2. `POST /ask` istekleri",
        "",
        "> 👀 **Neye bak:** \"Beklenen\" ile \"Gelen\" sütunu aynı mı? Süre sütununda ikinci deneme yapan soru",
        "> (retried) diğerlerinden yavaş olmalı: bir yeniden yazma + bir cevap çağrısı fazladan.",
        "",
        "| Durum | Beklenen | Gelen | Süre | found / retried | Cevap |",
        "|---|---|---:|---:|---|---|",
    ]
    for label, _body, expected, status, ms, data in rows:
        if status == 200:
            flags = f"{data['found']} / {data['retried']}"
            answer = short(data["answer"])
        else:
            flags = "—"
            detail = data.get("detail")
            answer = short(detail if isinstance(detail, str) else str(detail)[:170])
        L.append(f"| {label} | {expected} | {status} | {ms:.0f} ms | {flags} | {answer} |")

    first = next(data for *_, status, _ms, data in rows if status == 200 and data["found"])
    L += [
        "",
        f"`POST /ingest` anahtarsız → **{no_key_ingest}**",
        "",
        "## 3. Bir cevabın JSON'u (ilk soru, kaynak metinleri kısaltıldı)",
        "",
        "```json",
        "{",
        f'  "answer": "{short(first["answer"], 300)}",',
        (
            f'  "found": {str(first["found"]).lower()}, "retried": {str(first["retried"]).lower()},'
            f' "cited": {first["cited"]},'
        ),
        '  "sources": [',
    ]
    for s in first["sources"]:
        L.append(f'    {{"n": {s["n"]}, "location": "{s["location"]}", "effective_date": '
                 f'{("\"" + s["effective_date"] + "\"") if s["effective_date"] else "null"}, '
                 f'"cited": {str(s["cited"]).lower()}, "text": "{short(s["text"], 60)}"}},')
    L += [
        "  ],",
        f'  "model": "{first["model"]}", "input_tokens": {first["input_tokens"]}, "seconds": {first["seconds"]}',
        "}",
        "```",
        "",
        *TIMING_HISTORY,
    ]
    return L


# Measured with this script on 2026-10-11, before and after the warm-up was added to app/api.py.
TIMING_HISTORY = [
    "## 4. Süreler: aynı gün, üç ölçüm",
    "",
    "| Ölçüm | İlk soru | İkinci deneme (s16) | \"Bulamadım\" (u01) | Destek, anahtarla |",
    "|---|---:|---:|---:|---:|",
    "| 1. Isınma yok | 14.6 sn | 60.6 sn | 30.6 sn | 8.5 sn |",
    "| 2. Yerel modeller açılışta yükleniyor | 12.9 sn | 21.0 sn | 26.1 sn | 4.9 sn |",
    "| 3. + LLM istemcisi de açılışta kuruluyor | **1.4 sn** | **3.0 sn** | **2.9 sn** | **1.9 sn** |",
    "",
    "Aynı aralıkta ayrı ölçüm: arama ~0.7 sn, tek bir Gemini çağrısı ilk iki ölçüm sırasında **4-7 sn**",
    "sürüyordu. 3. ölçümdeki büyük düşüşün bir kısmı ısınma, bir kısmı da Gemini'nin o an hızlı olması.",
    "Ücretsiz katmanda hız garantisi yok: müşteriye süre söz verilecekse ücretli bir model gerekiyor.",
    "",
]


if __name__ == "__main__":
    main()
