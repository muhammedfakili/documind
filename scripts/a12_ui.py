"""A12: drive the REAL chat screen against the REAL API and write down what it showed.

Outputs:
  eval/results/a12_ui.md     explained report (Turkish)

tests/test_ui.py runs the page with a fake API. This script runs the same page
(Streamlit's AppTest: the page's code runs exactly as in the browser, without drawing
pixels) against the running API, the real database and the real Gemini call, and
checks what a user would see in eight situations: a normal answer, a table, a
closed-list "no", "not found", a customer asking about an internal document, an agent
without and with the key, and the API being down.

Usage (database running, API started in another terminal):
  uv run uvicorn app.api:app --port 8000
  uv run python -m scripts.a12_ui
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from streamlit.testing.v1 import AppTest

import ui.client
from app.config import PROJECT_ROOT, settings

APP = str(PROJECT_ROOT / "ui" / "app.py")
REPORT = PROJECT_ROOT / "eval" / "results" / "a12_ui.md"
DOWN_URL = "http://127.0.0.1:8009"  # nothing listens here: "API down"


@dataclass
class Case:
    label: str
    question: str
    expected: str
    check: Callable[[AppTest, dict | None], bool]
    agent: bool = False
    key: str | None = None  # "right" / "wrong" / None
    down: bool = False


def markdown(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def found_and_cited(at, resp) -> bool:
    return bool(resp and resp["found"] and resp["cited"])


CASES = [
    Case(
        "Müşteri, iade (c01)",
        "I opened my Pulse Pro earbuds 20 days ago. Can I still return them?",
        'cevap "No" + kaynak kartında tablo',
        lambda at, r: found_and_cited(at, r) and r["answer"].startswith("No") and "|---|" in markdown(at),
    ),
    Case(
        "Müşteri, tablo/fiyat (t01)",
        "How much does Pulse Pro cost in the UK?",
        "£169 + kaynak",
        lambda at, r: found_and_cited(at, r) and "169" in r["answer"],
    ),
    Case(
        "Müşteri, kapalı liste (s17)",
        "Can I order Lumora products to Japan?",
        '"No" + kaynak',
        lambda at, r: found_and_cited(at, r) and r["answer"].startswith("No"),
    ),
    Case(
        "Müşteri, belgelerde yok (u01)",
        "Who is the CEO of Lumora Audio?",
        'mavi "bulamadım" kutusu',
        lambda at, r: r is not None and not r["found"] and "couldn't find" in at.info[0].value,
    ),
    Case(
        "Müşteri, cevap iç belgede (h01)",
        "What margin do Platinum resellers get?",
        '"bulamadım", hiç iç kaynak yok',
        lambda at, r: (
            r is not None and not r["found"] and all(s["audience"] == "public" for s in r["sources"])
        ),
    ),
    Case(
        "Destek, yanlış anahtar (i01)",
        "A customer wants a $320 refund. Who is allowed to approve it?",
        "kırmızı kutu: anahtar yanlış (401)",
        lambda at, r: r is None and "agent key is wrong" in at.error[0].value,
        agent=True,
        key="wrong",
    ),
    Case(
        "Destek, doğru anahtar (i01)",
        "A customer wants a $320 refund. Who is allowed to approve it?",
        'Senior Support Specialist + "internal" etiketi',
        lambda at, r: (
            found_and_cited(at, r)
            and "Senior Support Specialist" in r["answer"]
            and ":red-badge[internal]" in markdown(at)
        ),
        agent=True,
        key="right",
    ),
    Case(
        "API kapalı",
        "How do I pair my Pulse earbuds?",
        "kırmızı kutu: API'ye ulaşılamıyor",
        lambda at, r: r is None and "Cannot reach the API" in at.error[0].value,
        down=True,
    ),
]


def run(case: Case) -> tuple[AppTest, dict | None, float]:
    original = ui.client.DEFAULT_URL
    if case.down:
        ui.client.DEFAULT_URL = DOWN_URL  # app.py reads it from the module on every run
    try:
        at = AppTest.from_file(APP, default_timeout=120).run()
        if case.agent:
            at.sidebar.radio[0].set_value("Support agent").run()
            key = settings.admin_api_key if case.key == "right" else "wrong-key-on-purpose"
            at.sidebar.text_input[0].input(key).run()
        started = time.perf_counter()
        at.chat_input[0].set_value(case.question).run()
        seconds = time.perf_counter() - started
    finally:
        ui.client.DEFAULT_URL = original
    last = at.session_state["messages"][-1]
    return at, last.get("response"), seconds


def main() -> None:
    rows = []
    for case in CASES:
        at, resp, seconds = run(case)
        ok = not at.exception and case.check(at, resp)
        rows.append((case, resp, seconds, ok))
        print(f"{'OK ' if ok else 'BAD'} {seconds:5.1f} s  {case.label}", flush=True)
        if not case.down and case.key != "wrong":
            time.sleep(4)  # Gemini free tier: stay under the per-minute limit
    REPORT.write_text("\n".join(report(rows)), encoding="utf-8-sig")
    print(f"Saved to {Path(REPORT).relative_to(PROJECT_ROOT)}")


def short(text: str, n: int = 150) -> str:
    one = " ".join(text.split()).replace("|", "/")
    return one if len(one) <= n else one[:n] + " …"


def rank(ranks: dict, key: str) -> str:
    return f"#{ranks[key]}" if key in ranks else "—"


def report(rows) -> list[str]:
    passed = sum(ok for *_, ok in rows)
    answered = [(c, r, s) for c, r, s, _ in rows if r is not None]
    L = [
        "# A12: Arayüz — gerçek API ile 8 durum",
        "",
        "> **Kısaca sonuç:**",
        f"> - 8 durumun **{passed}'i** ekranda beklenen şeyi gösterdi (aşağıdaki tablo).",
        "> - Arayüz kendi başına bir şey bilmiyor: her soruyu API'ye gönderiyor. Güvenlik kuralları",
        ">   (iç belge, anahtar) API'de olduğu için arayüz onları atlayamaz.",
        "> - Ekranda geçen sürenin neredeyse tamamı API'nin cevap süresi; arayüzün payı bölüm 1'de.",
        ">",
        "> Her satır: Streamlit'in kendi test aracı (AppTest) sayfanın kodunu tarayıcıdaki gibi çalıştırdı,",
        "> soruyu sohbet kutusuna yazdı, çalışan API'ye (gerçek veritabanı, gerçek Gemini) gitti. Ders:",
        "> `dersler/12-arayuz.md`. Betik: `scripts/a12_ui.py`.",
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        "| **Streamlit** | Python ile web sayfası yazma kütüphanesi. HTML/JavaScript yazmadan sohbet ekranı. |",
        "| **AppTest** | Streamlit'in test aracı: sayfayı tarayıcısız çalıştırır, butona basar, yazı yazar. |",
        "| **İnce istemci (thin client)** | Hiçbir iş yapmayan, sadece API'ye sorup cevabı çizen arayüz. |",
        "",
        "## 1. Sekiz durum",
        "",
        '> 👀 **Neye bak:** "Beklenen" ile "Sonuç" aynı mı? "Toplam" süre ile "API" süresi arasındaki fark',
        "> arayüzün kendi payı (sayfa kodunu çalıştırmak + HTTP). Gerçek tarayıcıda çizim süresi bunun üstüne",
        "> eklenir; onu ölçmedik.",
        "",
        "| Durum | Beklenen | Sonuç | Toplam | API | Cevap / ekrandaki mesaj |",
        "|---|---|---|---:|---:|---|",
    ]
    for case, resp, seconds, ok in rows:
        api = f"{resp['seconds']:.1f} sn" if resp else "—"
        shown = short(resp["answer"]) if resp else "(hata kutusu)"
        L.append(
            f"| {case.label} | {case.expected} | {'✅' if ok else '❌'} | {seconds:.1f} sn | {api} | {shown} |"
        )

    overhead = [s - r["seconds"] for _c, r, s in answered]
    if overhead:
        L += [
            "",
            (
                f"Arayüzün payı (toplam − API), cevap gelen {len(overhead)} durumda: en az "
                f"{min(overhead):.2f} sn, en çok {max(overhead):.2f} sn."
            ),
        ]

    L += [
        "",
        '## 2. "Neden bu kaynak?" panelinin gösterdiği sıralar',
        "",
        '> 👀 **Neye bak:** Cevabın kullandığı her kaynak hangi aramada kaçıncıydı? "—" = o arama bu parçayı',
        "> ilk 30'da hiç bulamadı. Bir aramanın kaçırdığını diğeri yakalıyorsa hybrid search işini yapıyor.",
        '> 2. deneme = evet ise anlam/kelime sırası asıl sorunun aramasından, "birleşik" sıra ise soru +',
        "> yeniden yazılmış sorguların birleşiminden (A10).",
        "",
        "| Durum | Kaynak | 2. deneme | Anlam (vektör) | Kelime (BM25) | Birleşik | Reranker |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for case, resp, _s in answered:
        retried = "evet" if resp["retried"] else "hayır"
        for src in resp["sources"]:
            if src["cited"]:
                r = src["ranks"]
                L.append(
                    f"| {case.label} | [{src['n']}] {src['location']} | {retried} | {rank(r, 'vector')} "
                    f"| {rank(r, 'keyword')} | {rank(r, 'hybrid')} | {rank(r, 'reranker')} |"
                )
    L.append("")
    return L


if __name__ == "__main__":
    main()
