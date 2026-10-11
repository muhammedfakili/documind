"""A6: the first end-to-end RAG answers -- and a check of every one of them.

Output: eval/results/a6_answers.md (explained report, Turkish)

Two experiments:
  1. A0 rematch   the four questions the bare LLM hallucinated in A0, now with RAG
  2. checklist    16 questions with a known right answer (or a known "not found"):
                  is the fact right, is the right source cited, does it refuse correctly?

Usage (from the project root, database running: docker compose up -d):
  uv run python -m scripts.a6_answer
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from datetime import datetime

from app import store
from app.config import PROJECT_ROOT
from app.generate import NOT_FOUND, SYSTEM_PROMPT, Answer, answer
from app.llm import get_llm

REPORT = PROJECT_ROOT / "eval" / "results" / "a6_answers.md"
PAUSE = 6  # seconds between LLM calls: the Gemini free tier allows ~10 per minute

# --- Experiment 1: the A0 questions ----------------------------------------

A0_QUESTIONS = [
    (
        "What is Lumora Audio's return window for headphones that have been opened?",
        "30 gün",
        "14 gün (2026 politikası; 2025'te 30 gündü)",
    ),
    (
        "How long is the warranty on the Lumora Aria X2 speaker, and does it cover water damage?",
        "1 yıl; su hasarı kapsanmaz",
        "ABD/AB'de 24 ay; IP67 sınırları içinde tatlı su kapsanır, tuzlu/klorlu su kapsanmaz",
    ),
    (
        "What does error code E-214 mean on Lumora Pulse earbuds, and how do I fix it?",
        "Bağlantı/eşleşme sorunu; sıfırla, yeniden eşle",
        "SAĞ kulaklıkta batarya sıcaklık sensörü arızası; 20 dk dinlendir, firmware 3.2.1'e güncelle",
    ),
    (
        "In Lumora's support team, who has to approve a refund above $500?",
        "Supervisor veya Team Lead",
        "Customer Operations Manager (🔒 iç belge: müşteriye söylenmemeli)",
    ),
]

# --- Experiment 2: checklist --------------------------------------------------


@dataclass(frozen=True)
class Case:
    question: str
    audience: store.Audience
    expect: str | None  # regex the answer must match; None = the right reply is NOT_FOUND
    needle: str | None = None  # text only the right source contains (for the citation check)
    kind: str = "public"  # public | internal | unanswerable


CASES = [
    Case("What does error code E-214 mean on the Pulse earbuds?", "public",
         r"temperature sensor", "Battery temperature sensor fault"),
    Case("What does E-214 mean on Pulse Pro?", "public",
         r"calibrat", "ANC microphone calibration failed"),
    Case("How long do I have to return earbuds I already opened?", "public",
         r"14 days", "**14 days**"),
    Case("How much is express delivery to the UK?", "public", r"£9\.99", "£9.99"),
    Case("How many Aria X2 speakers can play together in Party Mode?", "public",
         r"\b8\b|eight", "up to 8 Aria X2"),
    Case("Is water damage covered for Halo headphones?", "public",
         r"never covered|not covered|no ip rating", "never covered"),
    Case("Can I still buy Halo Lite?", "public", r"discontinued", "discontinued"),
    Case("How long is the warranty on the Aria X2 in the US?", "public",
         r"24 months|two years|2 years", "Warranty US region (months): 24"),
    Case("Who has to approve a refund of more than 500 dollars?", "public", None, kind="internal"),
    Case("Who has to approve a refund of more than 500 dollars?", "all",
         r"Customer Operations Manager", "Above $500", kind="internal"),
    Case("What margin does a Gold reseller get?", "public", None, kind="internal"),
    Case("What margin does a Gold reseller get?", "all", r"35\s?%", "Margin on MSRP: 35%", kind="internal"),
    Case("Who is the CEO of Lumora Audio?", "public", None, kind="unanswerable"),
    Case("Who is the CEO of Lumora Audio?", "all", None, kind="unanswerable"),
    Case("Is there a Black Friday sale this year?", "public", None, kind="unanswerable"),
    Case("Do you have a trade-in program for old headphones?", "public", None, kind="unanswerable"),
]

MODE = {"public": "🙋 müşteri", "all": "🧑‍💼 destek"}


@dataclass
class Check:
    case: Case
    result: Answer
    seconds: float

    @property
    def retrieved(self) -> bool | None:
        """Was the right chunk among the 5 the LLM got? (None = nothing to find)"""
        if self.case.needle is None:
            return None
        return any(self.case.needle.lower() in h.text.lower() for h in self.result.sources)

    @property
    def cited_right(self) -> bool | None:
        if self.case.needle is None:
            return None
        return any(self.case.needle.lower() in h.text.lower() for _, h in self.result.cited_sources)

    @property
    def correct(self) -> bool:
        if self.case.expect is None:
            return not self.result.found
        return self.result.found and re.search(self.case.expect, self.result.text, re.IGNORECASE) is not None


def ask(conn, llm, question: str, audience: store.Audience) -> tuple[Answer, float]:
    started = time.perf_counter()
    result = answer(conn, question, audience=audience, llm=llm)
    seconds = time.perf_counter() - started
    print(f"\n[{audience}] {question}\n{result.text}")
    time.sleep(PAUSE)
    return result, seconds


# --- report helpers -----------------------------------------------------------


def quote(text: str) -> list[str]:
    return ["> " + line if line.strip() else ">" for line in text.strip().splitlines()]


def source_list(result: Answer) -> list[str]:
    if not result.found:
        return ["*Kaynak yok: sistem \"bulamadım\" dedi.*"]
    lines = []
    for n, h in result.cited_sources:
        lock = "🔒 " if h.audience == "internal" else ""
        section = f" › {h.section}" if h.section else ""
        lines.append(f"- **[{n}]** {lock}`{h.location}`{section}")
    unused = len(result.sources) - len(result.cited)
    if unused:
        lines.append(f"- *(+{unused} parça daha getirildi ama cevapta kullanılmadı)*")
    return lines


def yes_no(value: bool | None) -> str:
    return "—" if value is None else ("✅" if value else "❌")


def short(text: str, n: int = 110) -> str:
    one = " ".join(text.split()).replace("|", "/")
    return one if len(one) <= n else one[:n] + " …"


def main() -> None:
    llm = get_llm()
    conn = store.connect()
    if not store.count(conn):
        raise SystemExit("The chunks table is empty: run `uv run python -m scripts.a5_store` first.")

    # Experiment 1 -------------------------------------------------------------
    rematch = []
    for question, _, _ in A0_QUESTIONS:
        rematch.append(("public", question, *ask(conn, llm, question, "public")))
    refund_q = A0_QUESTIONS[3][0]
    rematch.append(("all", refund_q, *ask(conn, llm, refund_q, "all")))

    # Experiment 2 -------------------------------------------------------------
    checks = [Check(c, *ask(conn, llm, c.question, c.audience)) for c in CASES]

    # Numbers ------------------------------------------------------------------
    facts = [c for c in checks if c.case.expect is not None]
    refusals = [c for c in checks if c.case.expect is None]
    all_answers = [r for *_, r, _ in rematch] + [c.result for c in checks]
    all_seconds = [s for *_, s in rematch] + [c.seconds for c in checks]
    n_fact_ok = sum(c.correct for c in facts)
    n_cited_ok = sum(bool(c.cited_right) for c in facts)
    n_retrieved = sum(bool(c.retrieved) for c in facts)
    n_refuse_ok = sum(c.correct for c in refusals)
    leaks = sum(
        1 for c in checks if c.case.audience == "public"
        for _, h in c.result.cited_sources if h.audience == "internal"
    )
    found = [a for a in all_answers if a.found]
    uncited = sum(1 for a in found if not a.cited)
    avg_in = sum(a.input_tokens for a in all_answers) / len(all_answers)
    avg_out = sum(a.output_tokens for a in all_answers) / len(all_answers)
    avg_s = sum(all_seconds) / len(all_seconds)

    # Report -------------------------------------------------------------------
    L = [
        "# A6 — İlk uçtan uca RAG cevapları",
        "",
        "> **Bu rapor ne?** Sistem ilk defa soruya **cevap** verdi: soruya en yakın 5 parçayı buldu,",
        "> numaralayıp LLM'e verdi, LLM de sadece onlara bakarak **kaynak numaralı** bir cevap yazdı.",
        "> İki şeyi kontrol ettik: A0'da uydurulan cevaplar artık doğru mu, ve cevabı bilinen 16 soruda",
        "> sistem doğru bilgiyi, doğru kaynağı gösterip veriyor mu, bilmediğinde \"bulamadım\" diyor mu.",
        (f"> Bu dosyayı `scripts/a6_answer.py` üretiyor ({llm.name} / {llm.model},"
        f" {datetime.now().astimezone():%Y-%m-%d %H:%M}). Ders anlatımı: `dersler/06-ilk-cevap.md`."),
        "",
        "## ✅ Kısaca sonuç",
        "",
        f"- **Doğru bilgi:** Cevabı belgelerde olan {len(facts)} sorunun **{n_fact_ok}** tanesinde cevap doğru.",
        (f"- **Doğru kaynak:** {len(facts)} sorunun **{n_cited_ok}** tanesinde cevap, doğru bilginin geçtiği parçayı"
        " `[n]` ile gösterdi."),
        (f"- **\"Bulamadım\" demesi gereken yerde:** {len(refusals)} sorunun **{n_refuse_ok}** tanesinde doğru"
        " şekilde \"bulamadım\" dedi (iç belge sorusu müşteri modunda + belgelerde hiç olmayan sorular)."),
        f"- **Sızıntı:** Müşteri modundaki cevaplarda iç belgeye verilen kaynak sayısı: **{leaks}**.",
        f"- **Kaynaksız cevap:** \"Bulamadım\" dışındaki {len(found)} cevabın **{uncited}** tanesinde hiç `[n]` yok.",
        (f"- **Maliyet:** Bir cevap ortalama **{avg_in:.0f}** token okuyup **{avg_out:.0f}** token yazdı,"
        f" ortalama **{avg_s:.1f} sn** sürdü (arama + LLM)."),
        "",
        "## 📖 Terimler",
        "",
        "| Terim | Ne demek |",
        "|---|---|",
        "| **Uçtan uca (end-to-end)** | Soru girdi, cevap çıktı. Arama + cevap yazma, ikisi birden. |",
        ("| **Kaynak / atıf `[n]`** | Cümlenin hangi parçadan geldiğini gösteren numara."
        " `[1]` = LLM'e verilen 1. parça. |"),
        "| **Grounding (temellendirme)** | LLM'in sadece verilen parçalara dayanması, kendi bildiğini katmaması. |",
        (f"| **\"Bulamadım\" cümlesi** | Cevap yoksa LLM'in yazdığı sabit cümle: *{NOT_FOUND}*"
        " Kod bunu tanıyıp \"cevap yok\" olarak işaretliyor. |"),
        "| **Bilgi doğru** | Cevapta beklenen bilgi var mı (ör. \"14 days\", \"£9.99\")? |",
        "| **Kaynak doğru** | Cevabın `[n]` ile gösterdiği parçalardan biri, doğru bilginin geçtiği parça mı? |",
        ("| **Getirildi mi** | Doğru parça LLM'e verilen 5 parçanın içinde var mıydı?"
        " Yoksa hata aramada, varsa hata cevap yazmada. |"),
        "| **Token** | LLM'in okuduğu/yazdığı metin birimi (~¾ kelime). Ücretli modellerde fatura buna göre. |",
        "",
        "## 1. A0'ın rövanşı: aynı 4 soru, bu sefer belgelerle",
        "",
        "A0'da LLM'e belge vermeden sormuştuk ve kendinden emin şekilde **uydurmuştu**. Şimdi aynı soruları",
        "aynı modele soruyoruz. Tek fark: önüne bizim bulduğumuz 5 parça konuyor.",
        "",
        "| Soru | A0'da uydurulan | Belgelerdeki doğru cevap |",
        "|---|---|---|",
    ]
    for question, invented, truth in A0_QUESTIONS:
        L.append(f"| {short(question, 80)} | ❌ {invented} | {truth} |")
    L.append("")
    for audience, question, result, seconds in rematch:
        L += [f"### {MODE[audience]} modu — \"{question}\"", ""]
        L += quote(result.text)
        L += [""] + source_list(result) + ["", f"*{seconds:.1f} sn, {result.input_tokens} token okundu.*", ""]
    L += [
        "👀 **Neye bak:**",
        ("- Her cümlenin sonunda bir `[n]` var. Destek ekibi o numaraya tıklayıp cümleyi belgeyle karşılaştırabilir."
        " A0'daki cevaplarda bunu yapmanın yolu yoktu."),
        ("- E-214 cevabında **Pulse Pro'nun** E-214'ü (ANC kalibrasyonu) kullanılmadı, oysa o parça da LLM'e"
        " verildi (5 parçanın 3.'süydü). Prompt'taki \"sadece sorulan ürünün kaynağını kullan\""
        " kuralı burada işe yarıyor."),
        ("- \\$500 iade sorusu: **müşteri modunda** sistem \"bulamadım\" diyor çünkü cevap iç belgede ve müşteri"
        " o belgeyi hiç göremiyor. **Destek modunda** aynı soru doğru cevabı 🔒 iç belgeyi göstererek veriyor."),
        ("- İade sorusunda LLM'e **2025 politikası** (30 gün) da verildi ama cevap 2026'yı (14 gün) seçti, çünkü"
         " 2026 belgesi \"eskisinin yerine geçer\" diyor. Bu sefer işe yaradı ama şansa bağlı; A10'da tarih"
         " etiketiyle bunu garantiye alacağız."),
        "",
        "## 2. Kontrol listesi: 16 soru",
        "",
        ("Her sorunun doğru cevabını (ya da \"belgelerde yok\" olduğunu) önceden biliyoruz. Kod her cevabı"
        " otomatik kontrol ediyor."),
        "",
        "| # | Soru | Mod | Tür | Getirildi mi | Bilgi doğru | Kaynak doğru | Cevap (kısa) |",
        "|---:|---|---|---|:-:|:-:|:-:|---|",
    ]
    kinds = {"public": "herkese açık", "internal": "🔒 iç belge", "unanswerable": "❓ belgelerde yok"}
    for i, c in enumerate(checks, start=1):
        verdict = "✅" if c.correct else "❌"
        L.append(
            f"| {i} | {c.case.question} | {MODE[c.case.audience]} | {kinds[c.case.kind]} |"
            f" {yes_no(c.retrieved)} | {verdict} | {yes_no(c.cited_right)} | {short(c.result.text)} |"
        )
    wrong = [c for c in checks if not c.correct]
    L += [
        "",
        "**Nasıl okunur:**",
        "- **Getirildi mi** ❌ ise LLM'in suçu yok: doğru parça önüne hiç gelmedi (arama hatası, A8/A9'un işi).",
        ("- **Getirildi mi** ✅ ama **Bilgi doğru** ❌ ise parça oradaydı ama LLM yanlış okudu/yanlış parçayı"
        " seçti (cevap yazma hatası)."),
        "- \"Bulamadım\" sorularında doğru davranış cevap **vermemek**; o yüzden kaynak sütunları `—`.",
        "",
        '### ⚠️ Bu 10/10 ne demek, ne demek değil',
        '',
        '- **Sıra artık o kadar önemli değil:** Pulse E-214 parçası aramada **4.** sıradaydı (A5). Yine de LLM önündeki',
        "  5 parçayı okuyup doğru olanı seçti. Doğru parça **ilk 5'in içinde** olduğu sürece LLM işi tamamlıyor.",
        "  Asıl tehlike, doğru parçanın ilk 5'e **hiç girmemesi**.",
        "- **Kolay sınav:** Bu 16 soru kısa, tek belgelik sorular ve bir kısmını A4'ten beri biliyoruz. Gerçek kullanıcılar",
        "  birden fazla belgeye yayılan, farklı kelimelerle yazılmış sorular sorar. A7'de 40-50 soruluk zor bir sınav",
        '  hazırlayıp **asıl** başlangıç rakamını orada ölçeceğiz.',
        '- **Kontrol kaba:** "Bilgi doğru" sadece beklenen ifadenin cevapta geçip geçmediğine bakıyor. Cevabın geri kalanı',
        "  yanlış olsa bunu yakalamaz. A7'de cevapları bir LLM'e notlandıracağız (LLM-as-judge).",
        "",
    ]
    if wrong:
        L += ["### ❌ Yanlış çıkan cevaplar (tam metin)", ""]
        for c in wrong:
            L += [f"**{c.case.question}** ({MODE[c.case.audience]} modu)", ""]
            L += quote(c.result.text)
            L += [""] + source_list(c.result) + [""]
    L += [
        "## 3. LLM'e giden prompt",
        "",
        "Sistem talimatı (her soruda aynı):",
        "",
        "```",
        SYSTEM_PROMPT,
        "```",
        "",
        ("Kullanıcı mesajı: `Sources:` başlığı altında 5 parça, her biri `[n] (kaynak, sayfa)` + metin;"
        " en sonda `Question: ...`."),
        "",
        "## 🧪 Testler ne kontrol ediyor (`tests/test_generate.py`)",
        "",
        ("Testler **sahte bir LLM** ve sahte bir arama kullanıyor: veritabanı ve API anahtarı gerekmiyor,"
        " saniyenin altında bitiyor."),
        "",
        "| Test | Neyi garanti ediyor |",
        "|---|---|",
        ("| `test_prompt_numbers_sources_with_location` | Parçalar `[1]`, `[2]` diye numaralanıyor ve yanında"
        " kaynak/sayfa yazıyor. |"),
        "| `test_citation_formats` | `[1]`, `[1][2]`, `[3, 1]` yazımlarının hepsi okunuyor. |",
        ("| `test_citation_to_missing_source_is_dropped` | LLM `[7]` uydurursa (5 kaynak varken) kullanıcıya"
        " gösterilmiyor. |"),
        ("| `test_not_found_detection_tolerates_small_differences` | \"Bulamadım\" cümlesi küçük farklarla da"
        " tanınıyor, ama cevabın içinde geçmesi \"bulamadım\" sayılmıyor. |"),
        "| `test_answer_links_citations_to_sources` | `[1]` doğru parçaya bağlanıyor, LLM bütün parçaları görüyor. |",
        "| `test_not_found_answer_has_no_sources_cited` | \"Bulamadım\" cevabının altında kaynak listesi çıkmıyor. |",
        "| `test_empty_search_skips_the_llm` | Arama boş dönerse LLM hiç çağrılmıyor (boşuna para yok). |",
        "",
    ]
    REPORT.write_text("\n".join(L), encoding="utf-8-sig")
    print(f"\nfacts {n_fact_ok}/{len(facts)}  cited {n_cited_ok}/{len(facts)}  retrieved {n_retrieved}/{len(facts)}"
          f"  refusals {n_refuse_ok}/{len(refusals)}  leaks {leaks}  uncited {uncited}")
    print(f"Saved to {REPORT.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
