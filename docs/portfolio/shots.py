"""Portfolio images for DocuMind: cover + 4 screens of the running app (docker compose up)."""

import sys
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[2]  # the project folder
OUT = ROOT / "docs" / "portfolio"
URL = "http://localhost:8501"
W, H = 1568, 698  # same size as the ScrapePilot screenshots
MODEL = "gemini-3.5-flash-lite"

sys.path.insert(0, str(ROOT))
from app.config import settings  # agent key for the agent screenshot; never printed

HIDE = """
[data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] {display:none !important;}
"""


def fresh(page: Page) -> None:
    page.goto(URL)
    page.wait_for_selector("text=API online", timeout=60_000)
    page.add_style_tag(content=HIDE)


def ask(page: Page, question: str) -> None:
    before = page.inner_text("body").count(MODEL) + page.inner_text("body").count("couldn't find")
    box = page.get_by_placeholder("Ask about Lumora")
    box.fill(question)
    box.press("Enter")
    # wait until one more answer (model name in its status line) or "not found" box is on the page
    page.wait_for_function(
        "([n, model]) => (document.body.innerText.split(model).length - 1)"
        " + (document.body.innerText.split(\"couldn't find\").length - 1) > n",
        arg=[before, MODEL],
        timeout=120_000,
    )
    page.wait_for_timeout(1500)
    page.add_style_tag(content=HIDE)


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")

        cover = browser.new_page(viewport={"width": 1000, "height": 800})
        cover.goto((OUT / "cover.html").as_uri())
        cover.locator(".card").screenshot(path=str(OUT / "00-cover.png"))

        ctx = browser.new_context(viewport={"width": W, "height": H}, color_scheme="dark")
        page = ctx.new_page()

        # 1. a cited answer with its source card (the 2025 vs 2026 return policy trap)
        fresh(page)
        page.get_by_role("button", name="I opened my Pulse Pro earbuds").click()
        page.wait_for_selector(f"text={MODEL}", timeout=120_000)
        page.wait_for_timeout(1500)
        page.add_style_tag(content=HIDE)
        page.get_by_test_id("stChatMessage").first.evaluate("e => e.scrollIntoView({block: 'start'})")
        page.mouse.move(900, 300)
        page.mouse.wheel(0, -90)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "01-cited-answer.png"))

        # 2. "why this source?": where the chunk ranked in each search step
        fresh(page)
        ask(page, "How much does Pulse Pro cost in the UK?")
        page.get_by_text("Why this source?").first.click()
        page.wait_for_timeout(800)
        page.get_by_text("Reranker (reads question and text together)").first.scroll_into_view_if_needed()
        page.mouse.wheel(0, 150)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "02-why-this-source.png"))

        # 3. a customer asks something only an internal document answers: honest "not found"
        fresh(page)
        ask(page, "What margin do Platinum resellers get?")
        page.screenshot(path=str(OUT / "03-customer-not-found.png"))

        # 4. the same question from a support agent with the key: internal source, red badge
        fresh(page)
        page.get_by_test_id("stRadioGroup").get_by_text("Support agent").click()
        page.get_by_test_id("stTextInputField").fill(settings.admin_api_key or "")
        page.get_by_test_id("stTextInputField").press("Enter")
        page.wait_for_timeout(1500)
        ask(page, "What margin do Platinum resellers get?")
        page.get_by_text("internal", exact=True).locator("visible=true").first.scroll_into_view_if_needed()
        page.mouse.wheel(0, 100)
        page.wait_for_timeout(500)
        page.screenshot(path=str(OUT / "04-agent-internal.png"))

        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
