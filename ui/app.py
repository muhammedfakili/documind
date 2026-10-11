"""A12: DocuMind's chat screen -- ask a question, read the answer, check every source.

Run it (the API must be running, see app/api.py):
  uv run streamlit run ui/app.py
then open http://localhost:8501.

Streamlit re-runs this whole file from top to bottom after every click or message.
Anything that must survive a re-run (the chat so far) lives in st.session_state.

Under each answer:
  sources used     the chunks the answer cites with [n], their file, page and date
  why this source  where the chunk stood in each search step (vector, keyword, reranker)
  everything retrieved  also the chunks the model read but did not cite
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import streamlit as st

# `streamlit run ui/app.py` puts ui/ on the import path, not the project folder; add it so
# `from ui...` works the same here as in the tests.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ui.client import DEFAULT_URL, ApiClient, ApiError
from ui.format import (
    answer_markdown,
    escape_markdown,
    excerpt,
    highlight,
    question_terms,
    rank_lines,
    source_meta,
    status_line,
)

CUSTOMER_EXAMPLES = [
    "I opened my Pulse Pro earbuds 20 days ago. Can I still return them?",
    "How much does Pulse Pro cost in the UK?",
    "Can I order Lumora products to Japan?",
    "Who is the CEO of Lumora Audio?",  # not in the documents: shows the honest "not found"
]
AGENT_EXAMPLES = [
    "A customer wants a $320 refund. Who is allowed to approve it?",
    "A customer says their Halo ANC got very hot while charging. What should I do?",
]

# What the user should do about each API error (the API's own text is shown below it).
ERROR_HINTS = {
    None: "The API is not reachable.",
    401: "The agent key is wrong. Check it in the sidebar.",
    403: "Agent mode is switched off on the server (ADMIN_API_KEY is not set).",
    422: "The question could not be processed.",
    502: "The language model is not answering right now. Try again in a minute.",
    503: "A part of the system is down.",
}

st.set_page_config(page_title="DocuMind", page_icon="📚")


# --- sidebar: who is asking, how many sources, is the API up -------------------------


@st.cache_data(ttl=15, show_spinner=False)
def api_health(base_url: str) -> tuple[bool, str]:
    """Checked at most every 15 s: the sidebar re-draws on every click."""
    try:
        chunks = ApiClient(base_url).health()["chunks"]
    except ApiError as exc:
        return False, exc.message
    total = sum(chunks.values())
    parts = ", ".join(f"{n} {audience}" for audience, n in sorted(chunks.items()))
    return True, f"{total} chunks ({parts})"


with st.sidebar:
    st.title("📚 DocuMind")
    st.caption("Answers from Lumora Audio's documents, with the sources to check them.")
    role = st.radio("Who is asking?", ["Customer", "Support agent"], horizontal=True)
    agent = role == "Support agent"
    api_key = None
    if agent:
        api_key = st.text_input(
            "Agent key", type="password", help="Sent as X-API-Key. Unlocks internal documents."
        )
        st.caption("Agents also see internal documents (policies, known issues, reseller terms).")
    k = st.slider("Sources the model reads", 1, 10, 5, help="k: how many chunks go into the prompt.")

    ok, message = api_health(DEFAULT_URL)
    (st.success if ok else st.error)(
        f"API {'online' if ok else 'offline'}: {message}", icon="🟢" if ok else "🔴"
    )
    if st.button("Clear chat", width="stretch"):
        st.session_state.messages = []
    st.caption(f"API: {DEFAULT_URL}")


# --- drawing one answer ----------------------------------------------------------------


def source_card(source: dict[str, Any], terms: list[str]) -> None:
    with st.container(border=True):
        badge = ":red-badge[internal]" if source["audience"] == "internal" else ""
        st.markdown(f":blue-badge[{source['n']}] **{source['title']}** {badge}")
        st.caption(source_meta(source))
        st.markdown(highlight(excerpt(source["text"]), terms))
        with st.expander("Why this source?"):
            lines = rank_lines(source["ranks"])
            if lines:
                st.markdown("\n".join(f"- {line}" for line in lines))
                st.caption(
                    "Lower is better: #1 = first place. Both searches are fused, then the reranker "
                    "re-reads the top 10 and the two orders are fused again."
                )
            st.caption("Highlighted: words from your question. Meaning search also finds text without them.")
            if len(source["text"]) > len(excerpt(source["text"])):
                st.markdown("**Full text**")
                st.markdown(highlight(source["text"], terms))


def show_answer(response: dict[str, Any]) -> None:
    if response["found"]:
        st.markdown(answer_markdown(response["answer"]))
    else:
        st.info(response["answer"] + " Try other words, or ask a support agent.", icon="🔎")
    st.caption(status_line(response))

    sources = response["sources"]
    if not sources:
        return
    terms = question_terms(response["question"])
    cited = [s for s in sources if s["cited"]]
    tab_used, tab_all = st.tabs([f"Sources used ({len(cited)})", f"Everything retrieved ({len(sources)})"])
    with tab_used:
        if not cited:
            st.caption("The answer cites no source.")
        for source in cited:
            source_card(source, terms)
    with tab_all:
        st.caption("All chunks the model read, in search order. Uncited ones were judged not relevant.")
        for source in sources:
            source_card(source, terms)


def show_error(status: int | None, message: str) -> None:
    st.error(f"{ERROR_HINTS.get(status, 'Something went wrong.')}\n\n`{status or 'no response'}` {message}")


# --- the chat --------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# An example button stores its question and re-runs the page, so it arrives here like typed text.
question = st.chat_input("Ask about Lumora products, orders, returns, warranty ...") or st.session_state.pop(
    "pending", None
)

if not st.session_state.messages and not question:
    st.markdown("### Ask Lumora Audio's documents")
    st.markdown(
        "Every answer cites its sources as :blue-badge[1] badges. Open a source to see where it "
        "comes from and why the search picked it. If the documents do not say, DocuMind says so."
    )
    examples = AGENT_EXAMPLES + CUSTOMER_EXAMPLES[:2] if agent else CUSTOMER_EXAMPLES
    st.caption("Try one:")
    for i, example in enumerate(examples):
        if st.button(example, key=f"example-{i}"):
            st.session_state.pending = example
            st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(escape_markdown(msg["text"]))
        elif "error" in msg:
            show_error(msg["status"], msg["error"])
        else:
            show_answer(msg["response"])

if question:
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(escape_markdown(question))
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching the documents ..."):
                response = ApiClient(DEFAULT_URL, api_key=api_key or None).ask(
                    question, audience="all" if agent else "public", k=k
                )
        except ApiError as exc:
            st.session_state.messages.append(
                {"role": "assistant", "status": exc.status, "error": exc.message}
            )
            show_error(exc.status, exc.message)
        else:
            st.session_state.messages.append({"role": "assistant", "response": response})
            show_answer(response)
