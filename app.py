"""
Argus — Market & Competitive Intelligence  (single-process Streamlit app)

Everything runs in-process: staging sources, the multi-agent pipeline, the RAG
index for "Ask Argus", and email sending. No separate backend — so it deploys as
a single app on Streamlit Community Cloud.

Run locally:
    streamlit run app.py

Configuration (GROQ_API_KEY, SMTP_*, EMBEDDING_MODEL, RAG_BACKEND) comes from the
environment / a local .env, or — on Streamlit Cloud — from the app's Secrets.
"""
from __future__ import annotations

import os

import streamlit as st

# Bridge Streamlit Cloud secrets into the environment BEFORE importing config,
# because config.Settings reads env vars at import time. Reading st.secrets here
# is safe (it is not a page-rendering command).
try:
    for _k in st.secrets:
        _v = st.secrets[_k]
        if isinstance(_v, (str, int, float, bool)):
            os.environ.setdefault(_k, str(_v))
except Exception:  # no secrets configured (e.g. local dev) -> ignore
    pass

import pandas as pd
import plotly.express as px

from agents import EmailAgent, QAAgent
from config import APP_DESCRIPTION, APP_NAME, APP_TAGLINE, Settings
from orchestrator import Orchestrator
from utils.rag import build_index

# --------------------------------------------------------------------------
# Page config + light theming
# --------------------------------------------------------------------------
st.set_page_config(
    page_title=f"{APP_NAME} — Market Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 2rem; max-width: 1200px;}
    h1, h2, h3 {letter-spacing:-0.01em;}
    .stAlert {border-radius:10px;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session state
# --------------------------------------------------------------------------
if "sources" not in st.session_state:
    st.session_state.sources = []          # staged source dicts
if "result" not in st.session_state:
    st.session_state.result = None         # last PipelineResult
if "rag_index" not in st.session_state:
    st.session_state.rag_index = None      # RAG index for the last run
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0      # bump to reset the file uploaders
if "chat" not in st.session_state:
    st.session_state.chat = []             # Ask Argus history

max_sentences = 5  # default summary length


# --------------------------------------------------------------------------
# Pipeline (in-process)
# --------------------------------------------------------------------------
def run_pipeline(sources: list, **opts):
    """Run the multi-agent pipeline and build the RAG index for Ask Argus."""
    orch = Orchestrator(Settings())
    res = orch.run(sources, **opts)
    st.session_state.rag_index = build_index(
        [{"title": d.title, "text": d.text or ""} for d in res.documents]
    )
    st.session_state.chat = []  # new run -> fresh conversation
    return res


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title(f"🛰️ {APP_NAME}")
st.markdown(f"### {APP_TAGLINE}")
st.markdown(
    f"{APP_DESCRIPTION} Feed **competitor pages, market reports (PDF), metrics "
    "(CSV), and news APIs** — a team of coordinated agents **summarizes, "
    "analyzes the landscape, and acts**, producing a market brief, a stakeholder "
    "email, and a live dashboard."
)

tab_sources, tab_report, tab_dash, tab_ask, tab_email = st.tabs(
    ["📥 Signals", "📄 Brief", "📊 Dashboard", "💬 Ask", "✉️ Email"]
)

# --------------------------------------------------------------------------
# TAB: Sources
# --------------------------------------------------------------------------
with tab_sources:
    st.subheader("1 · Add market signals")

    def _stage_uploads(files, kind: str) -> None:
        """Auto-stage uploaded files (no extra click), deduped by name."""
        existing = {(s["kind"], s["title"]) for s in st.session_state.sources
                    if "bytes" in s}
        for f in files or []:
            key = (kind, f.name)
            if key not in existing:
                st.session_state.sources.append(
                    {"kind": kind, "bytes": f.getvalue(), "title": f.name}
                )
                existing.add(key)

    def _stage_url(url: str, kind: str, title: str) -> None:
        """Auto-stage a typed URL (on Enter/blur), deduped by value."""
        url = (url or "").strip()
        if not url:
            return
        if url not in {s.get("url") for s in st.session_state.sources
                       if s["kind"] == kind}:
            st.session_state.sources.append({"kind": kind, "url": url, "title": title})

    def _stage_text(text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        if text not in {s.get("text") for s in st.session_state.sources
                        if s["kind"] == "text"}:
            st.session_state.sources.append(
                {"kind": "text", "text": text, "title": "Pasted text"})

    st.caption("Add anything below — uploads and typed URLs/text stage "
               "automatically (press Enter after typing a URL).")

    if st.button("📦 Load sample data", help="Loads the bundled samples/ files + a "
                 "demo web-page URL and note, ready to run."):
        base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
        titles = {s.get("title") for s in st.session_state.sources}
        notes = ""
        try:
            if "market_report.pdf" not in titles:
                with open(os.path.join(base, "market_report.pdf"), "rb") as f:
                    st.session_state.sources.append(
                        {"kind": "pdf", "bytes": f.read(), "title": "market_report.pdf"})
            if "sales_metrics.csv" not in titles:
                with open(os.path.join(base, "sales_metrics.csv"), "rb") as f:
                    st.session_state.sources.append(
                        {"kind": "csv", "bytes": f.read(), "title": "sales_metrics.csv"})
            with open(os.path.join(base, "analyst_notes.txt"), encoding="utf-8") as f:
                notes = f.read()
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not load samples: {e}")
        # reset the input widgets to fresh keys, then pre-fill the URL + text boxes
        nk = st.session_state.uploader_key + 1
        st.session_state.uploader_key = nk
        st.session_state[f"web_{nk}"] = "https://en.wikipedia.org/wiki/Cloud_computing"
        if notes:
            st.session_state[f"text_{nk}"] = notes
        st.rerun()

    uk = st.session_state.uploader_key
    c1, c2 = st.columns(2)

    with c1:
        up_pdfs = st.file_uploader("Market report(s) — PDF", type=["pdf"],
                                   accept_multiple_files=True, key=f"pdf_{uk}")
        _stage_uploads(up_pdfs, "pdf")

        up_csvs = st.file_uploader("Metrics / sales — CSV/TSV", type=["csv", "tsv"],
                                   accept_multiple_files=True, key=f"csv_{uk}")
        _stage_uploads(up_csvs, "csv")

        sheet_url = st.text_input("Public Google Sheet URL", key=f"sheet_{uk}")
        _stage_url(sheet_url, "sheet", "Google Sheet")

    with c2:
        web_url = st.text_input("Competitor / market web page URL", key=f"web_{uk}")
        _stage_url(web_url, "web", web_url)

        api_url = st.text_input("News / market API URL (returns JSON)", key=f"api_{uk}")
        _stage_url(api_url, "api", f"API: {api_url[:30]}")

        paste = st.text_area("Paste raw text (notes, transcripts, snippets)",
                             height=90, key=f"text_{uk}")
        _stage_text(paste)

    st.divider()
    n_sources = len(st.session_state.sources)
    if n_sources:
        cs1, cs2 = st.columns([3, 1])
        cs1.success(f"✅ {n_sources} source(s) staged — ready to run.")
        if cs2.button("🗑️ Clear all"):
            st.session_state.sources = []
            st.session_state.result = None
            st.session_state.rag_index = None
            st.session_state.uploader_key += 1   # reset the file uploaders
            st.rerun()
    else:
        st.info("Add at least one source above — a single file is enough to run.")

    st.divider()
    st.subheader("2 · Configure & run")
    report_title = st.text_input("Brief title", "Market Intelligence Brief")
    objective = st.text_input(
        "Intelligence objective (optional)",
        "Assess the competitive landscape: key players, positioning, momentum, "
        "opportunities, and threats",
    )
    a1, a2, a3 = st.columns(3)
    do_report = a1.checkbox("Generate brief", True)
    do_email = a2.checkbox("Draft email", True)
    do_dashboard = a3.checkbox("Build dashboard", True)
    email_recipient = st.text_input("Email recipient (label)", "the team")

    run = st.button(f"🛰️ Run {APP_NAME}", type="primary", use_container_width=True,
                    disabled=not st.session_state.sources)

    if run:
        try:
            with st.spinner("Analyzing your market signals…"):
                st.session_state.result = run_pipeline(
                    st.session_state.sources,
                    report_title=report_title,
                    objective=objective,
                    max_sentences=max_sentences,
                    do_report=do_report,
                    do_email=do_email,
                    do_dashboard=do_dashboard,
                    email_recipient=email_recipient,
                )
            st.success("Done — see the Brief, Dashboard, Ask, and Email tabs.")
        except Exception as e:  # noqa: BLE001
            st.error(f"Run failed: {e}")


result = st.session_state.result

# --------------------------------------------------------------------------
# TAB: Brief
# --------------------------------------------------------------------------
with tab_report:
    report = result.report if result else None
    if report and report.get("markdown"):
        st.subheader("Market intelligence brief")

        analysis = (result.analysis if result else None) or {}
        if analysis:
            sentiment = analysis.get("overall_sentiment", "neutral")
            color = {"positive": "🟢", "negative": "🔴"}.get(sentiment, "🟡")
            s1, s2 = st.columns(2)
            s1.metric("Market sentiment", f"{color} {sentiment}")
            s2.metric("Confidence", analysis.get("confidence", "n/a"))

        md = report.get("markdown", "")
        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download Markdown", md, file_name="market_brief.md",
                           mime="text/markdown", use_container_width=True)
        if report.get("pdf_bytes"):
            d2.download_button("⬇️ Download PDF", report["pdf_bytes"],
                               file_name="market_brief.pdf", mime="application/pdf",
                               use_container_width=True)
        st.markdown("---")
        st.markdown(md)
    else:
        st.info("Run Argus to generate a market brief.")

# --------------------------------------------------------------------------
# TAB: Dashboard
# --------------------------------------------------------------------------
with tab_dash:
    d = result.dashboard if result else None
    if d and d.get("metrics"):
        m = d["metrics"]
        sentiment = m.get("sentiment", "neutral")
        color = {"positive": "🟢", "negative": "🔴"}.get(sentiment, "🟡")
        st.subheader("Market overview")
        cols = st.columns(3)
        cols[0].metric("Signals analyzed", m["sources"])
        cols[1].metric("Market sentiment", f"{color} {sentiment}")
        cols[2].metric("Confidence", m.get("confidence", "n/a"))

        g1, g2 = st.columns(2)
        with g1:
            if d.get("source_type_counts"):
                fig = px.pie(
                    names=list(d["source_type_counts"].keys()),
                    values=list(d["source_type_counts"].values()),
                    title="Signals by type", hole=0.45,
                )
                st.plotly_chart(fig, use_container_width=True)
        with g2:
            if d.get("top_keywords"):
                st.markdown("#### 🔥 Trending topics")
                st.write(" ".join(f"`{k}`" for k in d["top_keywords"]))

        # chart any numeric columns from an uploaded metrics/sales table
        for t in d.get("tables", []):
            df = t.get("dataframe")
            if isinstance(df, pd.DataFrame):
                num = df.select_dtypes("number")
                if not num.empty:
                    st.markdown(f"#### 📊 {t['title']}")
                    fig = px.line(num.reset_index(), x="index", y=list(num.columns),
                                  markers=True, title=t["title"])
                    st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Run Argus to build the market dashboard.")

# --------------------------------------------------------------------------
# TAB: Ask Argus  (RAG Q&A grounded in the run's sources)
# --------------------------------------------------------------------------
with tab_ask:
    index = st.session_state.rag_index
    if index is None or getattr(index, "size", 0) == 0:
        st.info("Run Argus first, then ask questions about your market signals.")
    else:
        st.caption("Ask anything about the sources you just analyzed — answers are "
                   "retrieved from and grounded in those sources (RAG).")

        def _render_sources(sources: list, retriever: str = "") -> None:
            if not sources:
                return
            label = f"🔎 Retrieved {len(sources)} passage(s)"
            if retriever:
                label += f" · {retriever}"
            with st.expander(label):
                for s in sources:
                    st.markdown(
                        f"**{s['title']}**  ·  relevance {s['score']}\n\n"
                        f"> {s['excerpt']}"
                    )

        for msg in st.session_state.chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant":
                    _render_sources(msg.get("sources", []), msg.get("retriever", ""))

        question = st.chat_input("e.g. What are the biggest threats to our position?")
        if question:
            st.session_state.chat.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Retrieving & answering…"):
                    retrieved = index.retrieve(question, k=4)
                    if not retrieved:
                        # overview/vague question: fall back to a sample of chunks
                        retrieved = [(c, 0.0) for c in getattr(index, "chunks", [])[:4]]
                    context = [c for c, _ in retrieved]
                    data = QAAgent().answer(question, context)
                    answer = data.get("answer", "—")
                    retriever = getattr(index, "kind", "")
                    sources = [
                        {"title": c["title"], "score": round(score, 3),
                         "excerpt": c["text"][:280]}
                        for c, score in retrieved
                    ]
                st.markdown(answer)
                _render_sources(sources, retriever)
            st.session_state.chat.append(
                {"role": "assistant", "content": answer,
                 "sources": sources, "retriever": retriever}
            )

# --------------------------------------------------------------------------
# TAB: Email
# --------------------------------------------------------------------------
with tab_email:
    email = result.email if result else None
    if email:
        st.subheader("Stakeholder email")
        subject = st.text_input("Subject", email.get("subject", ""))
        body = st.text_area("Body", email.get("body", ""), height=280)
        st.download_button("⬇️ Download .eml", f"Subject: {subject}\n\n{body}",
                           file_name="draft.eml", mime="message/rfc822")

        st.divider()
        st.caption("Send this brief to a colleague:")
        to_addr = st.text_input("Recipient email address")
        if st.button("📤 Send email", disabled=not to_addr):
            try:
                status = EmailAgent().send(to_addr, subject, body)
                if status.get("sent"):
                    st.success(f"Sent to {to_addr}")
                else:
                    st.warning(f"Not sent: {status.get('reason')}")
            except Exception as e:  # noqa: BLE001
                st.error(f"Send failed: {e}")
    else:
        st.info("Run the pipeline to draft an email.")
