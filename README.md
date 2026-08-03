# 🛰️ Argus — Market & Competitive Intelligence

> *Your market, watched.*

**Argus** turns scattered market signals — competitor web pages, market reports
(PDF), metrics (CSV), Google Sheets, news/market APIs, and pasted notes — into a
**decision-ready intelligence brief**: an executive read of the competitive
landscape (market trends, competitor moves, opportunities, threats), a
**stakeholder email**, a **live dashboard**, and an **Ask Argus** chatbot that
answers questions grounded in your sources (RAG).

It's a **single Streamlit app**: a team of coordinated agents retrieves,
summarizes in parallel, synthesizes, and acts — all in one process, no separate
backend. LLM calls go to the **Groq API**; retrieval runs locally.

---

## How it maps to the project objectives

| # | Objective | Where it lives |
|---|-----------|----------------|
| 1 | **Data Retrieval** (PDF, Sheets, APIs, web) | `agents/retrieval_agent.py` — PDF, CSV/TSV, public Google Sheets, JSON APIs, web pages, raw text |
| 2 | **Processing & Summarization** with LLMs | `agents/summarizer_agent.py` (per-source, parallel) + `agents/analysis_agent.py` (competitive-landscape synthesis → structured JSON: trends, competitor moves, opportunities, threats) |
| 3 | **Automated Actions** | `agents/report_agent.py` (Markdown + PDF brief), `agents/email_agent.py` (draft + optional SMTP send), `agents/dashboard_agent.py` (metrics + charts) |
| 4 | **Workflow Orchestration** (sequential + parallel) | `orchestrator.py` — sequential stages with a **parallel** summarization stage (thread pool) |
| 5 | **Presentation & Explanation** | `app.py` (single Streamlit app, 5 tabs) + the *Demo script* below |
| 6 | **Testing & Evaluation** | `tests/test_pipeline.py` (pytest), `evaluation/evaluate.py` (reliability / accuracy / efficiency gates) |
| + | **RAG chatbot** (Ask Argus) | `utils/rag.py` (chunk → index → retrieve) + `agents/qa_agent.py` (grounded answer with citations) |

---

## Architecture

```
                    ┌──────────────────────────────────────────┐
                    │               Orchestrator                │
                    │      (shared LLMClient + RunTrace)         │
                    └──────────────────────────────────────────┘
                                     │
   Stage 1            Stage 2 (∥)            Stage 3           Stage 4 (actions)
 ┌───────────┐   ┌──────────────────┐   ┌───────────┐   ┌──────────────────────┐
 │ Retrieval │──▶│   Summarizer ×N   │──▶│ Analysis  │──▶│ Report │ Email │ Dash │
 │  Agent    │   │ (parallel threads)│   │  Agent    │   │  Agent │ Agent │ Agent│
 └───────────┘   └──────────────────┘   └───────────┘   └──────────────────────┘
   PDF/CSV/         LLM summary per        cross-source      Markdown+PDF │ draft │
   Sheet/API/       source (+ extractive   synthesis →       email │ metrics + Plotly
   web/text         fallback)              strict JSON        charts

 Ask Argus (on demand):
   question ─▶ RAG retrieve top-k chunks ─▶ QAAgent ─▶ Groq ─▶ answer + [Source N]
              (utils/rag.py, local)         (agents/qa_agent.py, LLM)
```

**Reliability by design:**
- **Model fallback chain** — if the primary model errors/rate-limits, `LLMClient`
  retries the next model (`llama-3.3-70b-versatile → llama-3.1-8b-instant`).
- **Offline degradation** — with no Groq key, every LLM step falls back to a
  dependency-free extractive summarizer / heuristic analyzer, and Ask Argus
  returns the retrieved passages verbatim. The whole app still runs.
- **Per-source isolation** — a failure loading one source is traced and skipped;
  the others still process.

---

## Quickstart (local)

```bash
python -m venv venv && venv/Scripts/activate     # Windows (or: source venv/bin/activate)
pip install -r requirements.txt

# add your Groq key (optional — without it the app uses extractive fallbacks)
cp .env.example .env      # then edit GROQ_API_KEY

streamlit run app.py
```

Open http://localhost:8501 → **Signals** tab → add a source (upload a PDF/CSV, or
type a URL and press Enter) → **Run Argus**.

Everything runs in one process — staging sources, the multi-agent pipeline, the
RAG index for *Ask Argus*, and email sending. Configuration (`GROQ_API_KEY`,
`SMTP_*`, `RAG_BACKEND`, `EMBEDDING_MODEL`) comes from the environment / `.env`.
Get a free Groq key at <https://console.groq.com>.

`requirements.txt` includes the **semantic (dense) RAG** stack — HuggingFace
embeddings + FAISS — so Ask Argus does true semantic retrieval. First install
downloads PyTorch (a few hundred MB). If you ever need the lightweight path, set
`RAG_BACKEND=sparse` and it uses TF-IDF instead.

---

## Deployment (Streamlit Community Cloud)

A single Streamlit process, so it deploys directly — no Docker, no backend.

1. Push this repo to GitHub.
2. <https://share.streamlit.io> → **New app** → pick this repo → main file `app.py`.
3. **Advanced settings → Secrets**, from
   [`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example):
   ```toml
   GROQ_API_KEY = "gsk_your_key_here"
   # SMTP_HOST = "smtp.gmail.com"     # uncomment to enable email sending
   # SMTP_PORT = "587"
   # SMTP_USER = "you@gmail.com"
   # SMTP_PASSWORD = "app-password"
   ```
4. Deploy.

**Notes**
- **Secrets → env.** The app copies Streamlit secrets into environment variables
  at startup, so `Settings` and the email agent pick them up automatically.
- **RAG on the cloud.** `requirements.txt` installs the dense stack (CPU-only
  torch + FAISS) so Ask Argus behaves the same on the cloud as locally. It's heavy
  for the free tier — if the build or runtime can't fit torch, `utils/rag.py`
  auto-falls-back to TF-IDF (set `RAG_BACKEND=sparse` in Secrets to force it).
- **Email sending** requires the `SMTP_*` secrets (Gmail needs a 16-char App
  Password + 2-Step Verification).
  
## Application link:

https://multi-agent-systemmm.streamlit.app/

---

## The app (5 tabs)

- **📥 Signals** — add sources (uploads and typed URLs/text auto-stage), set the
  brief title + objective, and **Run Argus**.
- **📄 Brief** — the market-intelligence brief (sentiment + confidence, then the
  full report) with Markdown/PDF download.
- **📊 Dashboard** — signals analyzed, market sentiment, signals-by-type, trending
  topics, and charts of any numeric columns from an uploaded CSV.
- **💬 Ask** — RAG chatbot grounded in the run's sources, with a "🔎 Retrieved
  passages" expander showing what was retrieved and by which retriever.
- **✉️ Email** — the drafted stakeholder email; download `.eml` or send via SMTP.

---

## Testing & evaluation

```bash
pytest -q                       # unit + integration tests (offline, no key)
python evaluation/evaluate.py   # reliability / accuracy / efficiency report
```

`pytest` covers the extractive utils, each agent (retrieval, summarizer, analysis),
the RAG retriever + QAAgent, and the orchestrator end-to-end (13 tests, all
offline). `evaluate.py` runs the full pipeline on a fixed, self-contained corpus
and reports retrieval completeness, agent success rate, keyword-coverage &
ground-truth recall, per-agent latency, and a parallel-vs-serial speed comparison,
ending with PASS/FAIL gates.

---

## Sample data

The [`samples/`](samples/) folder has ready-made inputs — one of each source type
(PDF, CSV, text, plus a website URL and Google Sheet instructions) — sharing one
"MENA cloud market" theme so the brief reads coherently. See
[`samples/SAMPLES.md`](samples/SAMPLES.md) for how to load them.

---

## Project structure

```
Multi-Agent-System/
├── app.py                 # Streamlit app (runs the whole pipeline in-process)
├── orchestrator.py        # coordinates all agents (sequential + parallel)
├── llm_client.py          # Groq LLM client + model fallback chain
├── config.py              # env-driven settings + product identity
├── agents/
│   ├── base.py            # BaseAgent + Document model + timing/trace
│   ├── retrieval_agent.py # PDF/CSV/Sheet/API/web/text loaders
│   ├── summarizer_agent.py
│   ├── analysis_agent.py  # competitive-landscape synthesis → strict JSON
│   ├── report_agent.py    # Markdown + PDF
│   ├── email_agent.py     # draft + optional SMTP send
│   ├── dashboard_agent.py # metrics + chart data
│   └── qa_agent.py        # Ask Argus (RAG answer generation)
├── utils/
│   ├── extractive.py      # offline summarizer/keyword fallback
│   ├── rag.py             # RAG retriever (TF-IDF, or dense embeddings+FAISS)
│   └── logging_utils.py   # RunTrace / TraceEvent
├── samples/               # ready-made demo inputs (PDF, CSV, text + guide)
├── evaluation/evaluate.py # evaluation harness (CLI + run_evaluation())
├── tests/test_pipeline.py
├── requirements.txt       # all deps (incl. semantic RAG: torch + FAISS)
├── .env.example
└── .streamlit/config.toml
```

---

## 🎬 Demo script (for the face-cam video — objective 5)

Keep it ~4–6 minutes:

1. **Problem (20s).** "Analysts waste hours pulling data from PDFs, sheets, and
   web pages, then summarizing and reporting by hand. Argus automates that whole
   workflow with cooperating AI agents."
2. **Architecture (60s).** Show the diagram above. Explain the four stages and
   *why summarization is parallel* (sources are independent → biggest time saving)
   while the rest is sequential (each stage needs the previous stage's output).
3. **Live run (120s).** Load the bundled samples (`samples/` — a PDF, CSV, text,
   and a web-page URL). Click **Run Argus** — the agents run in-process and produce
   the brief, dashboard, and email.
4. **Outputs (90s).** Walk the tabs: **Brief** (download the PDF), **Dashboard**
   (charts + metrics), **Email** (the drafted stakeholder email).
5. **Ask Argus (40s).** In the **Ask** tab, ask a content question and open the
   "🔎 Retrieved passages" expander — the answer is grounded in the chunks the RAG
   retriever pulled, with `[Source N]` citations, not invented.
6. **Reliability & testing (30s).** Mention the offline fallbacks, then in a
   terminal run `pytest -q` (13 pass) and `python evaluation/evaluate.py` (PASS
   gates) to show the automated test & evaluation harness.

---

## Notes & troubleshooting

- **Groq model names change over time.** If a model 404s, update
  `GROQ_PRIMARY_MODEL` / `GROQ_FALLBACK_MODEL` in `.env` to current Groq models.
- **Google Sheets** must be shared publicly ("anyone with the link") for the
  CSV-export retrieval to work without OAuth.
- **PDF export** uses core Latin-1 fonts; non-Latin characters are replaced. For
  full Unicode, register a TTF font in `report_agent.py`.
- **Email sending** is off by default. It only sends when the `SMTP_*` vars are
  set *and* you click send in the Email tab.
