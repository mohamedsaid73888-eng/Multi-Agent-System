"""
Unit + integration tests (Objective 6: Testing & Evaluation).

These run fully offline (no GROQ_API_KEY is set) so CI never needs a key: the
agents fall back to extractive logic. They verify each agent's contract and the
orchestrator's end-to-end behavior including the parallel summarization stage.

Run with:  pytest -q
"""
import os
import sys

# ensure no key is picked up so agents use the extractive fallback path
os.environ.pop("GROQ_API_KEY", None)
# force the lightweight TF-IDF retriever so RAG tests are fast and need no torch
os.environ["RAG_BACKEND"] = "sparse"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from agents import RetrievalAgent, SummarizerAgent, AnalysisAgent, Document  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from utils.extractive import summarize, keywords, split_sentences  # noqa: E402


# ---- extractive utils ---------------------------------------------------
def test_split_sentences():
    text = "This is the first sentence here. And here is a second longer one to keep."
    sents = split_sentences(text)
    assert len(sents) >= 1


def test_summarize_bounds():
    text = " ".join(f"Sentence number {i} carries some unique content about topic {i}." for i in range(20))
    out = summarize(text, max_sentences=3)
    assert out
    assert out.count(".") <= 4  # roughly 3 sentences


def test_keywords_excludes_stopwords():
    kws = keywords("the cloud market grew and the cloud revenue increased with cloud demand", top_n=5)
    assert "cloud" in kws
    assert "the" not in kws


# ---- retrieval agent ----------------------------------------------------
def test_retrieval_csv():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    data = df.to_csv(index=False).encode()
    agent = RetrievalAgent()
    doc = agent.from_csv_bytes(data, "t.csv")
    assert doc.source_type == "csv"
    assert doc.meta["rows"] == 3
    assert doc.meta["cols"] == 2
    assert doc.meta.get("dataframe") is not None


def test_retrieval_text():
    agent = RetrievalAgent()
    doc = agent.from_text("hello world " * 10, "note")
    assert doc.source_type == "text"
    assert doc.char_count > 0


def test_retrieval_skips_bad_source():
    agent = RetrievalAgent()
    docs = agent.run([{"kind": "csv", "bytes": b"not,really\x00valid", "title": "bad"},
                      {"kind": "text", "text": "good source", "title": "ok"}])
    # bad one is skipped, good one survives
    titles = [d.title for d in docs]
    assert "ok" in titles


# ---- summarizer agent ---------------------------------------------------
def test_summarizer_offline_fallback():
    agent = SummarizerAgent()
    doc = Document(source_id="d", source_type="text", title="d",
                   text=" ".join(f"Fact {i} about the system is recorded." for i in range(15)))
    out = agent.run(doc, max_sentences=3)
    assert out.summary
    assert out.keywords


# ---- analysis agent -----------------------------------------------------
def test_analysis_offline_structured():
    docs = [
        Document(source_id="1", source_type="text", title="A",
                 text="Revenue grew strongly and profit increased.", summary="Revenue grew strongly."),
        Document(source_id="2", source_type="text", title="B",
                 text="Churn declined and reliability improved.", summary="Churn declined."),
    ]
    out = AnalysisAgent().run(docs)
    for key in ("executive_summary", "market_trends", "competitor_moves",
                "opportunities", "threats", "recommendations", "overall_sentiment"):
        assert key in out
    assert out["overall_sentiment"] in ("positive", "neutral", "negative")


# ---- orchestrator (integration) ----------------------------------------
def test_orchestrator_end_to_end():
    sources = [
        {"kind": "text", "text": "The cloud market grew 30% with strong demand.", "title": "s1"},
        {"kind": "text", "text": "DevOps and SRE hiring accelerated across the region.", "title": "s2"},
    ]
    res = Orchestrator().run(sources, report_title="Test Brief")
    assert len(res.documents) == 2
    assert res.analysis
    assert res.report["markdown"]
    assert res.email["subject"].startswith("[Argus Brief]")
    assert res.dashboard["metrics"]["sources"] == 2
    # every stage recorded something
    assert res.trace.success_rate() == 1.0
    assert len(res.trace.events) >= 6


def test_orchestrator_preserves_source_order():
    sources = [{"kind": "text", "text": f"Doc {i} content here.", "title": f"doc{i}"} for i in range(5)]
    res = Orchestrator().run(sources, do_report=False, do_email=False)
    assert [d.title for d in res.documents] == [f"doc{i}" for i in range(5)]


# ---- RAG / Ask Argus ----------------------------------------------------
def test_rag_chunking_splits_long_text():
    from utils.rag import chunk_document
    text = " ".join(f"Sentence number {i} about topic {i} here." for i in range(60))
    chunks = chunk_document("doc", text, max_chars=300)
    assert len(chunks) > 1
    assert all(c["title"] == "doc" for c in chunks)


def test_rag_retrieves_relevant_chunk():
    from utils.rag import build_index
    docs = [
        {"title": "a", "text": "Acme cut cloud prices sharply this quarter."},
        {"title": "b", "text": "Globex hired fifty new engineers in Cairo."},
        {"title": "c", "text": "Customer retention improved after the redesign."},
    ]
    idx = build_index(docs)
    hits = idx.retrieve("cloud prices", k=1)
    assert hits
    assert "price" in hits[0][0]["text"].lower()   # retrieved the pricing chunk


def test_qa_agent_answers_from_context_offline():
    from agents import QAAgent
    context = [{"title": "memo", "text": "Acme cut cloud prices 15% in MENA."}]
    out = QAAgent().answer("what did Acme do to prices?", context)
    assert out["answer"]
    assert out.get("grounded") is True


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
