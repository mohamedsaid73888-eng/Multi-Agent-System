"""
SummarizerAgent  (Objective 2: Data Processing & Summarization)

Summarizes a single Document. Tries the LLM first; on any failure (no key,
rate-limit, network) it automatically falls back to the extractive summarizer so
the pipeline always produces *something*. The chosen path is recorded in the
trace ("ok" for LLM, "fallback" for extractive) for the reliability metrics.
"""
from __future__ import annotations

import time

from agents.base import BaseAgent, Document
from llm_client import LLMError
from utils.extractive import keywords as extract_keywords
from utils.extractive import summarize as extractive_summarize

_SYSTEM = (
    "You are a precise research summarizer. Produce a faithful, self-contained "
    "summary of the provided source. Do not invent facts. Prefer concrete "
    "numbers, names, and findings over generic statements."
)


class SummarizerAgent(BaseAgent):
    name = "SummarizerAgent"

    def run(self, doc: Document, max_sentences: int = 5) -> Document:
        t0 = time.time()
        doc.keywords = extract_keywords(doc.text, top_n=8)

        prompt = (
            f"Summarize the following source titled '{doc.title}' "
            f"({doc.source_type}) in {max_sentences} sentences. "
            f"Focus on the key facts and insights.\n\n"
            f"SOURCE:\n{doc.text[:8000]}"
        )

        if self.llm.available:
            try:
                res = self.llm.chat(_SYSTEM, prompt)
                doc.summary = res.text
                status = "fallback_model" if res.used_fallback else "ok"
                self._record(
                    f"summarize:{doc.title}",
                    "ok",
                    time.time() - t0,
                    detail=f"LLM {res.model} ({res.latency_s}s)"
                    + (" [model-fallback]" if res.used_fallback else ""),
                    model=res.model,
                )
                return doc
            except LLMError as e:
                # fall through to extractive
                doc.summary = extractive_summarize(doc.text, max_sentences)
                self._record(
                    f"summarize:{doc.title}",
                    "fallback",
                    time.time() - t0,
                    detail=f"extractive fallback ({e})",
                )
                return doc

        # offline path
        doc.summary = extractive_summarize(doc.text, max_sentences)
        self._record(
            f"summarize:{doc.title}",
            "fallback",
            time.time() - t0,
            detail="extractive (no LLM key configured)",
        )
        return doc
