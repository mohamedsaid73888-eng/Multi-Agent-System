"""
AnalysisAgent  (Objective 2: Processing & Analysis, cross-source)

Takes the per-source summaries and produces a competitive-landscape read:
  * an executive synthesis across all sources
  * a structured JSON object: market_trends, competitor_moves, opportunities,
    threats, recommendations, sentiment

Robustness:
  * The LLM is asked for STRICT JSON. We parse defensively (strip code fences,
    locate the outer braces). If parsing or the LLM fails, we build a structured
    result from the extractive keywords/summaries so downstream agents still work.
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from typing import Any, Dict, List

from agents.base import BaseAgent, Document
from llm_client import LLMError
from utils.extractive import keywords as extract_keywords

_SYSTEM = (
    "You are a competitive and market-intelligence analyst. Synthesize the "
    "provided sources into one coherent read of the market: momentum, the moves "
    "of key players, where the openings are, and what threatens the position. "
    "Respond with STRICT JSON only - no prose, no markdown fences. Ground every "
    "claim in the sources; never invent competitors, numbers, or events."
)

_SCHEMA_HINT = """
Return JSON with exactly these keys:
{
  "executive_summary": "3-5 sentence read of the competitive landscape",
  "market_trends": ["trend or shift in demand/technology/pricing", "..."],
  "competitor_moves": ["a named player and what they are doing", "..."],
  "opportunities": ["an exploitable opening or unmet need", "..."],
  "threats": ["a competitive or market risk to guard against", "..."],
  "recommendations": ["actionable next step", "..."],
  "overall_sentiment": "positive" | "neutral" | "negative",
  "confidence": 0.0
}
""".strip()


class AnalysisAgent(BaseAgent):
    name = "AnalysisAgent"

    def run(self, docs: List[Document], objective: str = "") -> Dict[str, Any]:
        t0 = time.time()
        joined = self._build_context(docs)

        if self.llm.available:
            try:
                prompt = (
                    (f"Analysis objective: {objective}\n\n" if objective else "")
                    + f"{_SCHEMA_HINT}\n\nSOURCE SUMMARIES:\n{joined}"
                )
                res = self.llm.chat(_SYSTEM, prompt)
                parsed = self._safe_json(res.text)
                if parsed:
                    parsed.setdefault("overall_sentiment", "neutral")
                    parsed["_engine"] = f"llm:{res.model}"
                    self._record(
                        "analyze", "ok", time.time() - t0,
                        detail=f"LLM {res.model}", model=res.model,
                    )
                    return parsed
                # JSON parse failed -> degrade
                self._record(
                    "analyze", "fallback", time.time() - t0,
                    detail="LLM returned non-JSON, used heuristic synthesis",
                )
                return self._heuristic(docs)
            except LLMError as e:
                self._record(
                    "analyze", "fallback", time.time() - t0,
                    detail=f"LLM failed ({e}), heuristic synthesis",
                )
                return self._heuristic(docs)

        self._record(
            "analyze", "fallback", time.time() - t0,
            detail="heuristic synthesis (no LLM key)",
        )
        return self._heuristic(docs)

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def _build_context(docs: List[Document]) -> str:
        blocks = []
        for i, d in enumerate(docs, 1):
            blocks.append(
                f"[Source {i}] {d.title} ({d.source_type})\n"
                f"Keywords: {', '.join(d.keywords)}\n"
                f"Summary: {d.summary or d.text[:600]}"
            )
        return "\n\n".join(blocks)

    @staticmethod
    def _safe_json(text: str) -> Dict[str, Any]:
        # strip ```json ... ``` fences if present
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE)
        cleaned = cleaned.strip()
        # try direct
        try:
            return json.loads(cleaned)
        except Exception:
            pass
        # try to locate the outermost braces
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return {}
        return {}

    def _heuristic(self, docs: List[Document]) -> Dict[str, Any]:
        all_text = " ".join(d.summary or d.text for d in docs)
        kws = extract_keywords(all_text, top_n=12)
        # each source is treated as a market signal / competitor mention
        moves = [
            f"{d.title}: {(d.summary or d.text[:160]).strip()}" for d in docs
        ]
        # extremely rough sentiment via lexicon
        pos = len(re.findall(r"\b(growth|increase|improve|success|gain|strong|profit)\b", all_text, re.I))
        neg = len(re.findall(r"\b(decline|decrease|loss|risk|fail|weak|drop|concern)\b", all_text, re.I))
        sentiment = "positive" if pos > neg else "negative" if neg > pos else "neutral"
        return {
            "executive_summary": " ".join(
                (d.summary or d.text[:200]) for d in docs[:3]
            )[:600],
            "market_trends": [
                f"Recurring themes across sources: {', '.join(kws[:6])}."
            ],
            "competitor_moves": moves[:8],
            "opportunities": [
                f"Underdeveloped angles worth exploring: {', '.join(kws[6:11]) or 'n/a'}.",
            ],
            "threats": [
                "Automated heuristic synthesis - configure a Groq key for a deeper "
                "competitive read.",
            ],
            "recommendations": [
                f"Track the leading signals: {', '.join(kws[:5])}.",
                "Cross-check figures against primary sources before acting.",
            ],
            "overall_sentiment": sentiment,
            "confidence": 0.4,
            "_engine": "heuristic",
            "_top_keywords": kws,
        }
