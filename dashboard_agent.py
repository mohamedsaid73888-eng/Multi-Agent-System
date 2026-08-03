"""
DashboardAgent  (Objective 3: Automated Actions - updating dashboards)

Computes metrics and chart-ready structures from the documents, analysis, and
run trace. The Streamlit UI renders these as live dashboard panels.

Returns plain dicts/lists so it is fully decoupled from the plotting library.
"""
from __future__ import annotations

import time
from collections import Counter
from typing import Any, Dict, List

from agents.base import BaseAgent, Document
from utils.extractive import keywords as extract_keywords
from utils.logging_utils import RunTrace


class DashboardAgent(BaseAgent):
    name = "DashboardAgent"

    def run(
        self,
        docs: List[Document],
        analysis: Dict[str, Any],
        trace: RunTrace,
    ) -> Dict[str, Any]:
        t0 = time.time()

        source_types = Counter(d.source_type for d in docs)
        chars_per_source = {d.title: d.char_count for d in docs}

        # summary compression ratio (efficiency signal)
        total_in = sum(d.char_count for d in docs) or 1
        total_sum = sum(len(d.summary) for d in docs) or 1
        compression = round(total_sum / total_in, 3)

        # aggregate keywords across all sources
        all_text = " ".join(d.summary or d.text for d in docs)
        top_keywords = extract_keywords(all_text, top_n=15)

        # per-agent latency from the trace
        agent_latency: Dict[str, float] = {}
        for e in trace.events:
            agent_latency[e.agent] = round(agent_latency.get(e.agent, 0.0) + e.latency_s, 3)

        metrics = {
            "sources": len(docs),
            "total_input_chars": total_in,
            "total_summary_chars": total_sum,
            "compression_ratio": compression,
            "success_rate": trace.success_rate(),
            "total_agent_latency_s": trace.total_latency_s,
            "wall_clock_s": trace.wall_clock_s,
            "sentiment": analysis.get("overall_sentiment", "neutral"),
            "confidence": analysis.get("confidence", "n/a"),
        }

        result = {
            "metrics": metrics,
            "source_type_counts": dict(source_types),
            "chars_per_source": chars_per_source,
            "top_keywords": top_keywords,
            "agent_latency": agent_latency,
            # pass through any tabular data for optional deeper charts
            "tables": [
                {"title": d.title, "dataframe": d.meta.get("dataframe")}
                for d in docs
                if d.meta.get("dataframe") is not None
            ],
        }
        self._record("build_dashboard", "ok", time.time() - t0, detail="metrics computed")
        return result
