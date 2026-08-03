"""
Orchestrator  (Objective 4: Workflow Orchestration)

Coordinates the full multi-agent pipeline:

    Retrieve  ->  Summarize (PARALLEL across sources)  ->  Analyze (synthesis)
              ->  [Report | Email | Dashboard]  (automated actions)

Sequential stages guarantee correct data dependencies; the summarization stage
runs in parallel with a thread pool because each source is independent, which
is where most of the wall-clock time is saved. All timing/status is captured in
a shared RunTrace so the UI and evaluator can inspect exactly what happened.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agents import (
    AnalysisAgent,
    DashboardAgent,
    Document,
    EmailAgent,
    ReportAgent,
    RetrievalAgent,
    SummarizerAgent,
)
from config import Settings, settings as default_settings
from llm_client import LLMClient
from utils.logging_utils import RunTrace


@dataclass
class PipelineResult:
    documents: List[Document] = field(default_factory=list)
    analysis: Dict[str, Any] = field(default_factory=dict)
    report: Dict[str, Any] = field(default_factory=dict)
    email: Dict[str, str] = field(default_factory=dict)
    dashboard: Dict[str, Any] = field(default_factory=dict)
    trace: Optional[RunTrace] = None


class Orchestrator:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or default_settings
        self.trace = RunTrace()
        self.llm = LLMClient(self.s)
        # instantiate agents sharing the same trace + llm client
        self.retrieval = RetrievalAgent(self.llm, self.trace)
        self.summarizer = SummarizerAgent(self.llm, self.trace)
        self.analysis = AnalysisAgent(self.llm, self.trace)
        self.report = ReportAgent(self.llm, self.trace)
        self.email = EmailAgent(self.llm, self.trace)
        self.dashboard = DashboardAgent(self.llm, self.trace)

    def run(
        self,
        sources: List[Dict[str, Any]],
        report_title: str = "Market Intelligence Brief",
        objective: str = "",
        max_sentences: int = 5,
        do_report: bool = True,
        do_email: bool = True,
        do_dashboard: bool = True,
        email_recipient: str = "team",
        progress_cb: Optional[Callable[[str, float], None]] = None,
    ) -> PipelineResult:
        """
        progress_cb(stage_label, fraction 0..1) lets the UI show live progress.
        """

        def _progress(label: str, frac: float) -> None:
            if progress_cb:
                progress_cb(label, frac)

        # ---- Stage 1: Retrieval (sequential) ----------------------------
        _progress("Retrieving sources", 0.05)
        docs = self.retrieval.run(sources)
        if not docs:
            _progress("No documents retrieved", 1.0)
            return PipelineResult(trace=self.trace)

        # ---- Stage 2: Summarization (PARALLEL) --------------------------
        _progress("Summarizing sources (parallel)", 0.25)
        docs = self._summarize_parallel(docs, max_sentences)

        # ---- Stage 3: Analysis / synthesis (sequential) -----------------
        _progress("Synthesizing analysis", 0.55)
        analysis = self.analysis.run(docs, objective=objective)

        result = PipelineResult(documents=docs, analysis=analysis, trace=self.trace)

        # ---- Stage 4: Automated actions ---------------------------------
        if do_report:
            _progress("Generating report", 0.70)
            result.report = self.report.run(report_title, docs, analysis)
        if do_email:
            _progress("Drafting email", 0.85)
            result.email = self.email.draft(report_title, analysis, recipient=email_recipient)
        if do_dashboard:
            _progress("Building dashboard", 0.95)
            result.dashboard = self.dashboard.run(docs, analysis, self.trace)

        _progress("Done", 1.0)
        return result

    # -- parallel summarization -------------------------------------------
    def _summarize_parallel(self, docs: List[Document], max_sentences: int) -> List[Document]:
        workers = max(1, min(self.s.parallel_workers, len(docs)))
        if workers == 1:
            return [self.summarizer.run(d, max_sentences) for d in docs]

        results: Dict[int, Document] = {}
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(self.summarizer.run, d, max_sentences): i
                for i, d in enumerate(docs)
            }
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
        # preserve original order
        return [results[i] for i in range(len(docs))]
