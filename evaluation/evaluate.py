"""
Evaluation harness (Objective 6: Testing & Evaluation).

Runs the full multi-agent pipeline on a fixed sample set and reports:

  RELIABILITY  - agent success rate, whether every stage produced output,
                 how many sources failed retrieval.
  ACCURACY     - keyword-coverage of the report vs. the source material
                 (a lightweight, reference-free faithfulness proxy) and whether
                 expected ground-truth terms appear.
  EFFICIENCY   - total wall-clock time, per-agent latency, and the wall-clock
                 speed-up from running summarization in parallel vs. serial.

Usage:
    python evaluation/evaluate.py            # extractive (no key needed)
    GROQ_API_KEY=... python evaluation/evaluate.py   # with the Groq LLM
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Settings  # noqa: E402
from orchestrator import Orchestrator  # noqa: E402
from utils.extractive import keywords  # noqa: E402

# terms we expect a faithful report to surface from the fixed corpus
GROUND_TRUTH_TERMS = ["mena", "cloud", "revenue", "churn", "devops", "sre"]


def _load_sources():
    """A fixed, self-contained corpus so evaluation needs no external files."""
    return [
        {
            "kind": "text",
            "title": "market_memo",
            "text": (
                "The MENA cloud market grew strongly in 2026. Cloud revenue "
                "increased about 30% year over year, while customer churn "
                "declined to record lows as regional demand for cloud "
                "infrastructure accelerated."
            ),
        },
        {
            "kind": "text",
            "title": "sales_summary",
            "text": (
                "Quarterly revenue rose across all regions. Cloud product "
                "revenue led the growth and churn fell further as customer "
                "retention improved."
            ),
        },
        {
            "kind": "text",
            "title": "analyst_note",
            "text": (
                "DevOps and SRE hiring in Egypt accelerated in 2026 with strong "
                "Jenkins and Kubernetes demand."
            ),
        },
    ]


def _keyword_coverage(report_md: str, source_text: str, top_n: int = 15) -> float:
    src_kw = set(keywords(source_text, top_n=top_n))
    if not src_kw:
        return 0.0
    report_l = report_md.lower()
    hit = sum(1 for k in src_kw if k in report_l)
    return round(hit / len(src_kw), 3)


def _ground_truth_recall(report_md: str) -> float:
    report_l = report_md.lower()
    hit = sum(1 for t in GROUND_TRUTH_TERMS if t in report_l)
    return round(hit / len(GROUND_TRUTH_TERMS), 3)


def run_once(parallel_workers: int, force_offline: bool = False):
    s = Settings()
    s.parallel_workers = parallel_workers
    if force_offline:
        s.groq_api_key = ""  # force the extractive path -> instant, no LLM quota
    orch = Orchestrator(s)
    t0 = time.time()
    res = orch.run(
        _load_sources(),
        report_title="Evaluation Run",
        objective="Assess MENA cloud growth and talent gaps",
    )
    wall = time.time() - t0
    return res, wall


def run_evaluation(force_offline: bool = False) -> dict:
    """Run the pipeline on the fixed corpus and return structured results.

    Shared by the CLI (`main`) and the in-app "System check" panel. Pass
    force_offline=True to skip the LLM (fast, deterministic, no quota use)."""
    res, wall_parallel = run_once(parallel_workers=4, force_offline=force_offline)

    source_text = " ".join(d.text for d in res.documents)
    report_md = res.report.get("markdown", "")

    n_expected_sources = 3
    reliability = {
        "sources_retrieved": len(res.documents),
        "sources_expected": n_expected_sources,
        "retrieval_completeness": round(len(res.documents) / n_expected_sources, 3),
        "agent_success_rate": res.trace.success_rate(),
        "report_produced": bool(report_md),
        "email_produced": bool(res.email.get("subject")),
        "dashboard_produced": bool(res.dashboard.get("metrics")),
        "pdf_produced": bool(res.report.get("pdf_bytes")),
    }

    accuracy = {
        "keyword_coverage": _keyword_coverage(report_md, source_text),
        "ground_truth_recall": _ground_truth_recall(report_md),
        "analysis_engine": res.analysis.get("_engine", "n/a"),
        "sentiment": res.analysis.get("overall_sentiment", "n/a"),
    }

    res_serial, wall_serial = run_once(parallel_workers=1, force_offline=force_offline)
    speedup = round(wall_serial / wall_parallel, 2) if wall_parallel else 0.0
    per_agent: dict = {}
    for e in res.trace.events:
        per_agent[e.agent] = round(per_agent.get(e.agent, 0.0) + e.latency_s, 3)
    efficiency = {
        "wall_clock_parallel_s": round(wall_parallel, 3),
        "wall_clock_serial_s": round(wall_serial, 3),
        "parallel_speedup_x": speedup,
        "total_agent_latency_s": res.trace.total_latency_s,
        "per_agent_latency_s": per_agent,
    }

    gates = {
        "all sources retrieved": reliability["retrieval_completeness"] == 1.0,
        "agent success >= 0.9": reliability["agent_success_rate"] >= 0.9,
        "all outputs produced": all(
            [reliability["report_produced"], reliability["email_produced"],
             reliability["dashboard_produced"]]
        ),
        "ground-truth recall >= 0.5": accuracy["ground_truth_recall"] >= 0.5,
    }

    return {
        "reliability": reliability,
        "accuracy": accuracy,
        "efficiency": efficiency,
        "gates": gates,
        "overall": all(gates.values()),
    }


def main():
    print("=" * 68)
    print(" MULTI-AGENT SYSTEM - EVALUATION REPORT")
    print("=" * 68)

    r = run_evaluation()

    def _section(name, d):
        print(f"\n[{name}]")
        for k, v in d.items():
            print(f"  {k:.<32} {v}")

    _section("RELIABILITY", r["reliability"])
    _section("ACCURACY", r["accuracy"])
    _section("EFFICIENCY", r["efficiency"])
    if r["efficiency"]["parallel_speedup_x"] < 1.0:
        print("  note: parallel < serial here because extractive summarization is")
        print("        near-instant offline, so thread overhead dominates. With a")
        print("        real LLM (each call ~1-3s) parallel summarization wins clearly.")

    print("\n[GATES]")
    for name, ok in r["gates"].items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    print("\n" + "=" * 68)
    print(f" OVERALL: {'PASS' if r['overall'] else 'FAIL'}")
    print("=" * 68)
    return 0 if r["overall"] else 1


if __name__ == "__main__":
    sys.exit(main())
