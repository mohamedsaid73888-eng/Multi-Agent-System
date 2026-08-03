"""
Lightweight run-trace collector.

Every agent appends a structured event to a shared RunTrace. The Streamlit UI
and the evaluation script both read this to show *what happened, in what order,
and how long each step took* - which is the visible proof of orchestration.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TraceEvent:
    agent: str
    action: str
    status: str  # "ok" | "fallback" | "error"
    latency_s: float
    detail: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class RunTrace:
    def __init__(self) -> None:
        self.events: List[TraceEvent] = []
        self.started_at = time.time()

    def add(self, event: TraceEvent) -> None:
        self.events.append(event)

    @property
    def total_latency_s(self) -> float:
        return round(sum(e.latency_s for e in self.events), 3)

    @property
    def wall_clock_s(self) -> float:
        return round(time.time() - self.started_at, 3)

    def success_rate(self) -> float:
        if not self.events:
            return 0.0
        ok = sum(1 for e in self.events if e.status in ("ok", "fallback"))
        return round(ok / len(self.events), 3)

    def as_rows(self) -> List[Dict[str, Any]]:
        return [
            {
                "Agent": e.agent,
                "Action": e.action,
                "Status": e.status,
                "Latency (s)": e.latency_s,
                "Detail": e.detail,
            }
            for e in self.events
        ]
