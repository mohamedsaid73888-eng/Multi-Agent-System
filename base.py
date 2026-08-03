"""
Base agent + shared data models.

Every specialized agent inherits from BaseAgent, which gives it:
  * a reference to the shared LLMClient and RunTrace
  * a `_timed()` helper that records a TraceEvent automatically
  * a consistent `run()` entrypoint contract
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_client import LLMClient
from utils.logging_utils import RunTrace, TraceEvent


@dataclass
class Document:
    """A normalized unit of retrieved content from any source."""

    source_id: str
    source_type: str  # "pdf" | "csv" | "sheet" | "api" | "web" | "text"
    title: str
    text: str
    meta: Dict[str, Any] = field(default_factory=dict)

    # populated by later agents
    summary: str = ""
    keywords: List[str] = field(default_factory=list)

    @property
    def char_count(self) -> int:
        return len(self.text)


class BaseAgent:
    name: str = "BaseAgent"

    def __init__(self, llm: Optional[LLMClient] = None, trace: Optional[RunTrace] = None):
        self.llm = llm or LLMClient()
        self.trace = trace or RunTrace()

    def _record(
        self,
        action: str,
        status: str,
        latency_s: float,
        detail: str = "",
        **meta: Any,
    ) -> None:
        if self.trace is not None:
            self.trace.add(
                TraceEvent(
                    agent=self.name,
                    action=action,
                    status=status,
                    latency_s=round(latency_s, 3),
                    detail=detail,
                    meta=meta,
                )
            )

    def _timed(self, action: str, fn, *args, **kwargs):
        """Run fn, record timing + status, return its result. Re-raises errors."""
        t0 = time.time()
        try:
            result = fn(*args, **kwargs)
            self._record(action, "ok", time.time() - t0)
            return result
        except Exception as e:  # noqa: BLE001
            self._record(action, "error", time.time() - t0, detail=str(e))
            raise

    def run(self, *args, **kwargs):  # pragma: no cover - abstract
        raise NotImplementedError
