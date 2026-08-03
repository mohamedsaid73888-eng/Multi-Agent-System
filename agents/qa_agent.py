"""
QAAgent  (Ask Argus - RAG question answering over the analyzed sources)

Answers a user's question using ONLY the chunks retrieved for that question by
the RAG index (see utils/rag.py). The retrieval step happens upstream in the
/ask endpoint; this agent receives the already-relevant chunks and generates the
answer with the Groq LLM. With no key it degrades to returning the retrieved
passages directly, so answers stay grounded and useful offline.

`context` is a list of {"title": str, "text": str} retrieved chunks.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

from agents.base import BaseAgent
from llm_client import LLMError

_SYSTEM = (
    "You are Argus, a market-intelligence analyst. Answer the user's question "
    "using ONLY the retrieved passages provided. Cite passages inline as "
    "[Source N]. If the passages do not contain the answer, say so plainly "
    "instead of guessing. Be concise and concrete."
)


class QAAgent(BaseAgent):
    name = "QAAgent"

    def answer(self, question: str, context: List[Dict[str, str]]) -> Dict[str, Any]:
        t0 = time.time()
        question = (question or "").strip()
        if not question:
            return {"answer": "Please enter a question.", "grounded": False}
        if not context:
            return {
                "answer": "I couldn't find anything relevant to that in your "
                          "sources. Try rephrasing, or add a source that covers it.",
                "grounded": False,
            }

        if self.llm.available:
            try:
                passages = "\n\n".join(
                    f"[Source {i}] {c.get('title', 'untitled')}\n{c.get('text', '')}"
                    for i, c in enumerate(context, 1)
                )
                prompt = f"RETRIEVED PASSAGES:\n{passages}\n\nQUESTION: {question}"
                res = self.llm.chat(_SYSTEM, prompt)
                self._record("ask", "ok", time.time() - t0,
                             detail=f"LLM {res.model}", model=res.model)
                return {"answer": res.text, "grounded": True,
                        "engine": f"llm:{res.model}"}
            except LLMError as e:
                self._record("ask", "fallback", time.time() - t0, detail=str(e))

        # no LLM: return the retrieved passages verbatim (still grounded)
        self._record("ask", "fallback", time.time() - t0, detail="retrieval-only")
        answer = "Based on the most relevant passages from your sources:\n\n" + "\n\n".join(
            f"- ({c.get('title', 'source')}) {c.get('text', '')}" for c in context[:3]
        )
        return {"answer": answer, "grounded": True, "engine": "retrieval-only"}
