"""
Provider-agnostic LLM client.

Design goals (maps to objective 6 - reliability):
  * One `chat()` interface backed by the Groq API.
  * A model fallback chain: if the primary model errors or rate-limits, try the
    next model in the chain before giving up.
  * Graceful offline degradation: if there is no key / no network, callers can
    fall back to extractive (non-LLM) logic instead of crashing.

The client lazily imports the Groq SDK so the app still starts even if the SDK
is not installed.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

from config import Settings, settings as default_settings


class LLMError(RuntimeError):
    """Raised when every model in the fallback chain fails."""


@dataclass
class LLMResult:
    text: str
    model: str
    provider: str
    latency_s: float
    used_fallback: bool = False
    attempts: List[str] = field(default_factory=list)


class LLMClient:
    def __init__(self, settings: Optional[Settings] = None):
        self.s = settings or default_settings

    # -- public API --------------------------------------------------------
    @property
    def available(self) -> bool:
        """True if we have a Groq API key configured."""
        return self.s.has_llm_key()

    def chat(self, system: str, user: str) -> LLMResult:
        """
        Send a system+user prompt through the fallback chain.
        Raises LLMError if all models fail (callers should catch and degrade).
        """
        if not self.available:
            raise LLMError(
                "No GROQ_API_KEY configured. "
                "Set the key or use extractive fallback."
            )

        models = self.s.active_models()
        attempts: List[str] = []
        last_err: Optional[Exception] = None

        for idx, model in enumerate(models):
            attempts.append(model)
            t0 = time.time()
            try:
                text = self._groq(model, system, user)
                return LLMResult(
                    text=text.strip(),
                    model=model,
                    provider="groq",
                    latency_s=round(time.time() - t0, 3),
                    used_fallback=idx > 0,
                    attempts=attempts,
                )
            except Exception as e:  # noqa: BLE001 - we intentionally try next model
                last_err = e
                # brief backoff before trying the fallback model
                time.sleep(0.5)
                continue

        raise LLMError(
            f"All Groq models failed. "
            f"Tried {attempts}. Last error: {last_err}"
        )

    # -- Groq call ---------------------------------------------------------
    def _groq(self, model: str, system: str, user: str) -> str:
        from groq import Groq

        client = Groq(api_key=self.s.groq_api_key, timeout=self.s.request_timeout)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=self.s.temperature,
            max_tokens=self.s.max_tokens,
        )
        return resp.choices[0].message.content or ""
