"""
Central configuration for Argus — Market & Competitive Intelligence.

Groq is the only supported LLM provider. Settings are read from environment
variables (or a local .env file) so no secret is ever hard-coded. If no Groq
key is present the app still runs, degrading to extractive (non-LLM) fallbacks.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

try:
    # optional: load a local .env if present
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # dotenv is optional
    pass


# ---- Product identity (single source of truth for branding) --------------
APP_NAME = os.getenv("APP_NAME", "Argus")
APP_TAGLINE = os.getenv("APP_TAGLINE", "Your market, watched.")
APP_DESCRIPTION = (
    "Turn competitor pages, market reports, and metrics into a "
    "decision-ready market-intelligence brief."
)


@dataclass
class Settings:
    # ---- LLM provider (Groq only) ---------------------------------------
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")

    # Fallback chain (first model is tried first, then the next on failure).
    # Mirrors a production-style router: strong model -> fast/cheap model.
    groq_models: List[str] = field(
        default_factory=lambda: [
            os.getenv("GROQ_PRIMARY_MODEL", "llama-3.3-70b-versatile"),
            os.getenv("GROQ_FALLBACK_MODEL", "llama-3.1-8b-instant"),
        ]
    )

    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.3"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))
    request_timeout: int = int(os.getenv("LLM_TIMEOUT", "60"))

    # ---- Retrieval -------------------------------------------------------
    max_chars_per_source: int = int(os.getenv("MAX_CHARS_PER_SOURCE", "20000"))

    # ---- Orchestration ---------------------------------------------------
    parallel_workers: int = int(os.getenv("PARALLEL_WORKERS", "4"))

    # ---- Email (automated action) ---------------------------------------
    smtp_host: str = os.getenv("SMTP_HOST", "")
    smtp_port: int = int(os.getenv("SMTP_PORT", "587"))
    smtp_user: str = os.getenv("SMTP_USER", "")
    smtp_password: str = os.getenv("SMTP_PASSWORD", "")

    def has_llm_key(self) -> bool:
        return bool(self.groq_api_key)

    def active_models(self) -> List[str]:
        return self.groq_models


# a singleton-ish default instance the app imports
settings = Settings()
