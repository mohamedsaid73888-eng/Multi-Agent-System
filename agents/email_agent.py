"""
EmailAgent  (Objective 3: Automated Actions - sending emails)

Drafts a stakeholder email from the analysis. Sending is OPTIONAL and only
happens when the caller explicitly passes send=True AND SMTP credentials are
configured - drafting is always the safe default so the demo never emails
anyone by accident.
"""
from __future__ import annotations

import smtplib
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, Optional

from agents.base import BaseAgent
from config import APP_NAME, settings
from llm_client import LLMError

_SYSTEM = (
    "You write concise, professional stakeholder emails. Return only the email "
    "body (no subject line, no markdown fences). Keep it under 180 words."
)


class EmailAgent(BaseAgent):
    name = "EmailAgent"

    def draft(self, title: str, analysis: Dict[str, Any], recipient: str = "team") -> Dict[str, str]:
        t0 = time.time()
        subject = f"[{APP_NAME} Brief] {title}"

        if self.llm.available:
            try:
                highlights = analysis.get("competitor_moves") or analysis.get("key_findings", [])
                findings = "\n".join(f"- {f}" for f in highlights[:5])
                recs = "\n".join(f"- {r}" for r in analysis.get("recommendations", [])[:3])
                prompt = (
                    f"Write a market-intelligence briefing email to {recipient}.\n\n"
                    f"Executive summary: {analysis.get('executive_summary','')}\n\n"
                    f"Key market signals:\n{findings}\n\nRecommendations:\n{recs}\n\n"
                    f"Market sentiment: {analysis.get('overall_sentiment','neutral')}"
                )
                res = self.llm.chat(_SYSTEM, prompt)
                body = res.text
                self._record("draft_email", "ok", time.time() - t0, detail=f"LLM {res.model}")
                return {"subject": subject, "body": body}
            except LLMError as e:
                self._record("draft_email", "fallback", time.time() - t0, detail=str(e))

        # template fallback
        body = self._template(title, analysis, recipient)
        self._record("draft_email", "fallback", time.time() - t0, detail="template body")
        return {"subject": subject, "body": body}

    @staticmethod
    def _template(title: str, analysis: Dict[str, Any], recipient: str) -> str:
        highlights = analysis.get("competitor_moves") or analysis.get("key_findings", [])
        findings = "\n".join(f"  - {f}" for f in highlights[:5])
        recs = "\n".join(f"  - {r}" for r in analysis.get("recommendations", [])[:3])
        return (
            f"Hi {recipient},\n\n"
            f"Here is your market-intelligence brief for: {title}.\n\n"
            f"{analysis.get('executive_summary','')}\n\n"
            f"Key market signals:\n{findings or '  - (none)'}\n\n"
            f"Recommendations:\n{recs or '  - (none)'}\n\n"
            f"Market sentiment: {analysis.get('overall_sentiment','neutral')}.\n\n"
            f"Best regards,\n{APP_NAME}"
        )

    def send(
        self,
        to_addr: str,
        subject: str,
        body: str,
        from_addr: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Send via SMTP using credentials from config. Returns a status dict."""
        t0 = time.time()
        if not (settings.smtp_host and settings.smtp_user and settings.smtp_password):
            self._record("send_email", "error", time.time() - t0, detail="SMTP not configured")
            return {"sent": False, "reason": "SMTP not configured in environment."}

        from_addr = from_addr or settings.smtp_user
        msg = MIMEMultipart()
        msg["From"] = from_addr
        msg["To"] = to_addr
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        try:
            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as server:
                server.starttls()
                server.login(settings.smtp_user, settings.smtp_password)
                server.sendmail(from_addr, [to_addr], msg.as_string())
            self._record("send_email", "ok", time.time() - t0, detail=f"to {to_addr}")
            return {"sent": True, "to": to_addr}
        except Exception as e:  # noqa: BLE001
            self._record("send_email", "error", time.time() - t0, detail=str(e))
            return {"sent": False, "reason": str(e)}
