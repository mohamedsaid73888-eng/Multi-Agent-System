"""
RetrievalAgent  (Objective 1: Data Retrieval)

Gathers data from heterogeneous sources and normalizes everything into
`Document` objects:

  * PDF        -> text extraction (pypdf)
  * CSV / TSV  -> pandas -> compact textual + tabular representation
  * Google Sheet (public) -> exported as CSV via the gviz endpoint
  * API (JSON) -> GET request, pretty-printed / flattened
  * Web page   -> fetched HTML, tags stripped to readable text

Each loader is defensive: a failure in one source is recorded in the trace but
does not abort retrieval of the others.
"""
from __future__ import annotations

import io
import json
import re
import time
from typing import Any, Dict, List, Optional

import requests

from agents.base import BaseAgent, Document
from config import settings


class RetrievalAgent(BaseAgent):
    name = "RetrievalAgent"

    # -- PDF ---------------------------------------------------------------
    def from_pdf_bytes(self, data: bytes, title: str = "PDF") -> Document:
        def _load() -> Document:
            from pypdf import PdfReader

            reader = PdfReader(io.BytesIO(data))
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            text = "\n".join(pages).strip()
            return Document(
                source_id=title,
                source_type="pdf",
                title=title,
                text=text[: settings.max_chars_per_source],
                meta={"pages": len(reader.pages)},
            )

        return self._timed(f"load_pdf:{title}", _load)

    # -- CSV / TSV ---------------------------------------------------------
    def from_csv_bytes(self, data: bytes, title: str = "CSV") -> Document:
        def _load() -> Document:
            import pandas as pd

            sep = "\t" if title.lower().endswith(".tsv") else ","
            df = pd.read_csv(io.BytesIO(data), sep=sep)
            return self._df_to_document(df, title, source_type="csv")

        return self._timed(f"load_csv:{title}", _load)

    def from_google_sheet(self, sheet_url: str, title: str = "GoogleSheet") -> Document:
        """
        Accepts a normal Google Sheets share URL of a *public* sheet and pulls it
        via the CSV export endpoint. No auth / API key required for public sheets.
        """

        def _load() -> Document:
            import pandas as pd

            m = re.search(r"/spreadsheets/d/([a-zA-Z0-9-_]+)", sheet_url)
            if not m:
                raise ValueError("Could not parse a spreadsheet ID from the URL.")
            sheet_id = m.group(1)
            gid_m = re.search(r"[#&]gid=(\d+)", sheet_url)
            gid = gid_m.group(1) if gid_m else "0"
            export = (
                f"https://docs.google.com/spreadsheets/d/{sheet_id}"
                f"/export?format=csv&gid={gid}"
            )
            resp = requests.get(export, timeout=settings.request_timeout)
            resp.raise_for_status()
            df = pd.read_csv(io.BytesIO(resp.content))
            return self._df_to_document(df, title, source_type="sheet")

        return self._timed(f"load_sheet:{title}", _load)

    def _df_to_document(self, df, title: str, source_type: str) -> Document:
        # keep the doc bounded but informative
        preview = df.head(50)
        text_parts = [
            f"Table '{title}' with {df.shape[0]} rows and {df.shape[1]} columns.",
            f"Columns: {', '.join(map(str, df.columns))}.",
            "",
            "Sample rows:",
            preview.to_csv(index=False),
        ]
        # add simple numeric summary for the analysis/dashboard agents
        try:
            desc = df.describe(include="all").to_csv()
            text_parts += ["", "Statistical summary:", desc]
        except Exception:
            pass
        text = "\n".join(text_parts)[: settings.max_chars_per_source]
        return Document(
            source_id=title,
            source_type=source_type,
            title=title,
            text=text,
            meta={
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "columns": list(map(str, df.columns)),
                "dataframe": df,  # kept for the dashboard agent
            },
        )

    # -- API (JSON) --------------------------------------------------------
    def from_api(
        self,
        url: str,
        title: str = "API",
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Document:
        def _load() -> Document:
            resp = requests.get(
                url, headers=headers or {}, params=params or {},
                timeout=settings.request_timeout,
            )
            resp.raise_for_status()
            try:
                payload = resp.json()
                text = json.dumps(payload, indent=2, ensure_ascii=False)
            except ValueError:
                text = resp.text
            return Document(
                source_id=title,
                source_type="api",
                title=title,
                text=text[: settings.max_chars_per_source],
                meta={"url": url, "status": resp.status_code},
            )

        return self._timed(f"load_api:{title}", _load)

    # -- Web page ----------------------------------------------------------
    def from_web(self, url: str, title: Optional[str] = None) -> Document:
        def _load() -> Document:
            resp = requests.get(
                url,
                timeout=settings.request_timeout,
                headers={"User-Agent": "MultiAgentSystem/1.0"},
            )
            resp.raise_for_status()
            text = self._html_to_text(resp.text)
            return Document(
                source_id=title or url,
                source_type="web",
                title=title or url,
                text=text[: settings.max_chars_per_source],
                meta={"url": url, "status": resp.status_code},
            )

        return self._timed(f"load_web:{url}", _load)

    @staticmethod
    def _html_to_text(html: str) -> str:
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
                tag.decompose()
            text = soup.get_text(separator=" ")
        except Exception:
            # crude fallback if bs4 missing
            text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()

    # -- raw text ----------------------------------------------------------
    def from_text(self, text: str, title: str = "Pasted text") -> Document:
        return self._timed(
            f"load_text:{title}",
            lambda: Document(
                source_id=title,
                source_type="text",
                title=title,
                text=text[: settings.max_chars_per_source],
            ),
        )

    # -- convenience batch runner -----------------------------------------
    def run(self, sources: List[Dict[str, Any]]) -> List[Document]:
        """
        sources: list of dicts, e.g.
          {"kind": "pdf", "bytes": b"...", "title": "report.pdf"}
          {"kind": "csv", "bytes": b"...", "title": "sales.csv"}
          {"kind": "sheet", "url": "https://docs.google.com/..."}
          {"kind": "api", "url": "https://api...."}
          {"kind": "web", "url": "https://..."}
          {"kind": "text", "text": "...", "title": "notes"}
        Returns the successfully-loaded documents (failures are traced, skipped).
        """
        docs: List[Document] = []
        for src in sources:
            kind = src.get("kind")
            try:
                if kind == "pdf":
                    docs.append(self.from_pdf_bytes(src["bytes"], src.get("title", "PDF")))
                elif kind == "csv":
                    docs.append(self.from_csv_bytes(src["bytes"], src.get("title", "CSV")))
                elif kind == "sheet":
                    docs.append(self.from_google_sheet(src["url"], src.get("title", "Sheet")))
                elif kind == "api":
                    docs.append(
                        self.from_api(
                            src["url"], src.get("title", "API"),
                            src.get("headers"), src.get("params"),
                        )
                    )
                elif kind == "web":
                    docs.append(self.from_web(src["url"], src.get("title")))
                elif kind == "text":
                    docs.append(self.from_text(src["text"], src.get("title", "Text")))
                else:
                    self._record("load_unknown", "error", 0.0, detail=f"kind={kind}")
            except Exception as e:  # noqa: BLE001
                # already traced by _timed; keep going with other sources
                continue
        return docs
