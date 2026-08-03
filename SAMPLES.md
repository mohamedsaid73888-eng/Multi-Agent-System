# Sample data for Argus

Ready-made inputs to demo the full pipeline — one of each source type. They all
share a fictional "MENA cloud market" theme so the brief reads coherently.

| Source type in the UI | Use this |
|---|---|
| **Market report(s) — PDF** | upload [`market_report.pdf`](market_report.pdf) |
| **Metrics / sales — CSV/TSV** | upload [`sales_metrics.csv`](sales_metrics.csv) |
| **Paste raw text** | paste the contents of [`analyst_notes.txt`](analyst_notes.txt) |
| **Competitor / market web page URL** | `https://en.wikipedia.org/wiki/Cloud_computing` |
| **Public Google Sheet URL** | see below (make one from the CSV) |

## How to run
1. Open the app → **📥 Signals** tab.
2. **Upload** `market_report.pdf` and `sales_metrics.csv`.
3. Paste the text from `analyst_notes.txt` into **Paste raw text**.
4. Paste the Wikipedia URL into **Competitor / market web page URL** and press **Enter**.
5. (Optional) paste a Google Sheet URL (below).
6. Click **🛰️ Run Argus** → check the **Brief**, **Dashboard**, **Ask**, **Email** tabs.

## Website URL (ready to paste)
```
https://en.wikipedia.org/wiki/Cloud_computing
```
Wikipedia is static HTML, so the web retriever pulls clean text. (Any content-rich
public page works; JavaScript-heavy pages may return little text.)

## Public Google Sheet URL (30-second setup)
I can't create a sheet on your Google account, so make a public one from the sample CSV:
1. Go to <https://sheets.google.com> → **File → Import** → upload `sales_metrics.csv`
   (or just paste the rows into a new sheet).
2. **Share** (top-right) → **General access → Anyone with the link → Viewer**.
3. Copy the URL — it looks like:
   `https://docs.google.com/spreadsheets/d/<SHEET_ID>/edit#gid=0`
4. Paste it into **Public Google Sheet URL** in the app and press **Enter**.

The app extracts the sheet ID and pulls it via the public CSV-export endpoint — no
Google API key needed, as long as the sheet is shared "anyone with the link".
