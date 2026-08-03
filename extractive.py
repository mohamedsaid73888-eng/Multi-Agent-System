"""
Dependency-free extractive summarizer + keyword extractor.

Used as an automatic fallback when no LLM key is available or when every model
in the fallback chain fails. It is a lightweight frequency-based (TextRank-ish)
approach: score sentences by the summed frequency of their non-stopword tokens
and return the top-k in original order.

This keeps the whole pipeline runnable end-to-end offline, which is important
for the demo and for the reliability story in the evaluation.
"""
from __future__ import annotations

import re
from collections import Counter
from typing import List

_STOPWORDS = set(
    """
a an the and or but if then else when while for to of in on at by with from as
is are was were be been being this that these those it its it's i you he she we
they them his her their our your my me us do does did has have had not no nor so
than too very can will just about into over under again further once here there
all any both each few more most other some such only own same s t don should now
""".split()
)

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]+")


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    # keep reasonably-sized sentences
    return [p.strip() for p in parts if len(p.strip()) > 20]


def _tokens(text: str) -> List[str]:
    return [w.lower() for w in _WORD.findall(text) if w.lower() not in _STOPWORDS]


def keywords(text: str, top_n: int = 10) -> List[str]:
    freq = Counter(_tokens(text))
    return [w for w, _ in freq.most_common(top_n)]


def summarize(text: str, max_sentences: int = 5) -> str:
    sentences = split_sentences(text)
    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    freq = Counter(_tokens(text))
    if not freq:
        return " ".join(sentences[:max_sentences])

    # normalise frequencies
    top = freq.most_common(1)[0][1]
    norm = {w: c / top for w, c in freq.items()}

    scored = []
    for i, sent in enumerate(sentences):
        toks = _tokens(sent)
        if not toks:
            continue
        score = sum(norm.get(t, 0) for t in toks) / (len(toks) ** 0.5)
        scored.append((score, i, sent))

    scored.sort(reverse=True)
    chosen = sorted(scored[:max_sentences], key=lambda x: x[1])  # original order
    return " ".join(s for _, _, s in chosen)
