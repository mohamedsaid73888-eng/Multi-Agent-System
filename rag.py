"""
RAG retriever for "Ask Argus".

The source documents from a run are split into overlapping chunks and indexed;
at question time we retrieve only the top-k most relevant chunks and feed those
to the LLM (retrieval-augmented generation) instead of stuffing every source
into the prompt.

Two interchangeable backends share one interface (`build` / `retrieve`):

  * DenseRagIndex  - semantic retrieval with a HuggingFace sentence-transformer
                     (default: all-MiniLM-L6-v2) + a FAISS vector index. This is
                     the default when the deps are installed.
  * RagIndex       - sparse TF-IDF + cosine on numpy only. Automatic fallback so
                     the app still runs on constrained hosts / offline with no
                     heavy dependencies.

`build_index()` picks the dense backend when available and degrades to sparse
otherwise, so callers never care which one they got.
"""
from __future__ import annotations

import math
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np

from utils.extractive import _tokens, split_sentences

Chunk = Dict[str, str]  # {"title": str, "text": str}

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


def chunk_document(title: str, text: str, max_chars: int = 700,
                   overlap_sentences: int = 1) -> List[Chunk]:
    """Split a document into ~max_chars chunks on sentence boundaries."""
    sents = split_sentences(text or "")
    if not sents:
        text = (text or "").strip()
        return [{"title": title, "text": text}] if text else []

    chunks: List[Chunk] = []
    cur: List[str] = []
    cur_len = 0
    for s in sents:
        if cur and cur_len + len(s) > max_chars:
            chunks.append({"title": title, "text": " ".join(cur)})
            cur = cur[-overlap_sentences:] if overlap_sentences else []
            cur_len = sum(len(x) for x in cur)
        cur.append(s)
        cur_len += len(s)
    if cur:
        chunks.append({"title": title, "text": " ".join(cur)})
    return chunks


class RagIndex:
    """TF-IDF + cosine retriever over chunked documents (sparse RAG)."""

    kind = "sparse (TF-IDF)"

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._vocab: Dict[str, int] = {}
        self._idf: np.ndarray | None = None
        self._matrix: np.ndarray | None = None   # (n_chunks, vocab) L2-normalized

    @property
    def size(self) -> int:
        return len(self.chunks)

    def build(self, documents: List[Chunk], max_chars: int = 700) -> "RagIndex":
        for d in documents:
            self.chunks.extend(
                chunk_document(d.get("title", "source"), d.get("text", ""), max_chars)
            )

        token_lists = [_tokens(c["text"]) for c in self.chunks]
        df: Counter = Counter()
        for toks in token_lists:
            for t in set(toks):
                df[t] += 1

        self._vocab = {t: i for i, t in enumerate(df)}
        n, v = len(self.chunks), len(self._vocab)
        if n == 0 or v == 0:
            self._matrix, self._idf = np.zeros((n, v)), np.zeros(v)
            return self

        idf = np.zeros(v)
        for t, i in self._vocab.items():
            idf[i] = math.log((1 + n) / (1 + df[t])) + 1.0

        m = np.zeros((n, v))
        for r, toks in enumerate(token_lists):
            for t, c in Counter(toks).items():
                m[r, self._vocab[t]] = c
        m = m * idf

        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = m / norms
        self._idf = idf
        return self

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
        """Return up to k (chunk, similarity) pairs, most relevant first."""
        if self._matrix is None or self._idf is None or self.size == 0:
            return []

        q = np.zeros(len(self._vocab))
        for t, c in Counter(_tokens(query)).items():
            i = self._vocab.get(t)
            if i is not None:
                q[i] = c
        q = q * self._idf

        norm = np.linalg.norm(q)
        if norm == 0:
            return []
        q = q / norm

        sims = self._matrix @ q
        order = np.argsort(-sims)[:k]
        return [(self.chunks[i], float(sims[i])) for i in order if sims[i] > 0]


class DenseRagIndex:
    """Semantic retriever: HuggingFace sentence-transformer embeddings + FAISS."""

    kind = "dense (embeddings + FAISS)"

    # Class-level model cache so the embedding model loads once per process.
    _model = None

    def __init__(self) -> None:
        self.chunks: List[Chunk] = []
        self._index = None  # faiss.IndexFlatIP

    @property
    def size(self) -> int:
        return len(self.chunks)

    @classmethod
    def _load_model(cls):
        if cls._model is None:
            from sentence_transformers import SentenceTransformer

            cls._model = SentenceTransformer(EMBEDDING_MODEL)
        return cls._model

    def _embed(self, texts: List[str]) -> np.ndarray:
        model = self._load_model()
        # normalize so inner product == cosine similarity
        emb = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return emb.astype("float32")

    def build(self, documents: List[Chunk], max_chars: int = 700) -> "DenseRagIndex":
        import faiss

        for d in documents:
            self.chunks.extend(
                chunk_document(d.get("title", "source"), d.get("text", ""), max_chars)
            )
        if not self.chunks:
            return self

        emb = self._embed([c["text"] for c in self.chunks])
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        self._index = index
        return self

    def retrieve(self, query: str, k: int = 4) -> List[Tuple[Chunk, float]]:
        if self._index is None or self.size == 0:
            return []
        q = self._embed([query])
        k = min(k, self.size)
        scores, idxs = self._index.search(q, k)
        out: List[Tuple[Chunk, float]] = []
        for score, i in zip(scores[0], idxs[0]):
            if i != -1 and score > 0:
                out.append((self.chunks[i], float(score)))
        return out


def build_index(documents: List[Chunk], max_chars: int = 700):
    """Build a dense (embeddings + FAISS) index when the deps are available;
    fall back to the sparse TF-IDF index otherwise.

    Set RAG_BACKEND=sparse to force the lightweight retriever (useful on
    memory-constrained hosts like Streamlit Community Cloud where loading a
    sentence-transformer can be too heavy)."""
    if os.getenv("RAG_BACKEND", "dense").strip().lower() == "sparse":
        return RagIndex().build(documents, max_chars)
    try:
        return DenseRagIndex().build(documents, max_chars)
    except Exception as e:  # noqa: BLE001 - any import/runtime issue -> degrade
        print(f"[rag] dense retriever unavailable ({e}); using TF-IDF fallback.",
              file=sys.stderr)
        return RagIndex().build(documents, max_chars)
