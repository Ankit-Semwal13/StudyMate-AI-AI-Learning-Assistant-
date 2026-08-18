"""
Embeddings + vector store step: embeds each transcript chunk with a local
sentence-transformers model and indexes it in FAISS for fast semantic
search. Both are free/local - no API key, no external service.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np

import config
from preprocess.chunking import Chunk

_embedder = None  # lazily-loaded, cached SentenceTransformer


def _get_embedder():
    global _embedder
    if _embedder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run "
                "`pip install sentence-transformers` first."
            ) from exc
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL)
    return _embedder


class VectorStore:
    """A thin wrapper around a FAISS flat index + the chunk metadata."""

    def __init__(self):
        self.index = None
        self.chunks: list[Chunk] = []

    def build(self, chunks: list[Chunk]) -> "VectorStore":
        import faiss

        self.chunks = chunks
        if not chunks:
            return self

        embedder = _get_embedder()
        vectors = embedder.encode(
            [c.text for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype="float32")

        dim = vectors.shape[1]
        # Inner product on normalized vectors == cosine similarity.
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(vectors)
        return self

    def search(self, query: str, top_k: int = config.RAG_TOP_K) -> list[tuple[Chunk, float]]:
        if self.index is None or not self.chunks:
            return []

        embedder = _get_embedder()
        q_vec = embedder.encode([query], normalize_embeddings=True)
        q_vec = np.asarray(q_vec, dtype="float32")

        scores, indices = self.index.search(q_vec, min(top_k, len(self.chunks)))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, path: str | Path) -> None:
        import faiss

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(path / "index.faiss"))
        with open(path / "chunks.pkl", "wb") as f:
            pickle.dump(self.chunks, f)

    @classmethod
    def load(cls, path: str | Path) -> Optional["VectorStore"]:
        import faiss

        path = Path(path)
        index_file = path / "index.faiss"
        chunks_file = path / "chunks.pkl"
        if not index_file.exists() or not chunks_file.exists():
            return None

        store = cls()
        store.index = faiss.read_index(str(index_file))
        with open(chunks_file, "rb") as f:
            store.chunks = pickle.load(f)
        return store
