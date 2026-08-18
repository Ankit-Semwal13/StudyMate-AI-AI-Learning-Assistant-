"""
RAG (Retrieval-Augmented Generation) question answering step: embeds the
user's question, retrieves the most relevant transcript chunks from FAISS,
and asks the LLM to answer using only that retrieved context - citing the
timestamps it drew from.
"""
from __future__ import annotations

import config
from embeddings.vector_store import VectorStore
from utils.helpers import format_timestamp
from llm import client

SYSTEM_PROMPT = (
    "You answer questions about a video using only the transcript excerpts "
    "provided below. If the excerpts don't contain the answer, say so "
    "honestly instead of guessing."
)

ANSWER_PROMPT = """\
Transcript excerpts (each tagged with its timestamp in the video):

{context}

Question: {question}

Answer the question using only the excerpts above. Be concise. If useful,
mention which timestamp(s) support your answer.
"""


def answer_question(question: str, store: VectorStore, top_k: int = config.RAG_TOP_K) -> dict:
    """
    Returns {"answer": str, "sources": [{"time": "MM:SS", "seconds": float, "text": str}]}
    """
    results = store.search(question, top_k=top_k)
    if not results:
        return {
            "answer": "The video hasn't been processed yet, so there's nothing to search.",
            "sources": [],
        }

    context_blocks = []
    sources = []
    for chunk, score in results:
        ts = format_timestamp(chunk.start)
        context_blocks.append(f"[{ts}] {chunk.text}")
        sources.append({"time": ts, "seconds": chunk.start, "text": chunk.text, "score": score})

    context = "\n\n".join(context_blocks)
    answer = client.chat(
        ANSWER_PROMPT.format(context=context, question=question),
        system=SYSTEM_PROMPT,
    )

    return {"answer": answer, "sources": sources}
