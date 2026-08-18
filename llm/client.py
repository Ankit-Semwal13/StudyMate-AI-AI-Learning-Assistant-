"""
Thin wrapper around a local Ollama server - the free LLM backend for this
project. No API key required; just have Ollama running locally
(https://ollama.com) with a model pulled, e.g. `ollama pull llama3.2`.
"""
from __future__ import annotations

import config


class OllamaUnavailableError(RuntimeError):
    """Raised when the local Ollama server can't be reached or the model
    isn't pulled yet, with a message telling the user how to fix it."""


def chat(prompt: str, system: str | None = None, model: str | None = None,
         temperature: float = 0.3) -> str:
    """Send a single-turn prompt to the local Ollama model and return the
    text response."""
    try:
        import ollama
    except ImportError as exc:
        raise RuntimeError(
            "The `ollama` Python package is not installed. Run "
            "`pip install ollama` first."
        ) from exc

    model = model or config.OLLAMA_MODEL
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    client = ollama.Client(host=config.OLLAMA_HOST, timeout=config.LLM_REQUEST_TIMEOUT)
    try:
        response = client.chat(
            model=model,
            messages=messages,
            options={"temperature": temperature},
        )
    except Exception as exc:  # ollama raises its own ResponseError, plus
        # connection errors from the underlying httpx client
        raise OllamaUnavailableError(
            f"Could not reach Ollama model '{model}' at {config.OLLAMA_HOST}.\n"
            "Make sure Ollama is installed and running "
            "(https://ollama.com), and that the model is pulled:\n"
            f"    ollama pull {model}\n"
            f"Original error: {exc}"
        ) from exc

    return response["message"]["content"].strip()
