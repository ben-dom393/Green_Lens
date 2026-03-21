"""LLM client: Ollama (local) with Groq (free API) fallback."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
import os

import httpx

from config import GROQ_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL

logger = logging.getLogger(__name__)


class LLMClient:
    """LLM client: Ollama (local) -> Groq (free API) fallback."""

    def __init__(self):
        self._ollama_available: bool | None = None

    # ------------------------------------------------------------------
    # Ollama availability check
    # ------------------------------------------------------------------

    def _check_ollama(self) -> bool:
        """Check if Ollama is running locally."""
        try:
            r = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Generation backends
    # ------------------------------------------------------------------

    def _generate_ollama(self, prompt: str, system: str | None = None) -> str:
        """Generate using local Ollama."""
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = httpx.post(
                f"{OLLAMA_BASE_URL}/api/chat",
                json={"model": OLLAMA_MODEL, "messages": messages, "stream": False},
                timeout=120.0,
            )
            response.raise_for_status()
            return response.json()["message"]["content"]
        except Exception as exc:
            logger.warning("Ollama generation failed: %s", exc)
            # Mark Ollama as unavailable so next call falls through to Groq
            self._ollama_available = False
            return self._generate_groq(prompt, system)

    def _generate_groq(self, prompt: str, system: str | None = None) -> str:
        """Generate using Groq free API."""
        api_key = os.environ.get("GROQ_API_KEY", "")
        if not api_key:
            logger.error("No GROQ_API_KEY set and Ollama unavailable.")
            return "[LLM unavailable: no Ollama and no GROQ_API_KEY set]"

        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": GROQ_MODEL, "messages": messages},
                timeout=60.0,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("Groq generation failed: %s", exc)
            return "[LLM unavailable: both Ollama and Groq failed]"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, prompt: str, system: str | None = None) -> str:
        """Generate text. Tries Ollama first, then Groq.

        Args:
            prompt: The user/query prompt.
            system: Optional system message.

        Returns:
            Generated text string. Returns a placeholder message if both
            backends are unavailable rather than raising an exception.
        """
        if self._ollama_available is None:
            self._ollama_available = self._check_ollama()
            logger.info("Ollama available: %s", self._ollama_available)

        if self._ollama_available:
            return self._generate_ollama(prompt, system)
        else:
            return self._generate_groq(prompt, system)
