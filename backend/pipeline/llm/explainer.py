"""LLM-based explanation generation for greenwashing verdicts.

Provides natural-language explanations using the LLM client (Ollama or
Groq) when available.  Falls back gracefully -- callers should check
the return value and use their template explanation if ``None`` is
returned.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pipeline.llm.client import LLMClient

logger = logging.getLogger(__name__)


class Explainer:
    """Generate natural language explanations using LLM."""

    def __init__(self) -> None:
        self._client = LLMClient()
        self._available: bool | None = None

    def is_available(self) -> bool:
        """Check whether an LLM backend (Ollama or Groq) is reachable."""
        if self._available is None:
            self._available = (
                self._client._check_ollama()
                or bool(os.environ.get("GROQ_API_KEY"))
            )
        return self._available

    def explain_vague_claim(
        self,
        claim_text: str,
        missing_info: list[str],
        vague_terms: list[str],
    ) -> str | None:
        """Generate explanation for why a claim is vague.

        Returns ``None`` if the LLM is unavailable or generation fails,
        so the caller can fall back to its template explanation.
        """
        if not self.is_available():
            return None

        prompt = (
            "Analyze this environmental claim from an ESG report and "
            "explain why it may be considered vague or unsubstantiated "
            "greenwashing. Be concise (2-3 sentences).\n\n"
            f'Claim: "{claim_text}"\n'
            f"Missing specifics: {', '.join(missing_info)}\n"
            f"Vague language detected: "
            f"{', '.join(vague_terms) if vague_terms else 'none'}\n\n"
            "Explain what makes this claim potentially misleading and "
            "what specific information would make it credible."
        )

        system = (
            "You are a greenwashing analyst. Give concise, factual "
            "assessments. Do not use bullet points."
        )
        try:
            return self._client.generate(prompt, system)
        except Exception:
            logger.debug("LLM vague-claim explanation failed")
            return None

    def explain_no_proof(
        self,
        claim_text: str,
        evidence_summary: str,
        missing_fields: list[str],
    ) -> str | None:
        """Generate explanation for insufficient evidence.

        Returns ``None`` if the LLM is unavailable or generation fails.
        """
        if not self.is_available():
            return None

        prompt = (
            "Analyze this environmental claim and its supporting "
            "evidence from an ESG report. Explain what proof is missing. "
            "Be concise (2-3 sentences).\n\n"
            f'Claim: "{claim_text}"\n'
            f"Evidence found: {evidence_summary}\n"
            f"Missing proof: "
            f"{', '.join(missing_fields) if missing_fields else 'none identified'}\n\n"
            "Explain what additional evidence or verification would be "
            "needed to substantiate this claim."
        )

        system = (
            "You are a greenwashing analyst. Give concise, factual "
            "assessments. Do not use bullet points."
        )
        try:
            return self._client.generate(prompt, system)
        except Exception:
            logger.debug("LLM no-proof explanation failed")
            return None
