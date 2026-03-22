"""Base class and data structures for all detection modules."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict


@dataclass
class Verdict:
    """Output from a detection module for a single claim.

    Each verdict represents a module's judgement on whether a particular
    environmental claim exhibits a specific greenwashing pattern.
    """

    item_id: str  # UUID for this verdict
    module_name: str  # e.g. "vague_claims"
    claim_id: str  # links to Claim.claim_id
    verdict: str  # "flagged", "pass", "needs_verification"
    explanation: str  # natural language explanation
    missing_info: list[str]  # what information is missing from the claim
    evidence: list[dict]  # supporting evidence references
    page: int
    claim_text: str
    section_path: list[str]
    judgment: dict | None = field(default=None)  # Full LLM JudgmentResult

    @staticmethod
    def create(
        module_name: str,
        claim_id: str,
        verdict: str,
        explanation: str,
        page: int,
        claim_text: str,
        section_path: list[str] | None = None,
        missing_info: list[str] | None = None,
        evidence: list[dict] | None = None,
        judgment: dict | None = None,
    ) -> "Verdict":
        """Convenience factory that auto-generates an item_id."""
        if judgment and "reasoning" in judgment:
            explanation = judgment["reasoning"]
        return Verdict(
            item_id=str(uuid.uuid4()),
            module_name=module_name,
            claim_id=claim_id,
            verdict=verdict,
            explanation=explanation,
            missing_info=missing_info or [],
            evidence=evidence or [],
            page=page,
            claim_text=claim_text,
            section_path=section_path or [],
            judgment=judgment,
        )


class BaseModule(ABC):
    """Base class for all detection modules.

    Every detection module (vague-claim detector, cherry-picking detector,
    etc.) inherits from this class and implements ``analyze``.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short machine-readable module identifier (e.g. ``'vague_claims'``)."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable name shown in the front-end."""
        ...

    @abstractmethod
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse a list of :class:`Claim` objects and return verdicts.

        Parameters
        ----------
        claims:
            A list of ``Claim`` dataclass instances produced by the
            :class:`ClaimExtractor`.
        **kwargs:
            Additional context that specific modules might need (e.g.
            a vector store handle for RAG-based modules).

        Returns
        -------
        list[Verdict]
            One verdict per claim that was evaluated.  Modules may choose
            to skip claims that are not relevant to their analysis.
        """
        ...
