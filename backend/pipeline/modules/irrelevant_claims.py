"""Module 3 -- Irrelevant Claims Detection.

Detects environmental claims that are technically true but carry no
informational value because the substance or practice is already banned,
required by law, or universally mandated.  This corresponds to the
"sin of irrelevance" in the TerraChoice greenwashing taxonomy.

Upgrade: combines fast regex KB matching with zero-shot classification
and an optional LLM Judge for final verdicts.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zero-shot candidate labels for irrelevance classification
# ---------------------------------------------------------------------------
_IRRELEVANCE_LABELS = [
    "legally mandated requirement",
    "banned substance compliance",
    "industry standard practice",
    "voluntary environmental action",
]


class IrrelevantClaimsModule(BaseModule):
    """Detect claims that state banned or legally required attributes as if
    they were voluntary environmental benefits."""

    name = "irrelevant_claims"
    display_name = "Irrelevant Claims"

    def __init__(self) -> None:
        self._kb_entries: list[dict] | None = None
        self._zeroshot = None

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        """Lazily load the irrelevance KB and zero-shot model."""
        self._load_kb()
        self._load_zeroshot()

    def _load_kb(self) -> None:
        """Lazily load the irrelevance knowledge base."""
        if self._kb_entries is not None:
            return

        kb_path = DATA_DIR / "irrelevance_kb.json"
        try:
            with open(kb_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._kb_entries = raw.get("entries", [])
            # Pre-compile the regex patterns for performance
            for entry in self._kb_entries:
                try:
                    entry["_compiled_re"] = re.compile(
                        entry["pattern_regex"], re.IGNORECASE
                    )
                except re.error:
                    logger.warning(
                        "Invalid regex in irrelevance KB entry '%s': %s",
                        entry.get("pattern", "?"),
                        entry["pattern_regex"],
                    )
                    entry["_compiled_re"] = None
            logger.info(
                "Loaded %d irrelevance KB entries from %s",
                len(self._kb_entries),
                kb_path,
            )
        except Exception:
            logger.exception(
                "Failed to load irrelevance KB from %s", kb_path
            )
            self._kb_entries = []

    def _load_zeroshot(self) -> None:
        """Lazily load the zero-shot classification model."""
        if self._zeroshot is None:
            try:
                from transformers import pipeline as hf_pipeline
                from config import ZEROSHOT_MODEL
                self._zeroshot = hf_pipeline(
                    "zero-shot-classification", model=ZEROSHOT_MODEL
                )
                logger.info("Loaded zero-shot model: %s", ZEROSHOT_MODEL)
            except Exception as e:
                logger.warning("Zero-shot model not available: %s", e)
                self._zeroshot = None

    # ------------------------------------------------------------------
    # Zero-shot irrelevance classification
    # ------------------------------------------------------------------
    def _classify_irrelevance(self, text: str) -> dict:
        """Run zero-shot classification against irrelevance candidate labels.

        Returns a dict mapping each label to its score, or an empty dict
        if the model is not available.
        """
        if self._zeroshot is None:
            return {}
        try:
            result = self._zeroshot(text, _IRRELEVANCE_LABELS, multi_label=False)
            return {
                label: round(score, 4)
                for label, score in zip(result["labels"], result["scores"])
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse each claim for irrelevant (legally required) attributes.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``claim_text``, ``page``, ``section_path`` attributes).
        **kwargs:
            ``llm_judge`` -- optional :class:`LLMJudge` instance for
            final verdict arbitration.

            ``regulatory_retriever`` -- optional regulatory RAG retriever
            that returns dicts with ``text``, ``source``, ``page``, and
            ``document_name`` keys.
        """
        self._load_resources()

        claim_list = list(claims)
        if not claim_list:
            return []

        llm_judge = kwargs.get("llm_judge")
        regulatory_retriever = kwargs.get("regulatory_retriever")

        verdicts: list[Verdict] = []

        for claim in claim_list:
            claim_text = claim.claim_text

            # ==============================================================
            # FAST PATH: Regex KB matching (high-confidence signal)
            # ==============================================================
            matched_entry: dict | None = None
            regex_matched = False
            matched_pattern = ""
            matched_reason = ""

            for entry in self._kb_entries or []:
                compiled_re = entry.get("_compiled_re")
                if compiled_re is None:
                    continue
                if compiled_re.search(claim_text):
                    matched_entry = entry
                    regex_matched = True
                    matched_pattern = entry["pattern"]
                    matched_reason = entry["reason"]
                    break  # first match is sufficient

            # ==============================================================
            # SLOW PATH: Zero-shot classification (runs for ALL claims)
            # ==============================================================
            zs_scores = self._classify_irrelevance(claim_text)

            # ==============================================================
            # Prepare signals for LLM Judge
            # ==============================================================
            signals_for_judge = {
                "page": claim.page,
                "section_path": getattr(claim, "section_path", []),
                "regex_match": (
                    f"Matched: {matched_pattern} - {matched_reason}"
                    if regex_matched
                    else "No regex match"
                ),
                "zeroshot_result": str(zs_scores) if zs_scores else "N/A",
            }

            # ==============================================================
            # Regulatory RAG retrieval
            # ==============================================================
            regulatory_evidence: list[dict] = []
            if regulatory_retriever is not None:
                try:
                    reg_results = regulatory_retriever.retrieve(
                        claim_text, top_k=3
                    )
                    for r in reg_results:
                        if isinstance(r, dict):
                            regulatory_evidence.append(r)
                except Exception:
                    pass

            # ==============================================================
            # LLM Judge: combine regex + zero-shot → final verdict
            # ==============================================================
            judgment_dict = None
            verdict_label = None

            if llm_judge is not None:
                jr = llm_judge.judge_claim(
                    module_name=self.name,
                    claim_text=claim_text,
                    signals=signals_for_judge,
                    kb_context={"regulatory": regulatory_evidence},
                )
                if jr is not None:
                    verdict_label = jr.verdict
                    jr.module_signals = signals_for_judge
                    judgment_dict = jr.to_dict()

            # ==============================================================
            # Fallback verdict (no LLM Judge or LLM unavailable)
            # ==============================================================
            if verdict_label is None:
                if regex_matched:
                    # Regex KB matched → high-confidence flag
                    verdict_label = "flagged"
                elif zs_scores:
                    # Check if zero-shot strongly suggests legal/banned
                    legally_mandated_score = zs_scores.get(
                        "legally mandated requirement", 0.0
                    )
                    banned_substance_score = zs_scores.get(
                        "banned substance compliance", 0.0
                    )
                    if (
                        legally_mandated_score > 0.6
                        or banned_substance_score > 0.6
                    ):
                        verdict_label = "needs_verification"
                    else:
                        verdict_label = "pass"
                else:
                    verdict_label = "pass"

            # ==============================================================
            # Build explanation and evidence
            # ==============================================================
            if regex_matched:
                regulation = matched_entry["regulation"]
                applies_to = matched_entry.get("applies_to", [])

                explanation = (
                    f"This claim mentions '{matched_pattern}', which is "
                    f"{matched_reason} Under {regulation}, this is not a "
                    f"meaningful environmental distinction."
                )

                missing_info = [
                    "Specific environmental benefit beyond regulatory compliance",
                    "Quantified improvement over the legal baseline",
                ]
                if applies_to:
                    missing_info.append(
                        "Context explaining relevance to the specific "
                        "product category ("
                        + ", ".join(applies_to)
                        + ")"
                    )

                evidence_list = [
                    {
                        "kb_pattern": matched_pattern,
                        "regulation": regulation,
                        "applies_to": applies_to,
                        "zeroshot_scores": zs_scores if zs_scores else {},
                    }
                ]
            elif verdict_label == "needs_verification" and zs_scores:
                # Zero-shot triggered needs_verification without regex match
                top_label = max(zs_scores, key=zs_scores.get) if zs_scores else "N/A"
                top_score = zs_scores.get(top_label, 0.0) if zs_scores else 0.0

                explanation = (
                    f"Zero-shot classification suggests this claim may "
                    f"relate to '{top_label}' (score: {top_score:.2f}). "
                    f"No regex KB match was found, but the claim warrants "
                    f"further review."
                )

                missing_info = [
                    "Confirmation whether this attribute is legally mandated",
                    "Evidence of voluntary environmental action beyond compliance",
                ]

                evidence_list = [
                    {
                        "zeroshot_scores": zs_scores,
                        "zeroshot_top_label": top_label,
                        "zeroshot_top_score": top_score,
                    }
                ]
            else:
                explanation = (
                    "No irrelevant or legally mandated attribute "
                    "patterns detected in this claim."
                )
                missing_info = []
                evidence_list = []
                if zs_scores:
                    evidence_list = [{"zeroshot_scores": zs_scores}]

            # Add regulatory evidence to evidence list if present
            if regulatory_evidence:
                for reg in regulatory_evidence:
                    evidence_list.append(
                        {
                            "evidence_type": "regulatory",
                            "text": reg.get("text", ""),
                            "source": reg.get("source", ""),
                            "page": reg.get("page", 0),
                            "document_name": reg.get("document_name", ""),
                        }
                    )

            verdicts.append(
                Verdict.create(
                    module_name=self.name,
                    claim_id=claim.claim_id,
                    verdict=verdict_label,
                    explanation=explanation,
                    page=claim.page,
                    claim_text=claim.claim_text,
                    section_path=getattr(claim, "section_path", []),
                    missing_info=missing_info,
                    evidence=evidence_list,
                    judgment=judgment_dict,
                )
            )

        return verdicts
