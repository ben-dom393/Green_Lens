"""Module 5 -- Hidden Tradeoffs Detection.

Flags environmental claims that highlight one narrow green attribute
while ignoring larger, more material environmental impacts.  This
corresponds to the "sin of hidden tradeoffs" in greenwashing taxonomies
-- e.g. a company promoting recyclable packaging while ignoring its
massive Scope 3 supply-chain emissions.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR, ESG_TOPIC_MODEL, HF_DEVICE, TCFD_MODEL
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Scope-narrowing language patterns
# ---------------------------------------------------------------------------
_RE_SCOPE_NARROWING = [
    re.compile(
        r"\b(our\s+offices?|our\s+headquarters?|one\s+facility|"
        r"this\s+product\s+line|single\s+site)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(packaging\s+only|packaging\s+is)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bin\s+(the\s+)?(US|UK|EU|Europe|North\s+America|"
        r"our\s+\w+\s+operations?|selected\s+markets?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bat\s+our\s+\w+\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(select\s+locations?|certain\s+products?|"
        r"pilot\s+program(?:me)?s?|limited\s+range|"
        r"specific\s+facilit(?:y|ies)|chosen\s+sites?)\b",
        re.IGNORECASE,
    ),
]

# ---------------------------------------------------------------------------
# ESG-BERT category to sector ID mapping (mirrors lesser_evil.py)
# ---------------------------------------------------------------------------
_ESG_TO_SECTOR: dict[str, list[str]] = {
    "Climate Change": [
        "fossil_fuels", "mining", "cement", "aviation",
        "shipping", "steel_metals", "automotive_ice",
    ],
    "Pollution & Waste": [
        "mining", "agrochemicals", "plastics_packaging",
        "steel_metals", "cement",
    ],
    "Natural Capital": [
        "mining", "agrochemicals", "food_beverage",
    ],
    "Human Capital": ["fast_fashion"],
    "Product Liability": ["agrochemicals", "tobacco"],
    "Community Relations": ["mining", "fossil_fuels"],
    "Corporate Governance": ["banking_finance"],
    "Business Ethics": ["banking_finance"],
    "Non-ESG": [],
}


class HiddenTradeoffsModule(BaseModule):
    """Detect green claims that highlight a narrow attribute while
    ignoring larger material environmental impacts."""

    name = "hidden_tradeoffs"
    display_name = "Hidden Tradeoffs"

    def __init__(self) -> None:
        self._esg_model = None
        self._tcfd_model = None
        self._keybert = None
        self._industry_data: list[dict] | None = None

    # ------------------------------------------------------------------
    # Resource loading (lazy)
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        """Lazily load all ML models and the industry risk knowledge base."""
        # --- Industry risk data ---
        if self._industry_data is None:
            kb_path = DATA_DIR / "industry_risk.json"
            try:
                with open(kb_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                self._industry_data = raw.get("sectors", [])
                logger.info(
                    "Loaded %d sector profiles from %s",
                    len(self._industry_data),
                    kb_path,
                )
            except Exception:
                logger.exception(
                    "Failed to load industry risk KB from %s", kb_path
                )
                self._industry_data = []

        # --- ESG-BERT classifier (sector detection) ---
        if self._esg_model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._esg_model = hf_pipeline(
                    "text-classification",
                    model=ESG_TOPIC_MODEL,
                    top_k=3,
                    device=HF_DEVICE,
                )
                logger.info("Loaded ESG classifier: %s", ESG_TOPIC_MODEL)
            except Exception as e:
                logger.warning("ESG classifier not available: %s", e)
                self._esg_model = None

        # --- TCFD pillar classifier ---
        if self._tcfd_model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._tcfd_model = hf_pipeline(
                    "text-classification",
                    model=TCFD_MODEL,
                    truncation=True,
                    device=HF_DEVICE,
                )
                logger.info("Loaded TCFD classifier: %s", TCFD_MODEL)
            except Exception as e:
                logger.warning("TCFD classifier not available: %s", e)
                self._tcfd_model = None

        # --- KeyBERT for key-phrase extraction ---
        if self._keybert is None:
            try:
                from keybert import KeyBERT

                self._keybert = KeyBERT(model="all-MiniLM-L6-v2")
                logger.info("Loaded KeyBERT with all-MiniLM-L6-v2")
            except Exception as e:
                logger.warning("KeyBERT not available: %s", e)
                self._keybert = None

    # ------------------------------------------------------------------
    # Scope-narrowing detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_scope_narrowing(text: str) -> list[str]:
        """Return a list of phrases that indicate the claim's scope is
        artificially narrowed to a small subset of operations."""
        matches: list[str] = []
        for pattern in _RE_SCOPE_NARROWING:
            for m in pattern.finditer(text):
                matches.append(m.group(0))
        return matches

    # ------------------------------------------------------------------
    # Sector helpers
    # ------------------------------------------------------------------
    def _get_sector_info(self, sector_id: str) -> dict:
        """Look up a sector in industry_risk.json by its ID.

        Returns a dict with ``risk_level``, ``primary_impacts``, and
        ``expected_material_topics`` (empty values if not found).
        """
        for sector in self._industry_data or []:
            if sector.get("sector_id") == sector_id:
                return {
                    "risk_level": sector.get("risk_level", "unknown"),
                    "primary_impacts": sector.get("primary_impacts", []),
                    "expected_material_topics": sector.get(
                        "expected_material_topics", []
                    ),
                    "display_name": sector.get("display_name", sector_id),
                    "lesser_evil_note": sector.get("lesser_evil_note", ""),
                }
        return {
            "risk_level": "unknown",
            "primary_impacts": [],
            "expected_material_topics": [],
            "display_name": sector_id,
            "lesser_evil_note": "",
        }

    def _classify_sector(self, text: str) -> str:
        """Classify *text* using ESG-BERT and map the result to a sector
        ID from industry_risk.json.  Returns the best-matching sector ID
        or ``'unknown'`` if classification fails.
        """
        if self._esg_model is None:
            return "unknown"
        try:
            from collections import Counter

            sector_hits: Counter = Counter()
            results = self._esg_model(text[:512])
            for result in results:
                esg_label = result["label"]
                score = result["score"]
                hint_sectors = _ESG_TO_SECTOR.get(esg_label, [])
                for sid in hint_sectors:
                    sector_hits[sid] += score

            if sector_hits:
                return sector_hits.most_common(1)[0][0]
        except Exception:
            logger.debug("ESG classifier sector detection failed")
        return "unknown"

    # ------------------------------------------------------------------
    # TCFD classification
    # ------------------------------------------------------------------
    def _classify_tcfd(self, text: str) -> str:
        """Classify *text* into a TCFD category (e.g. 'Metrics and
        Targets', 'Strategy', etc.).  Returns ``'unknown'`` on failure.
        """
        if self._tcfd_model is None:
            return "unknown"
        try:
            out = self._tcfd_model(text[:512])
            if isinstance(out, list):
                out = out[0]
            return out.get("label", "unknown")
        except Exception:
            logger.debug("TCFD classification failed")
            return "unknown"

    # ------------------------------------------------------------------
    # Key-phrase extraction
    # ------------------------------------------------------------------
    def _extract_key_phrases(self, text: str) -> list[str]:
        """Use KeyBERT to extract the top 5 key phrases from *text*.
        Returns an empty list on failure.
        """
        if self._keybert is None:
            return []
        try:
            keywords = self._keybert.extract_keywords(
                text,
                keyphrase_ngram_range=(1, 3),
                stop_words="english",
                top_n=5,
            )
            return [kw for kw, _score in keywords]
        except Exception:
            logger.debug("KeyBERT extraction failed")
            return []

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse claims for the hidden-tradeoffs greenwashing pattern.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``claim_text``, ``page``, ``section_path``, and optionally
            ``full_context`` attributes).
        **kwargs:
            ``document_topics`` -- BERTopic result from app.py (may be None).
            ``retriever``       -- RAG retriever for evidence search.
            ``llm_judge``       -- LLM judge for final verdict.
            ``sector_result``   -- cached sector classification result.
        """
        self._load_resources()

        claim_list = list(claims)
        if not claim_list:
            return []

        document_topics: list[str] | None = kwargs.get("document_topics")
        retriever = kwargs.get("retriever")
        llm_judge = kwargs.get("llm_judge")
        cached_sector: str | None = kwargs.get("sector_result")

        verdicts: list[Verdict] = []

        for claim in claim_list:
            claim_text = claim.claim_text
            full_ctx = getattr(claim, "full_context", None)
            analysis_text = claim_text
            if full_ctx:
                analysis_text += " " + full_ctx

            # (a) Classify sector
            if cached_sector:
                sector_id = cached_sector
            else:
                sector_id = self._classify_sector(analysis_text)

            sector_info = self._get_sector_info(sector_id)
            expected_topics = sector_info["expected_material_topics"]

            # (b) Check coverage of expected material topics
            missing_topics: list[str] = []
            covered_topics: list[str] = []

            if expected_topics:
                if document_topics is not None:
                    # Use BERTopic results to check coverage
                    doc_topics_lower = [
                        t.lower() for t in document_topics
                    ]
                    for topic in expected_topics:
                        topic_lower = topic.lower()
                        found = any(
                            topic_lower in dt or dt in topic_lower
                            for dt in doc_topics_lower
                        )
                        if found:
                            covered_topics.append(topic)
                        else:
                            missing_topics.append(topic)
                elif retriever is not None:
                    # Use RAG retriever to search for each expected topic
                    for topic in expected_topics:
                        try:
                            results = retriever.invoke(topic)
                            if results and len(results) > 0:
                                covered_topics.append(topic)
                            else:
                                missing_topics.append(topic)
                        except Exception:
                            missing_topics.append(topic)
                else:
                    # No coverage information available -- all topics
                    # are considered missing
                    missing_topics = list(expected_topics)

            # (c) TCFD classification
            tcfd_category = self._classify_tcfd(claim_text)

            # (d) Extract key phrases (narrow focus of the claim)
            key_phrases = self._extract_key_phrases(claim_text)

            # (e) Detect scope-narrowing language
            narrowing_phrases = self._detect_scope_narrowing(claim_text)

            # (f) Determine coverage ratio
            total_expected = len(expected_topics) if expected_topics else 0
            total_missing = len(missing_topics)
            coverage_ratio = (
                (total_expected - total_missing) / total_expected
                if total_expected > 0
                else 1.0
            )

            # (g) Build signals for LLM Judge
            signals = {
                "page": claim.page,
                "section_path": getattr(claim, "section_path", []),
                "sector": sector_info["display_name"],
                "risk_level": sector_info["risk_level"],
                "primary_impacts": (
                    ", ".join(sector_info["primary_impacts"])
                    if sector_info["primary_impacts"]
                    else "unknown"
                ),
                "tcfd_category": tcfd_category,
                "key_phrases": ", ".join(key_phrases) if key_phrases else "none",
                "scope_narrowing": (
                    ", ".join(narrowing_phrases) if narrowing_phrases else "none"
                ),
                "expected_material_topics": (
                    ", ".join(expected_topics) if expected_topics else "none"
                ),
                "missing_material_topics": (
                    ", ".join(missing_topics) if missing_topics else "none"
                ),
                "coverage_ratio": f"{coverage_ratio:.0%}",
            }

            judgment_dict = None
            verdict_label = "pass"

            if llm_judge is not None:
                jr = llm_judge.judge_claim(
                    module_name=self.name,
                    claim_text=claim_text,
                    signals=signals,
                )
                if jr is not None:
                    verdict_label = jr.verdict
                    jr.module_signals = signals
                    judgment_dict = jr.to_dict()

            # (h) Heuristic fallback when no LLM judge is available
            if llm_judge is None:
                should_flag = False

                # Flag if >= 50% of expected topics are missing AND the
                # claim focuses on a non-material topic
                if total_expected > 0 and coverage_ratio <= 0.5:
                    # Check if the claim's key phrases overlap with the
                    # expected material topics
                    claim_covers_material = False
                    for kp in key_phrases:
                        kp_lower = kp.lower()
                        for topic in expected_topics:
                            if kp_lower in topic.lower() or topic.lower() in kp_lower:
                                claim_covers_material = True
                                break
                        if claim_covers_material:
                            break

                    if not claim_covers_material:
                        should_flag = True

                # Also flag if scope-narrowing language is present and
                # the sector is high risk
                if (
                    narrowing_phrases
                    and sector_info["risk_level"] in ("very_high", "high")
                ):
                    should_flag = True

                verdict_label = "flagged" if should_flag else "pass"

            # Build verdict
            if verdict_label == "flagged":
                impact_list = (
                    ", ".join(sector_info["primary_impacts"])
                    if sector_info["primary_impacts"]
                    else "various environmental impacts"
                )

                explanation = (
                    f"This claim appears to highlight a narrow environmental "
                    f"attribute while potentially omitting larger impacts. "
                    f"Sector: {sector_info['display_name']} "
                    f"({sector_info['risk_level'].replace('_', ' ')} risk). "
                    f"Primary sector impacts include: {impact_list}. "
                    f"Material topic coverage: {coverage_ratio:.0%} "
                    f"({total_missing}/{total_expected} expected topics "
                    f"not addressed in the document)."
                )

                missing_info_list: list[str] = []
                if missing_topics:
                    missing_info_list.append(
                        "The following material topics are not addressed: "
                        + "; ".join(missing_topics)
                    )
                if narrowing_phrases:
                    missing_info_list.append(
                        "Scope-narrowing language detected: "
                        + "; ".join(narrowing_phrases)
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
                        missing_info=missing_info_list,
                        evidence=[
                            {
                                "sector_id": sector_id,
                                "sector_name": sector_info["display_name"],
                                "risk_level": sector_info["risk_level"],
                                "tcfd_category": tcfd_category,
                                "key_phrases": key_phrases,
                                "scope_narrowing": narrowing_phrases,
                                "coverage_ratio": coverage_ratio,
                                "missing_topics": missing_topics,
                                "covered_topics": covered_topics,
                            }
                        ],
                        judgment=judgment_dict,
                    )
                )
            else:
                verdicts.append(
                    Verdict.create(
                        module_name=self.name,
                        claim_id=claim.claim_id,
                        verdict=verdict_label,
                        explanation=(
                            f"Sector identified as "
                            f"{sector_info['display_name']} "
                            f"({sector_info['risk_level'].replace('_', ' ')} "
                            f"risk). Material topic coverage: "
                            f"{coverage_ratio:.0%}. This claim does not "
                            f"exhibit the hidden-tradeoffs pattern."
                        ),
                        page=claim.page,
                        claim_text=claim.claim_text,
                        section_path=getattr(claim, "section_path", []),
                        missing_info=[],
                        evidence=[
                            {
                                "sector_id": sector_id,
                                "sector_name": sector_info["display_name"],
                                "risk_level": sector_info["risk_level"],
                                "coverage_ratio": coverage_ratio,
                            }
                        ],
                        judgment=judgment_dict,
                    )
                )

        return verdicts
