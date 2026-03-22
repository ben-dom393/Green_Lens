"""Module 4 -- Lesser of Two Evils Detection.

Flags environmental claims from companies operating in inherently
high-impact industries where isolated improvements do not offset the
core business's environmental footprint.  This corresponds to the
"sin of the lesser of two evils" in greenwashing taxonomies.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR, HF_DEVICE
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Greenwashing buzzword patterns
# ---------------------------------------------------------------------------
_RE_GREEN_BUZZWORDS = re.compile(
    r"\b("
    r"sustainable|sustainability|green|eco[- ]?friendly|clean|"
    r"carbon[- ]?neutral|net[- ]?zero|renewable|"
    r"environmentally[- ]?friendly|climate[- ]?positive|"
    r"planet[- ]?friendly|earth[- ]?friendly|"
    r"zero[- ]?emission|low[- ]?carbon|decarboni[sz]"
    r")\b",
    re.IGNORECASE,
)

# Patterns that indicate specificity (numbers, methodology, etc.)
_RE_HAS_NUMBERS = re.compile(
    r"\b\d[\d,.]*\s*"
    r"(%|percent|tonnes?|tons?|kg|GW|MW|kW|MWh|GWh|TWh|x\b|X\b|times)"
    r"|\b\d{2,}[\d,.]*\b"        # any number with 2+ digits (e.g. 100, 67, 500)
    r"|\b\d+\.\d+\b"             # decimal numbers (e.g. 1.5, 0.41)
)
_RE_HAS_METHODOLOGY = re.compile(
    r"\b("
    r"GHG Protocol|ISO\s*1[4-9]\d{3}|ISO\s*50001|TCFD|CDP|SBTi|"
    r"Science Based Targets|GRI|SASB|ISSB|LCA|life cycle|"
    r"verified by|audited by|certified by|assured by|"
    r"third[- ]party|independently verified|independently audited|"
    r"externally assured"
    r")\b",
    re.IGNORECASE,
)


class LesserEvilModule(BaseModule):
    """Detect green claims that may constitute the 'lesser of two evils'
    pattern -- improvements that do not address core business impact."""

    name = "lesser_of_two_evils"
    display_name = "Lesser of Two Evils"

    def __init__(self) -> None:
        self._sectors: list[dict] | None = None
        self._esg_classifier = None
        self._sentiment_model = None
        self._stance_model = None

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_kb(self) -> None:
        """Lazily load the industry risk knowledge base and ESG classifier."""
        if self._sectors is not None:
            return

        kb_path = DATA_DIR / "industry_risk.json"
        try:
            with open(kb_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._sectors = raw.get("sectors", [])
            logger.info(
                "Loaded %d sector profiles from %s",
                len(self._sectors),
                kb_path,
            )
        except Exception:
            logger.exception(
                "Failed to load industry risk KB from %s", kb_path
            )
            self._sectors = []

        # Load ESG-BERT classifier for model-based sector detection
        if self._esg_classifier is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._esg_classifier = hf_pipeline(
                    "text-classification",
                    model="yiyanghkust/finbert-esg-9-categories",
                    top_k=3,
                    device=HF_DEVICE,
                )
                logger.info(
                    "Loaded ESG classifier: yiyanghkust/finbert-esg-9-categories"
                )
            except Exception as e:
                logger.warning("ESG classifier not available: %s", e)
                self._esg_classifier = None

        if self._sentiment_model is None:
            try:
                from transformers import pipeline as hf_pipeline
                from config import CLIMATE_SENTIMENT_MODEL
                self._sentiment_model = hf_pipeline(
                    "text-classification", model=CLIMATE_SENTIMENT_MODEL, truncation=True,
                    device=HF_DEVICE,
                )
                logger.info("Loaded sentiment model: %s", CLIMATE_SENTIMENT_MODEL)
            except Exception as e:
                logger.warning("Sentiment model not available: %s", e)
                self._sentiment_model = None

        if self._stance_model is None:
            try:
                from transformers import pipeline as hf_pipeline
                from config import CLIMATE_STANCE_MODEL
                self._stance_model = hf_pipeline(
                    "text-classification", model=CLIMATE_STANCE_MODEL, truncation=True,
                    device=HF_DEVICE,
                )
                logger.info("Loaded stance model: %s", CLIMATE_STANCE_MODEL)
            except Exception as e:
                logger.warning("Stance model not available: %s", e)
                self._stance_model = None

    # ------------------------------------------------------------------
    # ESG category to sector risk mapping
    # ------------------------------------------------------------------
    # Maps ESG-BERT output categories to sector IDs from industry_risk.json.
    # Multiple ESG categories may map to the same high-risk sector.
    _ESG_TO_SECTOR_HINTS: dict[str, list[str]] = {
        "Climate Change": [
            "oil_and_gas", "mining", "utilities", "aviation",
            "shipping", "heavy_industry",
        ],
        "Pollution & Waste": [
            "mining", "chemicals", "heavy_industry", "utilities",
        ],
        "Natural Capital": [
            "mining", "agriculture", "forestry", "fisheries",
        ],
        "Human Capital": [],
        "Product Liability": [],
        "Community Relations": [],
        "Corporate Governance": [],
        "Business Ethics": [],
        "Non-ESG": [],
    }

    # ------------------------------------------------------------------
    # Sector detection
    # ------------------------------------------------------------------
    def _detect_sector(self, claims) -> dict | None:
        """Determine the document's sector using ESG-BERT classification
        when available, falling back to keyword counting.

        Returns the best-matching sector dict, or ``None`` if no sector
        could be determined.
        """
        # --- ESG-BERT approach (preferred) ---
        if self._esg_classifier is not None:
            try:
                sector_hits: Counter = Counter()
                # Classify a sample of claims for efficiency
                sample_texts = []
                for claim in claims:
                    text = claim.claim_text
                    full_ctx = getattr(claim, "full_context", None)
                    if full_ctx:
                        text += " " + full_ctx
                    # Truncate to avoid model max length issues
                    sample_texts.append(text[:512])

                for text in sample_texts:
                    esg_results = self._esg_classifier(text)
                    for result in esg_results:
                        esg_label = result["label"]
                        score = result["score"]
                        # Map ESG categories to sector hints
                        hint_sectors = self._ESG_TO_SECTOR_HINTS.get(
                            esg_label, []
                        )
                        for sector_id in hint_sectors:
                            sector_hits[sector_id] += score

                if sector_hits:
                    best_sector_id = sector_hits.most_common(1)[0][0]
                    for sector in self._sectors or []:
                        if sector["sector_id"] == best_sector_id:
                            return sector
            except Exception:
                logger.debug(
                    "ESG classifier sector detection failed, "
                    "falling back to keywords"
                )

        # --- Fallback: keyword counting ---
        sector_hits_kw: Counter = Counter()

        for claim in claims:
            combined = claim.claim_text.lower()
            full_ctx = getattr(claim, "full_context", None)
            if full_ctx:
                combined += " " + full_ctx.lower()

            for sector in self._sectors or []:
                for keyword in sector.get("keywords", []):
                    if keyword.lower() in combined:
                        sector_hits_kw[sector["sector_id"]] += 1

        if not sector_hits_kw:
            return None

        best_sector_id = sector_hits_kw.most_common(1)[0][0]
        for sector in self._sectors or []:
            if sector["sector_id"] == best_sector_id:
                return sector
        return None

    # ------------------------------------------------------------------
    # Claim-level analysis helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _has_green_buzzwords(text: str) -> bool:
        """Return True if *text* contains greenwashing buzzwords."""
        return bool(_RE_GREEN_BUZZWORDS.search(text))

    @staticmethod
    def _is_vague(text: str) -> bool:
        """Return True if *text* lacks quantitative data and methodology
        references (i.e. is vague)."""
        has_numbers = bool(_RE_HAS_NUMBERS.search(text))
        has_methodology = bool(_RE_HAS_METHODOLOGY.search(text))
        return not has_numbers and not has_methodology

    # ------------------------------------------------------------------
    # Sentiment & stance helpers
    # ------------------------------------------------------------------
    def _analyze_sentiment(self, text: str) -> dict:
        if self._sentiment_model is None:
            return {"label": "unknown", "score": 0.0}
        try:
            out = self._sentiment_model(text[:512])
            if isinstance(out, list):
                out = out[0]
            return {"label": out["label"], "score": round(float(out["score"]), 4)}
        except Exception:
            return {"label": "unknown", "score": 0.0}

    def _analyze_stance(self, text: str) -> dict:
        if self._stance_model is None:
            return {"label": "unknown", "score": 0.0}
        try:
            out = self._stance_model(text[:512])
            if isinstance(out, list):
                out = out[0]
            return {"label": out["label"], "score": round(float(out["score"]), 4)}
        except Exception:
            return {"label": "unknown", "score": 0.0}

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse claims for the lesser-of-two-evils greenwashing pattern.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``claim_text``, ``page``, ``section_path``, and optionally
            ``full_context`` attributes).
        **kwargs:
            Unused by this module.
        """
        self._load_kb()

        claim_list = list(claims)
        if not claim_list:
            return []

        # --- First pass: determine the document's sector ---------------
        detected_sector = self._detect_sector(claim_list)

        if detected_sector is None:
            # No sector could be determined -- pass all claims through
            logger.info(
                "Could not determine document sector; skipping lesser-evil "
                "analysis for %d claims.",
                len(claim_list),
            )
            return [
                Verdict.create(
                    module_name=self.name,
                    claim_id=claim.claim_id,
                    verdict="pass",
                    explanation=(
                        "Could not determine the company's industry sector. "
                        "Lesser-of-two-evils analysis requires sector context."
                    ),
                    page=claim.page,
                    claim_text=claim.claim_text,
                    section_path=getattr(claim, "section_path", []),
                    missing_info=[],
                    evidence=[],
                )
                for claim in claim_list
            ]

        sector_name = detected_sector["display_name"]
        risk_level = detected_sector.get("risk_level", "unknown")
        primary_impacts = detected_sector.get("primary_impacts", [])
        lesser_evil_note = detected_sector.get("lesser_evil_note", "")
        expected_topics = detected_sector.get("expected_material_topics", [])

        is_high_risk = risk_level in ("very_high", "high")

        # --- Second pass: evaluate each claim --------------------------
        verdicts: list[Verdict] = []

        for claim in claim_list:
            claim_text = claim.claim_text
            has_buzzwords = self._has_green_buzzwords(claim_text)
            claim_is_vague = self._is_vague(claim_text)
            found_buzzwords = _RE_GREEN_BUZZWORDS.findall(claim_text)
            has_numbers = bool(_RE_HAS_NUMBERS.search(claim_text))

            should_flag = False

            if is_high_risk and has_buzzwords:
                # High/very-high risk sectors: flag any claim with green
                # buzzwords regardless of specificity
                should_flag = True
            elif not is_high_risk and has_buzzwords and claim_is_vague:
                # Medium or lower risk sectors: only flag if the claim
                # also lacks quantitative backing
                should_flag = True

            # -- LLM Judge integration --
            verdict_label = "flagged" if should_flag else "pass"
            llm_judge = kwargs.get("llm_judge")
            judgment_dict = None

            if llm_judge is not None:
                sentiment = self._analyze_sentiment(claim_text)
                stance = self._analyze_stance(claim_text)
                signals_for_judge = {
                    "page": claim.page,
                    "section_path": getattr(claim, "section_path", []),
                    "sector": sector_name,
                    "risk_level": risk_level,
                    "primary_impacts": ", ".join(primary_impacts) if primary_impacts else "unknown",
                    "sentiment": f"{sentiment['label']} ({sentiment['score']})",
                    "stance": f"{stance['label']} ({stance['score']})",
                    "buzzwords": ", ".join(found_buzzwords) if found_buzzwords else "none",
                    "has_numbers": str(has_numbers),
                }
                jr = llm_judge.judge_claim(
                    module_name=self.name,
                    claim_text=claim_text,
                    signals=signals_for_judge,
                )
                if jr is not None:
                    verdict_label = jr.verdict
                    jr.module_signals = signals_for_judge
                    judgment_dict = jr.to_dict()

            if should_flag:
                # Build a factual explanation without moral judgments
                impact_list = ", ".join(primary_impacts) if primary_impacts else "various environmental impacts"

                explanation = (
                    f"This claim uses environmental marketing language in "
                    f"the context of the {sector_name} sector, which is "
                    f"classified as {risk_level.replace('_', ' ')} risk. "
                    f"The sector's primary environmental impacts include: "
                    f"{impact_list}. {lesser_evil_note}"
                )

                missing_info = []
                if expected_topics:
                    missing_info.append(
                        "The company should address these material topics: "
                        + "; ".join(expected_topics)
                    )
                if claim_is_vague:
                    missing_info.append(
                        "Quantified metrics and methodology references to "
                        "substantiate the claim"
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
                        evidence=[
                            {
                                "sector_id": detected_sector["sector_id"],
                                "sector_name": sector_name,
                                "risk_level": risk_level,
                                "primary_impacts": primary_impacts,
                                "has_green_buzzwords": has_buzzwords,
                                "claim_is_vague": claim_is_vague,
                            }
                        ],
                        judgment=judgment_dict,
                    )
                )
            else:
                # Pass -- either no buzzwords, or medium/low risk with
                # adequate specificity
                verdicts.append(
                    Verdict.create(
                        module_name=self.name,
                        claim_id=claim.claim_id,
                        verdict=verdict_label,
                        explanation=(
                            f"Sector identified as {sector_name} "
                            f"({risk_level.replace('_', ' ')} risk). "
                            f"This claim does not exhibit the lesser-of-two-"
                            f"evils pattern."
                        ),
                        page=claim.page,
                        claim_text=claim.claim_text,
                        section_path=getattr(claim, "section_path", []),
                        missing_info=[],
                        evidence=[
                            {
                                "sector_id": detected_sector["sector_id"],
                                "sector_name": sector_name,
                                "risk_level": risk_level,
                            }
                        ],
                        judgment=judgment_dict,
                    )
                )

        return verdicts
