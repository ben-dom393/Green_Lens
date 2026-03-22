"""Module 1 -- Vague Claims Detection.

Detects environmental claims that lack specificity by combining a
ClimateBERT specificity classifier with rule-based lexicon matching.
This is the simplest detection module in the Green Lens pipeline.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR, HF_DEVICE, SPECIFICITY_MODEL, SPECIFICITY_THRESHOLD, ZEROSHOT_MODEL
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regular expressions for specificity-positive signal detection
# ---------------------------------------------------------------------------
_RE_NUMBER = re.compile(r"\b\d[\d,.]*\b")
_RE_PERCENTAGE = re.compile(r"\b\d[\d,.]*\s*%")
_RE_DATE = re.compile(
    r"\b(by\s+)?20[2-9]\d\b"            # "by 2030", "2025"
    r"|"
    r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"  # 03/15/2030
    r"|"
    r"\b(?:January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+\d{4}\b",
    re.IGNORECASE,
)
_RE_METHODOLOGY = re.compile(
    r"\b("
    r"GHG Protocol|ISO\s*1[4-9]\d{3}|ISO\s*50001|TCFD|CDP|SBTi|"
    r"Science Based Targets|GRI|SASB|ISSB|LCA|life cycle|"
    r"verified by|audited by|certified by|assured by|"
    r"third[- ]party|independently verified|independently audited|"
    r"externally assured"
    r")\b",
    re.IGNORECASE,
)
_RE_SCOPE_BOUNDARY = re.compile(
    r"\b("
    r"[Ss]cope\s*[123]|"
    r"Scope\s*1\+2|Scope\s*1\+2\+3|"
    r"operational boundary|organisational boundary|organizational boundary|"
    r"cradle[- ]to[- ]gate|cradle[- ]to[- ]grave|cradle[- ]to[- ]cradle|"
    r"market[- ]based|location[- ]based|"
    r"facility|facilities|global operations|"
    r"absolute emissions|intensity ratio"
    r")\b",
    re.IGNORECASE,
)

# Threshold: if the ratio of vague terms to total word count exceeds this
# the claim is considered rule-based vague.
_VAGUE_RATIO_THRESHOLD = 0.04
# Minimum vague term count to flag regardless of ratio
_VAGUE_MIN_COUNT = 3


class VagueClaimsModule(BaseModule):
    """Detect claims that use vague, non-specific environmental language."""

    name = "vague_claims"
    display_name = "Vague Claims"

    def __init__(self) -> None:
        self._model = None
        self._commitment_model = None
        self._zs_classifier = None
        self._keybert = None
        self._lexicon: dict | None = None

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        """Lazily load the ClimateBERT specificity model and vague lexicon."""
        if self._lexicon is None:
            lexicon_path = DATA_DIR / "vague_lexicon.json"
            try:
                with open(lexicon_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                self._lexicon = raw.get("categories", raw)
                logger.info("Loaded vague lexicon from %s", lexicon_path)
            except Exception:
                logger.exception("Failed to load vague lexicon from %s", lexicon_path)
                self._lexicon = {}

        if self._model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._model = hf_pipeline(
                    "text-classification",
                    model=SPECIFICITY_MODEL,
                    truncation=True,
                    device=HF_DEVICE,
                )
                logger.info("Loaded specificity model: %s", SPECIFICITY_MODEL)
            except Exception:
                logger.warning(
                    "Could not load specificity model '%s'. "
                    "Falling back to rule-based checks only.",
                    SPECIFICITY_MODEL,
                )
                self._model = None

        if self._commitment_model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._commitment_model = hf_pipeline(
                    "text-classification",
                    model="climatebert/distilroberta-base-climate-commitment",
                    truncation=True,
                    device=HF_DEVICE,
                )
                logger.info(
                    "Loaded commitment model: "
                    "climatebert/distilroberta-base-climate-commitment"
                )
            except Exception:
                logger.warning(
                    "Could not load commitment model "
                    "'climatebert/distilroberta-base-climate-commitment'. "
                    "Commitment analysis will be skipped."
                )
                self._commitment_model = None

        if self._zs_classifier is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._zs_classifier = hf_pipeline(
                    "zero-shot-classification",
                    model=ZEROSHOT_MODEL,
                    device=HF_DEVICE,
                )
                logger.info("Loaded zero-shot classifier: %s", ZEROSHOT_MODEL)
            except Exception:
                logger.warning(
                    "Could not load zero-shot classifier '%s'. "
                    "Zero-shot fallback will be unavailable.",
                    ZEROSHOT_MODEL,
                )
                self._zs_classifier = None

        if self._keybert is None:
            try:
                from keybert import KeyBERT
                self._keybert = KeyBERT(model="all-MiniLM-L6-v2")
                logger.info("Loaded KeyBERT model")
            except Exception as e:
                logger.warning("KeyBERT not available: %s", e)
                self._keybert = None

    # ------------------------------------------------------------------
    # KeyBERT key-phrase extraction
    # ------------------------------------------------------------------
    def _extract_key_phrases(self, text: str) -> list[str]:
        if self._keybert is None:
            return []
        try:
            keywords = self._keybert.extract_keywords(
                text, keyphrase_ngram_range=(1, 3), stop_words="english", top_n=5
            )
            return [kw[0] for kw in keywords]
        except Exception:
            return []

    # ------------------------------------------------------------------
    # ClimateBERT specificity model
    # ------------------------------------------------------------------
    def _run_specificity_model(self, texts: list[str]) -> list[dict]:
        """Run the ClimateBERT specificity classifier on a batch of texts.

        Returns a list of dicts with keys ``label`` (``"specific"`` or
        ``"vague"``) and ``score`` (float).  If the model is unavailable
        every text is reported as ``"vague"`` with score 0.0.
        """
        if self._model is None:
            return [{"label": "vague", "score": 0.0}] * len(texts)

        results: list[dict] = []
        try:
            raw_outputs = self._model(texts, batch_size=16)
            for out in raw_outputs:
                label_raw = out["label"].lower()
                score = float(out["score"])
                # The model may use LABEL_0/LABEL_1 or specific/vague.
                # ClimateBERT specificity: 0 = vague, 1 = specific
                if label_raw in ("specific", "label_1", "1"):
                    label = "specific"
                else:
                    label = "vague"
                results.append({"label": label, "score": score})
        except Exception:
            logger.exception("Specificity model inference failed")
            results = [{"label": "vague", "score": 0.0}] * len(texts)
        return results

    # ------------------------------------------------------------------
    # Rule-based specificity check
    # ------------------------------------------------------------------
    def _rule_based_check(self, text: str) -> dict:
        """Analyse *text* for vague terms and positive specificity signals.

        Returns
        -------
        dict
            ``vague_terms_found``  -- list of (term, category) tuples
            ``positive_signals``   -- list of signal descriptions
            ``missing_info``       -- list of absent information types
            ``vague_term_count``   -- int
            ``positive_signal_count`` -- int
        """
        text_lower = text.lower()

        # --- Vague term scan -------------------------------------------
        vague_terms_found: list[tuple[str, str]] = []
        for category_name, category_data in (self._lexicon or {}).items():
            # Skip the specificity_positive category -- those are good signals
            if category_name == "specificity_positive":
                continue
            terms = category_data.get("terms", []) if isinstance(category_data, dict) else []
            for term in terms:
                if term.lower() in text_lower:
                    vague_terms_found.append((term, category_name))

        # --- Positive signal scan --------------------------------------
        positive_signals: list[str] = []

        if _RE_NUMBER.search(text):
            positive_signals.append("quantitative figure")
        if _RE_PERCENTAGE.search(text):
            positive_signals.append("percentage value")
        if _RE_DATE.search(text):
            positive_signals.append("specific date or year")
        if _RE_METHODOLOGY.search(text):
            positive_signals.append("methodology or standard reference")
        if _RE_SCOPE_BOUNDARY.search(text):
            positive_signals.append("scope or boundary definition")

        # Also check specificity_positive lexicon terms
        sp_category = (self._lexicon or {}).get("specificity_positive", {})
        sp_terms = sp_category.get("terms", []) if isinstance(sp_category, dict) else []
        for term in sp_terms:
            if term.lower() in text_lower:
                positive_signals.append(f"specificity term: {term}")
                break  # one match from lexicon is enough to count

        # --- Missing information list ----------------------------------
        missing_info: list[str] = []
        if not _RE_NUMBER.search(text) and not _RE_PERCENTAGE.search(text):
            missing_info.append("quantitative metrics")
        if not _RE_DATE.search(text):
            missing_info.append("specific timeline")
        if not _RE_METHODOLOGY.search(text):
            missing_info.append("methodology reference")
        if not _RE_SCOPE_BOUNDARY.search(text):
            missing_info.append("scope/boundary definition")

        return {
            "vague_terms_found": vague_terms_found,
            "positive_signals": positive_signals,
            "missing_info": missing_info,
            "vague_term_count": len(vague_terms_found),
            "positive_signal_count": len(positive_signals),
        }

    # ------------------------------------------------------------------
    # ClimateBERT commitment model
    # ------------------------------------------------------------------
    def _run_commitment_model(self, text: str) -> dict:
        """Run the ClimateBERT commitment/action classifier on *text*.

        Returns a dict with keys ``label`` (``"commitment"`` or
        ``"action"``) and ``score`` (float).  If the model is unavailable
        returns ``{"label": "unknown", "score": 0.0}``.
        """
        if self._commitment_model is None:
            return {"label": "unknown", "score": 0.0}
        try:
            out = self._commitment_model(text)
            # HuggingFace pipeline returns a list of dicts
            if isinstance(out, list):
                out = out[0]
            label_raw = out["label"].lower()
            score = float(out["score"])
            # Normalise: the model may use LABEL_0/LABEL_1 or
            # commitment/action labels
            if label_raw in ("commitment", "label_0", "0"):
                label = "commitment"
            else:
                label = "action"
            return {"label": label, "score": score}
        except Exception:
            logger.exception("Commitment model inference failed")
            return {"label": "unknown", "score": 0.0}

    # ------------------------------------------------------------------
    # Zero-shot classification fallback
    # ------------------------------------------------------------------
    def _zero_shot_classify(self, text: str) -> dict:
        """Zero-shot classification for edge cases where ClimateBERT
        specificity confidence is low (0.4-0.6 range)."""
        if self._zs_classifier is None:
            return {"label": "unknown", "score": 0.0}
        try:
            result = self._zs_classifier(text, candidate_labels=[
                "specific measurable environmental commitment",
                "vague unsubstantiated environmental claim",
                "concrete environmental action with evidence",
            ])
            return {"label": result["labels"][0], "score": result["scores"][0]}
        except Exception:
            logger.exception("Zero-shot classification failed")
            return {"label": "unknown", "score": 0.0}

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse each claim for vagueness and return verdicts.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``text``, ``page``, ``section_path`` attributes).
        **kwargs:
            Unused by this module.
        """
        self._load_resources()

        # Collect claim texts for batched model inference
        claim_list = list(claims)
        if not claim_list:
            return []

        texts = [c.claim_text for c in claim_list]
        model_results = self._run_specificity_model(texts)

        verdicts: list[Verdict] = []

        for claim, model_out in zip(claim_list, model_results):
            rule_out = self._rule_based_check(claim.claim_text)

            # ---- Decision logic ----
            model_says_vague = (
                model_out["label"] == "vague"
                and model_out["score"] >= SPECIFICITY_THRESHOLD
            )

            # ---- Zero-shot fallback for uncertain specificity scores ----
            zs_result = {"label": "unknown", "score": 0.0}
            if (
                model_out["score"] >= 0.4
                and model_out["score"] <= 0.6
            ):
                zs_result = self._zero_shot_classify(claim.claim_text)
                if zs_result["label"] == "vague unsubstantiated environmental claim" and zs_result["score"] > 0.5:
                    model_says_vague = True
                elif zs_result["label"] == "specific measurable environmental commitment" and zs_result["score"] > 0.5:
                    model_says_vague = False

            word_count = max(len(claim.claim_text.split()), 1)
            vague_ratio = rule_out["vague_term_count"] / word_count
            rules_say_vague = (
                vague_ratio >= _VAGUE_RATIO_THRESHOLD
                or rule_out["vague_term_count"] >= _VAGUE_MIN_COUNT
            )

            # Strong positive signals can rescue a borderline claim
            has_strong_positives = rule_out["positive_signal_count"] >= 3

            # ---- Commitment model adjustment ----
            commitment_out = self._run_commitment_model(claim.claim_text)

            if (model_says_vague or rules_say_vague) and not has_strong_positives:
                verdict_label = "flagged"
            elif model_says_vague or rules_say_vague:
                # Borderline -- has some issues but also some concrete info
                verdict_label = "needs_verification"
            else:
                verdict_label = "pass"

            # Adjust verdict based on commitment model:
            # - Vague + commitment only (no action) → more confident flag
            # - Vague + actual action WITH positive signals → downgrade to needs_verification
            # - Vague + actual action WITHOUT positive signals → stay flagged (vague action is still vague)
            if commitment_out["label"] != "unknown" and commitment_out["score"] > 0.5:
                if commitment_out["label"] == "commitment" and verdict_label in ("flagged", "needs_verification"):
                    # Commitment without action strengthens the vague flag
                    verdict_label = "flagged"
                elif commitment_out["label"] == "action" and verdict_label == "flagged":
                    # Only downgrade if the claim has SOME positive specificity signals
                    # A vague action with no specifics is still vague
                    if rule_out["positive_signal_count"] >= 1:
                        verdict_label = "needs_verification"
                    # else: stay flagged — describing a vague action doesn't make it specific

            # ---- Build explanation text ----
            explanation_parts: list[str] = []

            if model_says_vague:
                explanation_parts.append(
                    f"The ClimateBERT specificity model classified this claim "
                    f"as vague (confidence: {model_out['score']:.2f})."
                )
            elif model_out["label"] == "specific":
                explanation_parts.append(
                    f"The ClimateBERT specificity model classified this claim "
                    f"as specific (confidence: {model_out['score']:.2f})."
                )

            # Commitment model insight
            if commitment_out["label"] == "commitment" and commitment_out["score"] > 0.5:
                explanation_parts.append(
                    f"The claim describes a commitment rather than an "
                    f"implemented action (confidence: {commitment_out['score']:.2f})."
                )
            elif commitment_out["label"] == "action" and commitment_out["score"] > 0.5:
                explanation_parts.append(
                    f"The claim describes an implemented action rather than "
                    f"just a commitment (confidence: {commitment_out['score']:.2f})."
                )

            # Zero-shot insight for edge cases
            if zs_result["label"] != "unknown" and zs_result["score"] > 0.5:
                explanation_parts.append(
                    f"Zero-shot classification: \"{zs_result['label']}\" "
                    f"(confidence: {zs_result['score']:.2f})."
                )

            if rule_out["vague_terms_found"]:
                # Group by category for a cleaner message
                by_cat: dict[str, list[str]] = {}
                for term, cat in rule_out["vague_terms_found"]:
                    by_cat.setdefault(cat, []).append(f'"{term}"')
                parts = [
                    f"{cat.replace('_', ' ')}: {', '.join(terms)}"
                    for cat, terms in by_cat.items()
                ]
                explanation_parts.append(
                    "Vague language detected -- " + "; ".join(parts) + "."
                )

            if rule_out["positive_signals"]:
                explanation_parts.append(
                    "Positive specificity signals found: "
                    + ", ".join(rule_out["positive_signals"])
                    + "."
                )

            if rule_out["missing_info"]:
                explanation_parts.append(
                    "Missing information: "
                    + ", ".join(rule_out["missing_info"])
                    + "."
                )

            explanation = " ".join(explanation_parts) if explanation_parts else (
                "No significant vagueness detected."
            )

            # ---- LLM Judge integration ----
            llm_judge = kwargs.get("llm_judge")
            judgment_dict = None

            if llm_judge is not None:
                key_phrases = self._extract_key_phrases(claim.claim_text)
                matched_vague = [t for t, _ in rule_out["vague_terms_found"]]
                zeroshot_summary = (
                    f"{zs_result['label']} ({zs_result['score']:.2f})"
                    if zs_result["label"] != "unknown"
                    else "not run"
                )
                signals_for_judge = {
                    "page": claim.page,
                    "section_path": getattr(claim, "section_path", []),
                    "specificity_score": round(model_out["score"], 3),
                    "commitment_label": commitment_out["label"],
                    "zeroshot_result": zeroshot_summary,
                    "vague_terms": ", ".join(matched_vague) if matched_vague else "none",
                    "positive_signals": ", ".join(rule_out["positive_signals"]) if rule_out["positive_signals"] else "none",
                    "key_phrases": ", ".join(key_phrases) if key_phrases else "none",
                }
                jr = llm_judge.judge_claim(
                    module_name=self.name,
                    claim_text=claim.claim_text,
                    signals=signals_for_judge,
                )
                if jr is not None:
                    verdict_label = jr.verdict
                    jr.module_signals = signals_for_judge
                    judgment_dict = jr.to_dict()

            verdicts.append(
                Verdict.create(
                    module_name=self.name,
                    claim_id=claim.claim_id,
                    verdict=verdict_label,
                    explanation=explanation,
                    page=claim.page,
                    claim_text=claim.claim_text,
                    section_path=getattr(claim, "section_path", []),
                    missing_info=rule_out["missing_info"],
                    evidence=[
                        {
                            "model_label": model_out["label"],
                            "model_score": model_out["score"],
                            "commitment_label": commitment_out["label"],
                            "commitment_score": commitment_out["score"],
                            "zero_shot_label": zs_result["label"],
                            "zero_shot_score": zs_result["score"],
                            "vague_term_count": rule_out["vague_term_count"],
                            "positive_signal_count": rule_out["positive_signal_count"],
                            "vague_terms": [
                                {"term": t, "category": c}
                                for t, c in rule_out["vague_terms_found"]
                            ],
                        }
                    ],
                    judgment=judgment_dict,
                )
            )

        return verdicts
