"""Module 6 -- Fake Labels Detection.

Detects when companies use self-created, fake, or misleading
eco-certifications.  Cross-references extracted certification names
against a legitimate labels knowledge base and uses semantic similarity,
zero-shot classification, and an optional LLM Judge for final verdicts.

This corresponds to the "sin of worshipping false labels" in the
TerraChoice greenwashing taxonomy.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR, HF_DEVICE, ZEROSHOT_MODEL
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Zero-shot candidate labels for label-type classification
# ---------------------------------------------------------------------------
_LABEL_TYPE_LABELS = [
    "third-party verified certification",
    "self-awarded label",
    "industry standard",
    "marketing language",
]

# ---------------------------------------------------------------------------
# Regex patterns for certification-like language
# ---------------------------------------------------------------------------
_CERT_PATTERNS = [
    # Phrases ending with certification keywords
    re.compile(
        r"(?i)\b[\w\s-]{2,50}?"
        r"(?:certified|certification|approved|accredited|verified"
        r"|compliant|standard|label|seal|mark)\b"
    ),
    # Trademark / registered symbols attached to terms
    re.compile(r"[\w\s-]{2,40}[™®℠]"),
    # "meets X standard", "in accordance with X", "aligned with X"
    re.compile(
        r"(?i)(?:meets|in accordance with|aligned with|certified by"
        r"|verified by|endorsed by|recognised by|recognized by)"
        r"\s+[\w\s'-]{2,60}"
    ),
]


class FakeLabelsModule(BaseModule):
    """Detect claims referencing self-created, fake, or misleading
    eco-certifications by cross-referencing against a legitimate
    labels knowledge base."""

    name = "fake_labels"
    display_name = "Fake Labels"

    def __init__(self) -> None:
        self._labels_kb: list[dict] | None = None
        self._zeroshot = None
        self._st_model = None
        self._label_embeddings = None
        self._label_names: list[str] | None = None
        self._nlp = None

    # ------------------------------------------------------------------
    # Resource loading (lazy)
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        """Lazily load all required resources."""
        self._load_labels_kb()
        self._load_zeroshot()
        self._load_sentence_transformer()
        self._load_spacy()

    def _load_labels_kb(self) -> None:
        """Load the legitimate labels knowledge base."""
        if self._labels_kb is not None:
            return
        kb_path = DATA_DIR / "legitimate_labels.json"
        try:
            with open(kb_path, encoding="utf-8") as fh:
                raw = json.load(fh)
            self._labels_kb = raw.get("labels", [])
            logger.info(
                "Loaded %d legitimate labels from %s",
                len(self._labels_kb),
                kb_path,
            )
        except Exception:
            logger.exception(
                "Failed to load legitimate labels KB from %s", kb_path
            )
            self._labels_kb = []

    def _load_zeroshot(self) -> None:
        """Lazily load the zero-shot classification model."""
        if self._zeroshot is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline

            self._zeroshot = hf_pipeline(
                "zero-shot-classification", model=ZEROSHOT_MODEL,
                device=HF_DEVICE,
            )
            logger.info("Loaded zero-shot model: %s", ZEROSHOT_MODEL)
        except Exception as exc:
            logger.warning("Zero-shot model not available: %s", exc)
            self._zeroshot = None

    def _load_sentence_transformer(self) -> None:
        """Lazily load sentence-transformers and pre-compute label embeddings."""
        if self._st_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("Loaded sentence-transformer for fuzzy matching")

            # Pre-compute embeddings for all label names + short_names
            all_names: list[str] = []
            for label in (self._labels_kb or []):
                all_names.append(label["name"])
                for sn in label.get("short_names", []):
                    if sn not in all_names:
                        all_names.append(sn)

            self._label_names = all_names
            self._label_embeddings = self._st_model.encode(
                all_names, convert_to_tensor=True
            )
            logger.info(
                "Pre-computed embeddings for %d label names", len(all_names)
            )
        except Exception as exc:
            logger.warning("Sentence-transformer not available: %s", exc)
            self._st_model = None
            self._label_names = []
            self._label_embeddings = None

    def _load_spacy(self) -> None:
        """Lazily load spaCy for ORG entity extraction."""
        if self._nlp is not None:
            return
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
            logger.info("Loaded spaCy model: en_core_web_sm")
        except Exception as exc:
            logger.warning("spaCy model not available: %s", exc)
            self._nlp = None

    # ------------------------------------------------------------------
    # Certification extraction
    # ------------------------------------------------------------------
    def _extract_certifications(self, text: str) -> list[str]:
        """Extract certification-like phrases from *text* using regex.

        Returns a deduplicated list of cleaned candidate names.
        """
        raw_matches: list[str] = []
        for pattern in _CERT_PATTERNS:
            for m in pattern.finditer(text):
                raw_matches.append(m.group(0))

        # Clean and deduplicate
        seen: set[str] = set()
        cleaned: list[str] = []
        for match in raw_matches:
            name = match.strip().strip("™®℠").strip()
            # Skip very short or clearly non-certification fragments
            if len(name) < 3:
                continue
            key = name.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(name)
        return cleaned

    # ------------------------------------------------------------------
    # Legitimate label lookup (exact match)
    # ------------------------------------------------------------------
    def _lookup_legitimate(self, cert_name: str) -> dict | None:
        """Check *cert_name* against all short_names in the KB (case-insensitive).

        Returns the matched label dict or ``None``.
        """
        lower = cert_name.lower().strip()
        for label in (self._labels_kb or []):
            for sn in label.get("short_names", []):
                if sn.lower() == lower:
                    return label
        return None

    # ------------------------------------------------------------------
    # Fuzzy matching via sentence-transformers
    # ------------------------------------------------------------------
    def _fuzzy_match(self, cert_name: str) -> tuple[str, float]:
        """Compute semantic similarity between *cert_name* and all known labels.

        Returns ``(best_match_name, similarity_score)``.
        """
        if (
            self._st_model is None
            or self._label_embeddings is None
            or not self._label_names
        ):
            return ("", 0.0)

        try:
            from sentence_transformers import util

            query_embedding = self._st_model.encode(
                cert_name, convert_to_tensor=True
            )
            scores = util.cos_sim(query_embedding, self._label_embeddings)[0]
            best_idx = int(scores.argmax())
            best_score = float(scores[best_idx])
            return (self._label_names[best_idx], best_score)
        except Exception:
            logger.debug(
                "Fuzzy match failed for '%s'", cert_name, exc_info=True
            )
            return ("", 0.0)

    # ------------------------------------------------------------------
    # Zero-shot label-type classification
    # ------------------------------------------------------------------
    def _classify_label_type(self, text: str) -> dict:
        """Zero-shot classify certification language.

        Returns a dict mapping each candidate label to its score,
        or an empty dict if the model is unavailable.
        """
        if self._zeroshot is None:
            return {}
        try:
            result = self._zeroshot(text, _LABEL_TYPE_LABELS, multi_label=False)
            return {
                label: round(score, 4)
                for label, score in zip(result["labels"], result["scores"])
            }
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # ORG entity extraction
    # ------------------------------------------------------------------
    def _extract_organizations(self, text: str) -> list[str]:
        """Use spaCy NER to extract ORG entities from *text*."""
        if self._nlp is None:
            return []
        try:
            doc = self._nlp(text)
            return list({ent.text for ent in doc.ents if ent.label_ == "ORG"})
        except Exception:
            return []

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse each claim for fake or misleading eco-labels.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``claim_text``, ``page``, ``section_path`` attributes).
        **kwargs:
            ``llm_judge`` -- optional :class:`LLMJudge` instance.
            ``regulatory_retriever`` -- optional regulatory RAG retriever.
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
            # Step (a): Extract certification-like phrases
            # ==============================================================
            cert_names = self._extract_certifications(claim_text)

            # ==============================================================
            # Step (b): No certification phrases → pass
            # ==============================================================
            if not cert_names:
                verdicts.append(
                    Verdict.create(
                        module_name=self.name,
                        claim_id=claim.claim_id,
                        verdict="pass",
                        explanation=(
                            "No certification or eco-label references "
                            "detected in this claim."
                        ),
                        page=claim.page,
                        claim_text=claim.claim_text,
                        section_path=getattr(claim, "section_path", []),
                    )
                )
                continue

            # ==============================================================
            # Step (c): For each cert name → KB lookup + fuzzy + zero-shot
            # ==============================================================
            cert_results: list[dict] = []
            any_not_found = False
            any_low_fuzzy = False
            all_found = True

            for cert_name in cert_names:
                kb_match = self._lookup_legitimate(cert_name)
                best_match, fuzzy_score = self._fuzzy_match(cert_name)
                zs_scores = self._classify_label_type(cert_name)

                entry = {
                    "cert_name": cert_name,
                    "kb_match": kb_match,
                    "fuzzy_best_match": best_match,
                    "fuzzy_score": round(fuzzy_score, 4),
                    "zeroshot_scores": zs_scores,
                }
                cert_results.append(entry)

                if kb_match is None:
                    all_found = False
                    any_not_found = True
                    if fuzzy_score < 0.8:
                        any_low_fuzzy = True

            # ==============================================================
            # Step (d): Extract ORG entities near certification mentions
            # ==============================================================
            organizations = self._extract_organizations(claim_text)

            # ==============================================================
            # Step (e): Regulatory RAG retrieval
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
            # Step (f): Build signals and call LLM Judge
            # ==============================================================
            signals_for_judge = {
                "page": claim.page,
                "section_path": getattr(claim, "section_path", []),
                "extracted_certifications": [
                    {
                        "cert_name": cr["cert_name"],
                        "found_in_kb": cr["kb_match"] is not None,
                        "kb_label_name": (
                            cr["kb_match"]["name"] if cr["kb_match"] else None
                        ),
                        "kb_issuing_body": (
                            cr["kb_match"]["issuing_body"]
                            if cr["kb_match"]
                            else None
                        ),
                        "fuzzy_best_match": cr["fuzzy_best_match"],
                        "fuzzy_score": cr["fuzzy_score"],
                        "zeroshot_scores": (
                            str(cr["zeroshot_scores"])
                            if cr["zeroshot_scores"]
                            else "N/A"
                        ),
                    }
                    for cr in cert_results
                ],
                "organizations_detected": organizations,
            }

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
            # Step (g): Fallback verdict (no LLM Judge)
            # ==============================================================
            if verdict_label is None:
                if all_found:
                    verdict_label = "pass"
                elif any_not_found and any_low_fuzzy:
                    verdict_label = "flagged"
                elif any_not_found:
                    # Not found in KB but fuzzy >= 0.8 → close match
                    verdict_label = "needs_verification"
                else:
                    verdict_label = "pass"

            # ==============================================================
            # Build explanation and evidence
            # ==============================================================
            found_names = [
                cr["cert_name"]
                for cr in cert_results
                if cr["kb_match"] is not None
            ]
            not_found_names = [
                cr["cert_name"]
                for cr in cert_results
                if cr["kb_match"] is None
            ]

            if verdict_label == "flagged":
                low_fuzzy = [
                    cr
                    for cr in cert_results
                    if cr["kb_match"] is None and cr["fuzzy_score"] < 0.8
                ]
                explanation = (
                    f"Potentially fake or misleading label(s) detected: "
                    f"{', '.join(cr['cert_name'] for cr in low_fuzzy)}. "
                    f"These were not found in the legitimate labels "
                    f"knowledge base and have low similarity to known "
                    f"certifications."
                )
                missing_info = [
                    "Third-party verification or issuing body for the label",
                    "Evidence the certification is recognised by an "
                    "independent standards body",
                ]
            elif verdict_label == "needs_verification":
                close_matches = [
                    cr
                    for cr in cert_results
                    if cr["kb_match"] is None and cr["fuzzy_score"] >= 0.8
                ]
                explanation = (
                    f"Label(s) not found in the legitimate KB but closely "
                    f"matching known certifications: "
                    f"{', '.join(cr['cert_name'] + ' (similar to ' + cr['fuzzy_best_match'] + ')' for cr in close_matches)}. "
                    f"Manual verification recommended."
                )
                missing_info = [
                    "Confirmation that the certification name is a variant "
                    "of a legitimate label",
                    "Issuing body and verification details",
                ]
            else:
                if found_names:
                    explanation = (
                        f"Certification(s) referenced in the claim are "
                        f"recognised legitimate labels: "
                        f"{', '.join(found_names)}."
                    )
                else:
                    explanation = (
                        "No certification or eco-label references "
                        "detected in this claim."
                    )
                missing_info = []

            evidence_list: list[dict] = []
            for cr in cert_results:
                ev: dict = {
                    "cert_name": cr["cert_name"],
                    "found_in_kb": cr["kb_match"] is not None,
                    "fuzzy_best_match": cr["fuzzy_best_match"],
                    "fuzzy_score": cr["fuzzy_score"],
                }
                if cr["kb_match"]:
                    ev["kb_label"] = cr["kb_match"]["name"]
                    ev["kb_issuing_body"] = cr["kb_match"]["issuing_body"]
                    ev["kb_category"] = cr["kb_match"]["category"]
                if cr["zeroshot_scores"]:
                    ev["zeroshot_scores"] = cr["zeroshot_scores"]
                evidence_list.append(ev)

            if organizations:
                evidence_list.append(
                    {"organizations_detected": organizations}
                )

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
