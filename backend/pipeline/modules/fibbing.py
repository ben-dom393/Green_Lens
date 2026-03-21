"""Module 7 -- Fibbing Detection.

Detects outright false or fabricated environmental claims by checking for
internal contradictions, implausibly absolute statements, and factual errors
within the document.  Unlike Module 2 (No Proof) which flags the *absence*
of evidence, this module flags the *presence* of contradiction.
"""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    CLIMATE_SENTIMENT_MODEL,
    FACT_CHECK_MODEL,
    FACT_CHECK_THRESHOLD,
    NLI_MODEL,
    TABLE_FACT_MODEL,
    TOP_K_RERANK,
)
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)

# Pre-compiled regex patterns for superlative / absolute claim detection
_SUPERLATIVE_PATTERNS: list[re.Pattern] = [
    re.compile(
        r"\b(?:first\s+company\s+to|first\s+to)\b", re.IGNORECASE
    ),
    re.compile(
        r"\b(?:only\s+company|the\s+only)\b", re.IGNORECASE
    ),
    re.compile(r"\b100\s*%", re.IGNORECASE),
    re.compile(
        r"\b(?:zero\s+emissions|zero\s+waste)\b", re.IGNORECASE
    ),
    re.compile(r"\bcompletely\s+eliminated\b", re.IGNORECASE),
    re.compile(
        r"\b(?:entirely|never|always)\b", re.IGNORECASE
    ),
    re.compile(r"\ball\s+of\s+our\b", re.IGNORECASE),
    re.compile(r"\bevery\s+single\b", re.IGNORECASE),
    re.compile(
        r"\b(?:world'?s\s+first|industry\s+first)\b", re.IGNORECASE
    ),
    re.compile(r"\bno\s+other\s+company\b", re.IGNORECASE),
]


class FibbingModule(BaseModule):
    """Detect claims that are outright false or internally contradicted."""

    name = "fibbing"
    display_name = "Fibbing"

    def __init__(self) -> None:
        self._fact_check_model = None
        self._nli_model = None
        self._sentiment_model = None
        self._tapas_model = None
        self._tapas_tokenizer = None

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        """Lazily load all models needed for fibbing detection."""

        # ClimateBERT fact-checker (same as no_proof.py)
        if self._fact_check_model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._fact_check_model = hf_pipeline(
                    "text-classification",
                    model=FACT_CHECK_MODEL,
                    truncation=True,
                )
                logger.info("Loaded fact-check model: %s", FACT_CHECK_MODEL)
            except Exception:
                logger.warning(
                    "Could not load fact-check model '%s'. "
                    "Fact-checking will be skipped.",
                    FACT_CHECK_MODEL,
                )
                self._fact_check_model = None

        # DeBERTa NLI cross-encoder for contradiction detection
        if self._nli_model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._nli_model = CrossEncoder(NLI_MODEL)
                logger.info("Loaded NLI cross-encoder: %s", NLI_MODEL)
            except Exception as exc:
                logger.warning("NLI cross-encoder not available: %s", exc)
                self._nli_model = None

        # Climate sentiment model
        if self._sentiment_model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._sentiment_model = hf_pipeline(
                    "text-classification",
                    model=CLIMATE_SENTIMENT_MODEL,
                    truncation=True,
                )
                logger.info(
                    "Loaded climate sentiment model: %s",
                    CLIMATE_SENTIMENT_MODEL,
                )
            except Exception as exc:
                logger.warning(
                    "Climate sentiment model not available: %s", exc
                )
                self._sentiment_model = None

        # TAPAS table fact verification (optional)
        if self._tapas_model is None:
            try:
                from transformers import (
                    TapasForSequenceClassification,
                    TapasTokenizer,
                )

                self._tapas_tokenizer = TapasTokenizer.from_pretrained(
                    TABLE_FACT_MODEL
                )
                self._tapas_model = TapasForSequenceClassification.from_pretrained(
                    TABLE_FACT_MODEL
                )
                logger.info("Loaded TAPAS table-fact model: %s", TABLE_FACT_MODEL)
            except Exception as exc:
                logger.warning(
                    "TAPAS table-fact model not available (optional): %s", exc
                )
                self._tapas_model = None
                self._tapas_tokenizer = None

    # ------------------------------------------------------------------
    # Superlative / absolute claim detection
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_superlatives(text: str) -> list[str]:
        """Return all absolute / superlative phrases found in *text*."""
        matches: list[str] = []
        for pat in _SUPERLATIVE_PATTERNS:
            for m in pat.finditer(text):
                matches.append(m.group(0))
        return matches

    # ------------------------------------------------------------------
    # ClimateBERT fact-checker
    # ------------------------------------------------------------------
    def _check_facts(
        self, claim_text: str, evidence_texts: list[str]
    ) -> list[dict]:
        """Run ClimateBERT fact-checker on (claim, evidence) pairs.

        Returns a list of dicts with ``label`` (SUPPORTS / REFUTES /
        NOT_ENOUGH_INFO) and ``score``.
        """
        if self._fact_check_model is None or not evidence_texts:
            return []

        results: list[dict] = []
        try:
            pairs = [f"{claim_text} [SEP] {ev}" for ev in evidence_texts]
            raw_outputs = self._fact_check_model(pairs, batch_size=16)

            for out in raw_outputs:
                label_raw = out["label"].upper()
                score = float(out["score"])

                if "SUPPORT" in label_raw or label_raw in ("LABEL_0", "0"):
                    label = "SUPPORTS"
                elif "REFUTE" in label_raw or label_raw in ("LABEL_1", "1"):
                    label = "REFUTES"
                else:
                    label = "NOT_ENOUGH_INFO"

                results.append({"label": label, "score": score})
        except Exception:
            logger.exception("Fact-check model inference failed")
        return results

    # ------------------------------------------------------------------
    # DeBERTa NLI contradiction detection
    # ------------------------------------------------------------------
    def _check_contradictions(
        self, claim_text: str, evidence_texts: list[str]
    ) -> list[dict]:
        """Run DeBERTa NLI cross-encoder on (claim, evidence) pairs.

        Returns a list of dicts with ``label`` (contradiction / entailment /
        neutral) and ``scores`` dict mapping each label to its score.
        """
        if self._nli_model is None or not evidence_texts:
            return []

        results: list[dict] = []
        try:
            pairs = [(claim_text, ev) for ev in evidence_texts]
            scores = self._nli_model.predict(pairs)
            labels = ["contradiction", "entailment", "neutral"]
            for score_set in scores:
                best_idx = score_set.argmax()
                results.append(
                    {
                        "label": labels[best_idx],
                        "scores": {
                            l: round(float(s), 4)
                            for l, s in zip(labels, score_set)
                        },
                    }
                )
        except Exception:
            logger.exception("DeBERTa NLI inference failed")
        return results

    # ------------------------------------------------------------------
    # Sentiment consistency
    # ------------------------------------------------------------------
    def _check_sentiment_consistency(
        self, claim_text: str, evidence_texts: list[str]
    ) -> dict:
        """Compare sentiment of claim vs surrounding evidence.

        Returns a dict with ``claim_sentiment``, ``evidence_sentiments``,
        and ``consistent`` (bool).
        """
        result: dict = {
            "claim_sentiment": None,
            "evidence_sentiments": [],
            "consistent": True,
        }

        if self._sentiment_model is None:
            return result

        try:
            claim_out = self._sentiment_model(claim_text[:512])
            if isinstance(claim_out, list):
                claim_out = claim_out[0]
            claim_label = claim_out.get("label", "unknown")
            result["claim_sentiment"] = claim_label

            if not evidence_texts:
                return result

            ev_labels: list[str] = []
            for ev in evidence_texts:
                ev_out = self._sentiment_model(ev[:512])
                if isinstance(ev_out, list):
                    ev_out = ev_out[0]
                ev_labels.append(ev_out.get("label", "unknown"))

            result["evidence_sentiments"] = ev_labels

            # Consistency: claim and majority of evidence should agree
            if ev_labels:
                mismatches = sum(
                    1 for el in ev_labels if el != claim_label
                )
                result["consistent"] = mismatches < len(ev_labels) / 2
        except Exception:
            logger.exception("Sentiment consistency check failed")

        return result

    # ------------------------------------------------------------------
    # TAPAS table verification (optional)
    # ------------------------------------------------------------------
    def _verify_against_tables(
        self, claim_text: str, table_data: list[dict]
    ) -> list[dict]:
        """Verify a claim against tabular data using TAPAS.

        *table_data* should be a list of dicts, each with a ``"table"``
        key holding a ``list[list[str]]`` (rows including header) and
        optionally a ``"title"`` key.

        Returns a list of dicts with ``entailed`` (bool) and ``score``.
        """
        if (
            self._tapas_model is None
            or self._tapas_tokenizer is None
            or not table_data
        ):
            return []

        results: list[dict] = []
        try:
            import torch
            import pandas as pd

            for tbl_entry in table_data:
                raw_table = tbl_entry.get("table", [])
                if not raw_table or len(raw_table) < 2:
                    continue

                # Build a DataFrame from the table rows
                header = [str(h) for h in raw_table[0]]
                rows = [[str(c) for c in row] for row in raw_table[1:]]
                df = pd.DataFrame(rows, columns=header)

                inputs = self._tapas_tokenizer(
                    table=df,
                    queries=[claim_text],
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                with torch.no_grad():
                    outputs = self._tapas_model(**inputs)

                logits = outputs.logits
                predicted_class = logits.argmax(-1).item()
                score = float(
                    torch.softmax(logits, dim=-1)[0][predicted_class]
                )

                # TAPAS TabFact: 0 = REFUTED, 1 = ENTAILED
                results.append(
                    {
                        "entailed": predicted_class == 1,
                        "score": round(score, 4),
                        "table_title": tbl_entry.get("title", ""),
                    }
                )
        except Exception:
            logger.exception("TAPAS table verification failed")

        return results

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse each claim for false or fabricated statements.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``claim_text``, ``page``, ``section_path`` attributes).
        **kwargs:
            ``retriever`` -- optional RAG retriever with
            ``retrieve(query, top_k)`` returning list of dicts/strings.

            ``table_data`` -- optional list of table dicts for TAPAS
            verification.

            ``llm_judge`` -- optional LLM judge for final verdict.
        """
        self._load_resources()

        retriever = kwargs.get("retriever")
        table_data = kwargs.get("table_data") or []
        claim_list = list(claims)
        if not claim_list:
            return []

        verdicts: list[Verdict] = []

        for claim in claim_list:
            claim_text = claim.claim_text

            # --- 1. Detect superlatives / absolute claims ------------------
            superlatives = self._detect_superlatives(claim_text)

            # --- 2. RAG retrieve evidence passages -------------------------
            evidence_texts: list[str] = []
            evidence_metadata: list[dict] = []

            if retriever is not None:
                try:
                    results = retriever.retrieve(
                        claim_text, top_k=TOP_K_RERANK
                    )
                    for r in results:
                        if isinstance(r, dict):
                            evidence_texts.append(r.get("text", ""))
                            evidence_metadata.append(r)
                        elif isinstance(r, str):
                            evidence_texts.append(r)
                            evidence_metadata.append({"text": r})
                        else:
                            txt = getattr(r, "text", str(r))
                            evidence_texts.append(txt)
                            evidence_metadata.append({"text": txt})
                except Exception:
                    logger.exception(
                        "Retriever failed for claim '%s'",
                        claim_text[:80],
                    )

            # --- 3. ClimateBERT fact-checker --------------------------------
            fc_results = self._check_facts(claim_text, evidence_texts)

            refutes_count = sum(
                1
                for r in fc_results
                if r["label"] == "REFUTES"
                and r["score"] >= FACT_CHECK_THRESHOLD
            )
            supports_count = sum(
                1
                for r in fc_results
                if r["label"] == "SUPPORTS"
                and r["score"] >= FACT_CHECK_THRESHOLD
            )

            # --- 4. DeBERTa NLI contradiction detection --------------------
            nli_results = self._check_contradictions(
                claim_text, evidence_texts
            )

            contradiction_scores = [
                r["scores"].get("contradiction", 0.0)
                for r in nli_results
            ]
            max_contradiction = (
                max(contradiction_scores) if contradiction_scores else 0.0
            )

            # --- 5. Sentiment consistency ----------------------------------
            sentiment_result = self._check_sentiment_consistency(
                claim_text, evidence_texts
            )

            # --- 6. TAPAS table verification (optional) --------------------
            tapas_results = self._verify_against_tables(
                claim_text, table_data
            )
            tapas_refuted = any(
                not tr["entailed"] and tr["score"] >= 0.7
                for tr in tapas_results
            )

            # --- 7. LLM Judge integration ----------------------------------
            llm_judge = kwargs.get("llm_judge")
            judgment_dict = None

            signals_for_judge = {
                "page": claim.page,
                "section_path": getattr(claim, "section_path", []),
                "superlatives_found": superlatives,
                "fact_check_results": (
                    f"{supports_count} supporting, "
                    f"{refutes_count} refuting"
                ),
                "max_contradiction_score": round(max_contradiction, 4),
                "nli_results": (
                    str(nli_results[:3]) if nli_results else "N/A"
                ),
                "sentiment_consistent": sentiment_result["consistent"],
                "claim_sentiment": sentiment_result["claim_sentiment"],
                "tapas_results": (
                    str(tapas_results[:3]) if tapas_results else "N/A"
                ),
                "evidence_count": str(len(evidence_texts)),
            }

            if llm_judge is not None:
                try:
                    jr = llm_judge.judge_claim(
                        module_name=self.name,
                        claim_text=claim_text,
                        signals=signals_for_judge,
                        evidence=evidence_metadata[:5],
                    )
                    if jr is not None:
                        verdict_label = jr.verdict
                        jr.module_signals = signals_for_judge
                        judgment_dict = jr.to_dict()
                except Exception:
                    logger.exception(
                        "LLM judge failed for claim '%s'",
                        claim_text[:80],
                    )
                    llm_judge = None  # fall through to heuristic

            # --- 8. Heuristic fallback (no LLM) ---------------------------
            if judgment_dict is None:
                if max_contradiction > 0.7:
                    verdict_label = "flagged"
                elif (
                    refutes_count > 0
                    and fc_results
                    and refutes_count >= len(fc_results) / 2
                ):
                    verdict_label = "flagged"
                elif tapas_refuted:
                    verdict_label = "flagged"
                elif superlatives and not evidence_texts:
                    verdict_label = "needs_verification"
                elif superlatives and supports_count == 0:
                    verdict_label = "needs_verification"
                elif not sentiment_result["consistent"]:
                    verdict_label = "needs_verification"
                else:
                    verdict_label = "pass"

            # --- 9. Build explanation --------------------------------------
            explanation_parts: list[str] = []

            if superlatives:
                explanation_parts.append(
                    f"Absolute/superlative language detected: "
                    f"{', '.join(repr(s) for s in superlatives)}."
                )

            if not evidence_texts:
                explanation_parts.append(
                    "No evidence passages found in the document to "
                    "verify this claim."
                )
            else:
                explanation_parts.append(
                    f"Found {len(evidence_texts)} evidence passage(s)."
                )

            if fc_results:
                explanation_parts.append(
                    f"Fact-check: {supports_count} supporting, "
                    f"{refutes_count} refuting."
                )

            if nli_results:
                explanation_parts.append(
                    f"Max contradiction score: "
                    f"{max_contradiction:.2f}."
                )

            if not sentiment_result["consistent"]:
                explanation_parts.append(
                    "Sentiment mismatch between claim and evidence."
                )

            if tapas_results:
                refuted_tables = sum(
                    1 for tr in tapas_results if not tr["entailed"]
                )
                explanation_parts.append(
                    f"Table verification: {refuted_tables}/{len(tapas_results)} "
                    f"table(s) refute the claim."
                )

            explanation = " ".join(explanation_parts)

            # --- 10. Build evidence entries --------------------------------
            evidence_entries: list[dict] = []
            for i, ev_text in enumerate(evidence_texts):
                entry: dict = {"text": ev_text}
                if i < len(fc_results):
                    entry["fact_check"] = fc_results[i]
                if i < len(nli_results):
                    entry["nli"] = nli_results[i]
                if i < len(evidence_metadata):
                    for key, val in evidence_metadata[i].items():
                        if key != "text":
                            entry[key] = val
                evidence_entries.append(entry)

            verdicts.append(
                Verdict.create(
                    module_name=self.name,
                    claim_id=claim.claim_id,
                    verdict=verdict_label,
                    explanation=explanation,
                    page=claim.page,
                    claim_text=claim.claim_text,
                    section_path=getattr(claim, "section_path", []),
                    missing_info=(
                        [f"Unsupported superlative: {s}" for s in superlatives]
                        if superlatives and supports_count == 0
                        else []
                    ),
                    evidence=evidence_entries,
                    judgment=judgment_dict,
                )
            )

        return verdicts
