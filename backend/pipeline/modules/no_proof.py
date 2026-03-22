"""Module 2 -- No Proof Detection.

Detects environmental claims that lack supporting evidence by combining
in-document RAG retrieval with a ClimateBERT fact-checking classifier and
structured proof checklists.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import DATA_DIR, FACT_CHECK_MODEL, FACT_CHECK_THRESHOLD, HF_DEVICE, TOP_K_RERANK
from pipeline.modules.base import BaseModule, Verdict

logger = logging.getLogger(__name__)


class NoProofModule(BaseModule):
    """Detect claims that lack adequate supporting evidence."""

    name = "no_proof"
    display_name = "No Proof"

    def __init__(self) -> None:
        self._model = None
        self._checklists: list[dict] | None = None
        self._st_model = None
        self._type_embeddings = None
        self._nli_model = None
        self._tcfd_model = None

    # ------------------------------------------------------------------
    # Resource loading
    # ------------------------------------------------------------------
    def _load_resources(self) -> None:
        """Lazily load the fact-checking model and proof checklists."""
        if self._checklists is None:
            checklist_path = DATA_DIR / "proof_checklists.json"
            try:
                with open(checklist_path, encoding="utf-8") as fh:
                    raw = json.load(fh)
                # The file stores claim types as a list under "claim_types"
                self._checklists = raw.get("claim_types", [])
                logger.info(
                    "Loaded %d proof checklists from %s",
                    len(self._checklists),
                    checklist_path,
                )
            except Exception:
                logger.exception(
                    "Failed to load proof checklists from %s", checklist_path
                )
                self._checklists = []

        if self._model is None:
            try:
                from transformers import pipeline as hf_pipeline

                self._model = hf_pipeline(
                    "text-classification",
                    model=FACT_CHECK_MODEL,
                    truncation=True,
                    device=HF_DEVICE,
                )
                logger.info("Loaded fact-check model: %s", FACT_CHECK_MODEL)
            except Exception:
                logger.warning(
                    "Could not load fact-check model '%s'. "
                    "Falling back to checklist-only analysis.",
                    FACT_CHECK_MODEL,
                )
                self._model = None

        # Load sentence-transformers for semantic claim type matching
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
                # Pre-compute embeddings for each claim type description
                if self._checklists:
                    descriptions = [
                        t["description"] for t in self._checklists
                    ]
                    self._type_embeddings = self._st_model.encode(descriptions)
                logger.info(
                    "Loaded sentence-transformers model for semantic matching"
                )
            except Exception as e:
                logger.warning("Sentence-transformers not available: %s", e)
                self._st_model = None
                self._type_embeddings = None

        if self._nli_model is None:
            try:
                from sentence_transformers import CrossEncoder
                from config import NLI_MODEL
                self._nli_model = CrossEncoder(NLI_MODEL)
                logger.info("Loaded NLI cross-encoder: %s", NLI_MODEL)
            except Exception as e:
                logger.warning("NLI cross-encoder not available: %s", e)
                self._nli_model = None

        if self._tcfd_model is None:
            try:
                from transformers import pipeline as hf_pipeline
                from config import TCFD_MODEL
                self._tcfd_model = hf_pipeline("text-classification", model=TCFD_MODEL, truncation=True, device=HF_DEVICE)
                logger.info("Loaded TCFD model: %s", TCFD_MODEL)
            except Exception as e:
                logger.warning("TCFD model not available: %s", e)
                self._tcfd_model = None

    # ------------------------------------------------------------------
    # Claim type matching
    # ------------------------------------------------------------------
    def _match_claim_type(self, claim_text: str) -> dict | None:
        """Match *claim_text* to the best-fitting proof checklist entry.

        Uses semantic similarity via sentence-transformers when available,
        falling back to simple keyword overlap when the model is not loaded.
        """
        # --- Semantic similarity matching (preferred) ---
        if self._st_model is not None and self._type_embeddings is not None:
            try:
                from sentence_transformers import util

                claim_emb = self._st_model.encode(claim_text)
                scores = util.cos_sim(claim_emb, self._type_embeddings)[0]
                best_idx = scores.argmax().item()
                best_score = scores[best_idx].item()
                if best_score > 0.3:  # minimum similarity threshold
                    return self._checklists[best_idx]
            except Exception:
                logger.debug(
                    "Semantic claim type matching failed, falling back to keywords"
                )

        # --- Fallback: keyword matching ---
        text_lower = claim_text.lower()
        best_match: dict | None = None
        best_score = 0

        for ct in self._checklists or []:
            keywords = ct.get("keywords", [])
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > best_score:
                best_score = score
                best_match = ct

        return best_match

    # ------------------------------------------------------------------
    # Fact-check model
    # ------------------------------------------------------------------
    def _check_evidence(
        self, claim_text: str, evidence_texts: list[str]
    ) -> list[dict]:
        """Run the ClimateBERT fact-checking model on (claim, evidence) pairs.

        Returns a list of dicts with keys ``label`` (``SUPPORTS``,
        ``REFUTES``, or ``NOT_ENOUGH_INFO``) and ``score``.  If the
        model is not loaded, returns an empty list.
        """
        if self._model is None or not evidence_texts:
            return []

        results: list[dict] = []
        try:
            # The fact-check model expects pairs: claim [SEP] evidence
            pairs = [f"{claim_text} [SEP] {ev}" for ev in evidence_texts]
            raw_outputs = self._model(pairs, batch_size=16)

            for out in raw_outputs:
                label_raw = out["label"].upper()
                score = float(out["score"])

                # Normalise label names
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
    # NLI cross-encoder check
    # ------------------------------------------------------------------
    def _check_nli(self, claim_text: str, evidence_texts: list[str]) -> list[dict]:
        """Run DeBERTa NLI cross-encoder on (claim, evidence) pairs."""
        if self._nli_model is None or not evidence_texts:
            return []
        results = []
        try:
            pairs = [(claim_text, ev) for ev in evidence_texts]
            scores = self._nli_model.predict(pairs)
            labels = ["contradiction", "entailment", "neutral"]
            for score_set in scores:
                best_idx = score_set.argmax()
                results.append({
                    "label": labels[best_idx],
                    "scores": {l: round(float(s), 4) for l, s in zip(labels, score_set)},
                })
        except Exception:
            logger.exception("DeBERTa NLI inference failed")
        return results

    # ------------------------------------------------------------------
    # TCFD classification
    # ------------------------------------------------------------------
    def _classify_tcfd(self, text: str) -> str:
        """Classify text using the TCFD model."""
        if self._tcfd_model is None:
            return "unknown"
        try:
            result = self._tcfd_model(text[:512])
            if isinstance(result, list):
                result = result[0]
            return result.get("label", "unknown")
        except Exception:
            return "unknown"

    # ------------------------------------------------------------------
    # Proof-checklist verification
    # ------------------------------------------------------------------
    def _check_proof_checklist(
        self,
        claim_text: str,
        evidence_texts: list[str],
        checklist: dict,
    ) -> list[str]:
        """Check which required evidence fields are missing.

        Uses semantic similarity when sentence-transformers is available:
        embeds each required_evidence description and the combined
        claim+evidence text, then computes cosine similarity. If
        similarity > 0.4, considers that evidence field as "addressed".

        Falls back to keyword matching when the model is not loaded.

        Returns a list of field descriptions that are **missing**.
        """
        combined = claim_text + " " + " ".join(evidence_texts)
        missing: list[str] = []

        # --- Semantic similarity approach (preferred) ---
        if self._st_model is not None:
            try:
                from sentence_transformers import util

                combined_emb = self._st_model.encode(combined)
                for req in checklist.get("required_evidence", []):
                    req_desc = req.get("description", req["field"])
                    req_emb = self._st_model.encode(req_desc)
                    sim = util.cos_sim(combined_emb, req_emb).item()
                    if sim <= 0.4:
                        missing.append(req_desc)
                return missing
            except Exception:
                logger.debug(
                    "Semantic checklist matching failed, falling back to keywords"
                )

        # --- Fallback: keyword matching ---
        combined_lower = combined.lower()
        for req in checklist.get("required_evidence", []):
            match_keywords = req.get("match_keywords", [])
            if not match_keywords:
                # Fallback: split field name on underscores/spaces
                match_keywords = req["field"].replace("_", " ").split()

            found = any(kw.lower() in combined_lower for kw in match_keywords)
            if not found:
                missing.append(req.get("description", req["field"]))

        return missing

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------
    def analyze(self, claims, **kwargs) -> list[Verdict]:
        """Analyse each claim for missing proof and return verdicts.

        Parameters
        ----------
        claims:
            Iterable of ``Claim`` objects (must have at least ``claim_id``,
            ``text``, ``page``, ``section_path`` attributes).
        **kwargs:
            ``retriever`` -- optional RAG retriever object with a
            ``retrieve(query, top_k)`` method returning a list of dicts
            with at least a ``"text"`` key.

            ``regulatory_retriever`` -- optional regulatory RAG retriever
            that returns dicts with ``text``, ``source``, ``page``, and
            ``document_name`` keys.  When present, regulatory evidence is
            appended with ``evidence_type: "regulatory"``.
        """
        self._load_resources()

        retriever = kwargs.get("retriever")
        regulatory_retriever = kwargs.get("regulatory_retriever")
        claim_list = list(claims)
        if not claim_list:
            return []

        verdicts: list[Verdict] = []

        for claim in claim_list:
            claim_text = claim.claim_text

            # --- 1. Match claim type ------------------------------------
            matched_type = self._match_claim_type(claim_text)

            # --- 2. Retrieve evidence via RAG ---------------------------
            evidence_texts: list[str] = []
            evidence_metadata: list[dict] = []

            if retriever is not None:
                try:
                    top_k = TOP_K_RERANK
                    results = retriever.retrieve(claim_text, top_k=top_k)
                    for r in results:
                        if isinstance(r, dict):
                            evidence_texts.append(r.get("text", ""))
                            evidence_metadata.append(r)
                        elif isinstance(r, str):
                            evidence_texts.append(r)
                            evidence_metadata.append({"text": r})
                        else:
                            # Handle objects with a .text attribute
                            txt = getattr(r, "text", str(r))
                            evidence_texts.append(txt)
                            evidence_metadata.append({"text": txt})
                except Exception:
                    logger.exception(
                        "Retriever failed for claim '%s'",
                        claim_text[:80],
                    )

            # --- 2b. Retrieve regulatory evidence ----------------------
            regulatory_evidence: list[dict] = []

            if regulatory_retriever is not None:
                try:
                    reg_results = regulatory_retriever.retrieve(
                        claim_text, top_k=3
                    )
                    for r in reg_results:
                        if isinstance(r, dict):
                            reg_entry = {
                                "text": r.get("text", ""),
                                "evidence_type": "regulatory",
                                "source": r.get("source", ""),
                                "page": r.get("page", 0),
                                "document_name": r.get("document_name", ""),
                                "score": r.get("score", 0.0),
                            }
                            evidence_texts.append(reg_entry["text"])
                            evidence_metadata.append(reg_entry)
                            regulatory_evidence.append(reg_entry)
                except Exception:
                    logger.exception(
                        "Regulatory retriever failed for claim '%s'",
                        claim_text[:80],
                    )

            # --- 3. Fact-check model ------------------------------------
            fc_results = self._check_evidence(claim_text, evidence_texts)

            # Summarise fact-check outcomes
            supports_count = sum(
                1 for r in fc_results
                if r["label"] == "SUPPORTS" and r["score"] >= FACT_CHECK_THRESHOLD
            )
            refutes_count = sum(
                1 for r in fc_results
                if r["label"] == "REFUTES" and r["score"] >= FACT_CHECK_THRESHOLD
            )
            nei_count = sum(
                1 for r in fc_results
                if r["label"] == "NOT_ENOUGH_INFO"
            )

            # --- 4. Proof checklist ------------------------------------
            missing_fields: list[str] = []
            if matched_type is not None:
                missing_fields = self._check_proof_checklist(
                    claim_text, evidence_texts, matched_type
                )

            # --- 5. Verdict logic --------------------------------------
            has_evidence = len(evidence_texts) > 0
            has_support = supports_count > 0
            has_refutation = refutes_count > 0
            checklist_coverage = (
                1.0
                - len(missing_fields)
                / max(
                    len(matched_type.get("required_evidence", [1])),
                    1,
                )
                if matched_type
                else 0.0
            )

            if not has_evidence:
                # No evidence found at all in the document
                verdict_label = "flagged"
            elif has_refutation and not has_support:
                # Evidence actively contradicts the claim
                verdict_label = "flagged"
            elif has_support and not has_refutation and checklist_coverage >= 0.5:
                # Evidence supports the claim and checklist is mostly covered
                verdict_label = "pass"
            elif has_support and checklist_coverage < 0.5:
                # Evidence supports but checklist has gaps
                verdict_label = "needs_verification"
            elif has_evidence and checklist_coverage >= 0.4:
                # Evidence exists and checklist is partially covered —
                # inconclusive fact-check doesn't mean no proof
                verdict_label = "needs_verification"
            elif has_evidence and nei_count > 0:
                # Evidence exists but fact-check is inconclusive and
                # checklist coverage is low
                verdict_label = "needs_verification"
            else:
                # No evidence or very low checklist coverage
                verdict_label = (
                    "flagged" if checklist_coverage < 0.2 else "needs_verification"
                )

            # --- 6. Build explanation ----------------------------------
            explanation_parts: list[str] = []

            if not has_evidence:
                explanation_parts.append(
                    "No supporting evidence was found in the document for "
                    "this claim."
                )
            else:
                explanation_parts.append(
                    f"Found {len(evidence_texts)} evidence passage(s) in "
                    f"the document."
                )

            if fc_results:
                explanation_parts.append(
                    f"Fact-check results: {supports_count} supporting, "
                    f"{refutes_count} refuting, {nei_count} inconclusive."
                )

            if matched_type is not None:
                type_label = matched_type.get("type", "unknown")
                total_required = len(
                    matched_type.get("required_evidence", [])
                )
                found_count = total_required - len(missing_fields)
                explanation_parts.append(
                    f"Claim type '{type_label}': {found_count}/{total_required} "
                    f"required evidence fields addressed."
                )
            else:
                explanation_parts.append(
                    "Could not match this claim to a known claim type for "
                    "checklist verification."
                )

            if missing_fields:
                explanation_parts.append(
                    "Missing evidence: " + "; ".join(missing_fields) + "."
                )

            # Regulatory citations
            if regulatory_evidence:
                reg_citations: list[str] = []
                for reg in regulatory_evidence:
                    doc_name = reg.get("document_name", reg.get("source", "unknown"))
                    page = reg.get("page", "?")
                    # Truncate the regulatory text for the citation
                    reg_text = reg["text"][:150].rstrip()
                    if len(reg["text"]) > 150:
                        reg_text += "..."
                    reg_citations.append(
                        f"According to [{doc_name}, p.{page}]: \"{reg_text}\""
                    )
                explanation_parts.append(
                    "Regulatory references: " + " | ".join(reg_citations)
                )

            explanation = " ".join(explanation_parts)

            # ---- LLM Judge integration ----
            llm_judge = kwargs.get("llm_judge")
            judgment_dict = None

            if llm_judge is not None:
                nli_results = self._check_nli(claim_text, evidence_texts)
                tcfd_cat = self._classify_tcfd(claim_text)
                signals_for_judge = {
                    "page": claim.page,
                    "section_path": getattr(claim, "section_path", []),
                    "fact_check_results": f"{supports_count} supporting, {refutes_count} refuting, {nei_count} inconclusive",
                    "nli_results": str(nli_results[:3]) if nli_results else "N/A",
                    "checklist_coverage": f"{checklist_coverage:.0%}",
                    "missing_fields": ", ".join(missing_fields) if missing_fields else "none",
                    "tcfd_category": tcfd_cat,
                    "evidence_count": str(len(evidence_texts)),
                }
                jr = llm_judge.judge_claim(
                    module_name=self.name,
                    claim_text=claim_text,
                    signals=signals_for_judge,
                    evidence=evidence_metadata[:5],
                    kb_context={"regulatory": regulatory_evidence},
                )
                if jr is not None:
                    verdict_label = jr.verdict
                    jr.module_signals = signals_for_judge
                    judgment_dict = jr.to_dict()

            # Build evidence list for the verdict
            evidence_entries: list[dict] = []
            for i, ev_text in enumerate(evidence_texts):
                entry: dict = {"text": ev_text}
                if i < len(fc_results):
                    entry["fact_check"] = fc_results[i]
                if i < len(evidence_metadata):
                    # Carry over any extra metadata (page, chunk_id, etc.)
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
                    missing_info=missing_fields,
                    evidence=evidence_entries,
                    judgment=judgment_dict,
                )
            )

        return verdicts
