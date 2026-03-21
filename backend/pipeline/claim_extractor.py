"""Extract environmental claims from parsed ESG report text.

Pipeline
--------
1. Receive a list of ``DocumentElement`` paragraphs from the PDF parser.
2. Split each paragraph into sentences.
3. Classify sentences with ``climatebert/environmental-claims``.
4. For every sentence above the confidence threshold, extract structured
   information (named entities, quantities) with spaCy.
5. Return a list of ``Claim`` objects ready for downstream detection modules.
"""

from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Make sure the backend package root is importable regardless of cwd.
# ---------------------------------------------------------------------------
_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from config import (
    CLAIM_DETECTION_MODEL,
    CLAIM_DETECTION_THRESHOLD,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class DocumentElement:
    """A single element produced by the PDF parser.

    Parameters
    ----------
    element_id : str
        Unique identifier assigned by the parser.
    text : str
        The textual content of the element.
    page : int
        1-based page number where the element appears.
    element_type : str
        Structural type such as ``"NarrativeText"``, ``"Title"``,
        ``"ListItem"``, etc.
    section_path : list[str]
        Hierarchical section headers leading to this element,
        e.g. ``["Environment", "Climate Change", "GHG Emissions"]``.
    """

    element_id: str
    text: str
    page: int
    element_type: str
    section_path: list[str] = field(default_factory=list)


@dataclass
class Claim:
    """An environmental claim extracted from the report.

    Parameters
    ----------
    claim_id : str
        UUID for this claim.
    element_id : str
        Links back to the source ``DocumentElement``.
    page : int
        Page number of the source element.
    claim_text : str
        The individual sentence that was classified as a claim.
    full_context : str
        The full paragraph the sentence belongs to (provides context for
        downstream modules).
    section_path : list[str]
        Inherited from the source element.
    confidence : float
        ClimateBERT confidence score (0-1).
    entities : list[dict]
        Named entities extracted by spaCy.  Each dict has keys
        ``text``, ``label``, ``start_char``, ``end_char``.
    quantities : list[dict]
        Numeric quantities extracted from the sentence.  Each dict has
        keys ``value``, ``unit``, ``text``.
    """

    claim_id: str
    element_id: str
    page: int
    claim_text: str
    full_context: str
    section_path: list[str] = field(default_factory=list)
    confidence: float = 0.0
    entities: list[dict] = field(default_factory=list)
    quantities: list[dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Claim extractor
# ---------------------------------------------------------------------------

# Regex for splitting quantities – matches numbers (with optional decimals,
# commas, and percentage signs) followed by an optional unit token.
_QUANTITY_RE = re.compile(
    r"(?P<value>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|"
    r"(?:tonnes?|tons?|t|kg|mt|MtCO2e?|tCO2e?|GWh|MWh|kWh|MW|GW|"
    r"litres?|liters?|gallons?|m3|km|miles?|hectares?|acres?|"
    r"USD|EUR|GBP|\$|billion|million|thousand|bn|mn|m|k)(?:\b|(?=\s|$)))",
    re.IGNORECASE,
)


class ClaimExtractor:
    """Extracts environmental claims from parsed ESG report elements.

    Models are **lazy-loaded** on the first call to :meth:`extract_claims`
    so that importing the module is cheap and instantiation does not trigger
    large downloads.

    Example
    -------
    >>> extractor = ClaimExtractor()
    >>> claims = extractor.extract_claims(elements)
    """

    def __init__(self) -> None:
        self._classifier = None  # HuggingFace pipeline (loaded lazily)
        self._nlp = None  # spaCy Language model (loaded lazily)
        self._models_loaded = False

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Lazy-load ClimateBERT classifier and spaCy NER model.

        Called automatically on the first invocation of
        :meth:`extract_claims`.  Subsequent calls are no-ops.
        """
        if self._models_loaded:
            return

        # --- ClimateBERT (transformers) --------------------------------
        from transformers import pipeline as hf_pipeline

        print(f"[ClaimExtractor] Loading ClimateBERT model: {CLAIM_DETECTION_MODEL}")
        self._classifier = hf_pipeline(
            "text-classification",
            model=CLAIM_DETECTION_MODEL,
            top_k=None,  # return scores for all labels
            truncation=True,
        )
        print("[ClaimExtractor] ClimateBERT model loaded.")

        # --- spaCy ------------------------------------------------------
        try:
            import spacy

            self._nlp = spacy.load("en_core_web_sm")
            print("[ClaimExtractor] spaCy model 'en_core_web_sm' loaded.")
        except Exception as exc:
            print(
                f"[ClaimExtractor] WARNING: Could not load spaCy model "
                f"'en_core_web_sm' ({exc}). NER extraction will be skipped."
            )
            self._nlp = None

        self._models_loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_claims(
        self,
        elements: list[DocumentElement],
    ) -> list[Claim]:
        """Run the full extraction pipeline on a list of document elements.

        Parameters
        ----------
        elements:
            Parsed paragraphs / sections from the PDF parser.

        Returns
        -------
        list[Claim]
            Claims that exceed the detection confidence threshold.
        """
        self._load_models()

        claims: list[Claim] = []

        for element in elements:
            text = element.text.strip()
            if not text:
                continue

            # Skip section titles — they are headings, not claims
            if element.element_type == "Title":
                continue

            # Skip very short text (likely captions, labels, navigation)
            if len(text) < 30:
                continue

            # 1. Split paragraph into sentences
            sentences = self._split_sentences(text)
            if not sentences:
                continue

            # 2. Classify all sentences in one batch
            classified = self._classify_sentences(sentences)

            # 3. Build Claim objects for sentences above the threshold
            for sentence, confidence in classified:
                if confidence < CLAIM_DETECTION_THRESHOLD:
                    continue

                entities, quantities = self._extract_entities(sentence)

                claim = Claim(
                    claim_id=str(uuid.uuid4()),
                    element_id=element.element_id,
                    page=element.page,
                    claim_text=sentence,
                    full_context=text,
                    section_path=list(element.section_path),
                    confidence=round(confidence, 4),
                    entities=entities,
                    quantities=quantities,
                )
                claims.append(claim)

        return claims

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[str]:
        """Split a paragraph into sentences.

        Uses spaCy's built-in sentencizer when available for more
        accurate sentence boundary detection.  Falls back to a
        lightweight regex approach if spaCy is not loaded.

        Sentences shorter than 8 characters are dropped as they are
        unlikely to be meaningful claims.
        """
        if self._nlp is not None:
            doc = self._nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 8]
            return sentences
        # Fallback to simple splitting if spaCy not available
        raw = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
        return [s.strip() for s in raw if len(s.strip()) >= 8]

    def _classify_sentences(
        self,
        sentences: list[str],
    ) -> list[tuple[str, float]]:
        """Classify sentences with ClimateBERT in a single batch.

        Parameters
        ----------
        sentences:
            List of sentence strings.

        Returns
        -------
        list[tuple[str, float]]
            Each tuple is ``(sentence, confidence)`` where *confidence*
            is the probability assigned to the ``"yes"`` (environmental
            claim) label.
        """
        results: list[tuple[str, float]] = []

        if not self._classifier or not sentences:
            return results

        # transformers pipeline accepts a list for batch inference
        batch_output = self._classifier(sentences, batch_size=32)

        for sentence, label_scores in zip(sentences, batch_output):
            # label_scores is a list of dicts: [{"label": "yes", "score": ...}, ...]
            claim_score = 0.0
            for entry in label_scores:
                if entry["label"].lower() == "yes":
                    claim_score = entry["score"]
                    break
            results.append((sentence, claim_score))

        return results

    def _extract_entities(
        self,
        text: str,
    ) -> tuple[list[dict], list[dict]]:
        """Extract named entities and numeric quantities from *text*.

        Parameters
        ----------
        text:
            A single sentence (the claim text).

        Returns
        -------
        entities:
            List of dicts with keys ``text``, ``label``, ``start_char``,
            ``end_char`` from spaCy NER.
        quantities:
            List of dicts with keys ``value`` (str), ``unit`` (str),
            ``text`` (str – the matched span) extracted via regex (and
            optionally supplemented by spaCy QUANTITY entities).
        """
        entities: list[dict] = []
        quantities: list[dict] = []

        # --- Regex-based quantity extraction (always available) ---------
        for m in _QUANTITY_RE.finditer(text):
            quantities.append(
                {
                    "value": m.group("value").replace(",", ""),
                    "unit": m.group("unit"),
                    "text": m.group(0),
                }
            )

        # --- spaCy NER (if model is available) --------------------------
        if self._nlp is not None:
            doc = self._nlp(text)
            for ent in doc.ents:
                entities.append(
                    {
                        "text": ent.text,
                        "label": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                    }
                )

                # Supplement quantities from spaCy's QUANTITY / PERCENT labels
                if ent.label_ in ("QUANTITY", "PERCENT", "MONEY", "CARDINAL"):
                    # Only add if not already captured by the regex
                    already_captured = any(
                        q["text"] in ent.text or ent.text in q["text"]
                        for q in quantities
                    )
                    if not already_captured:
                        # Try to split a rough value/unit from the entity text
                        num_match = re.search(r"-?\d[\d,.]*", ent.text)
                        value_str = (
                            num_match.group(0).replace(",", "")
                            if num_match
                            else ent.text
                        )
                        unit_str = ent.text[num_match.end() :].strip() if num_match else ""
                        quantities.append(
                            {
                                "value": value_str,
                                "unit": unit_str,
                                "text": ent.text,
                            }
                        )

        return entities, quantities
