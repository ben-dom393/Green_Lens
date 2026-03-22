"""Extract environmental claims from parsed ESG report text.

Pipeline
--------
1. Receive a list of ``DocumentElement`` paragraphs from the PDF parser.
2. Split each paragraph into sentences.
3. Classify sentences with ``climatebert/environmental-claims``.
4. For every sentence above the confidence threshold, extract structured
   information (named entities, quantities) with spaCy.
5. Return a list of ``Claim`` objects ready for downstream detection modules.
6. Optionally group co-located claims into ``ClaimGroup`` objects that
   preserve sentence-level highlights while providing paragraph-level context.
"""

from __future__ import annotations

import re
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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
    element_role : str
        Advisory role tag from the PDF parser. One of
        ``"claim_candidate"``, ``"evidence"``, ``"context"``, ``"skip"``.
    """

    element_id: str
    text: str
    page: int
    element_type: str
    section_path: list[str] = field(default_factory=list)
    element_role: str = "claim_candidate"


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
    element_role : str
        Advisory role tag inherited from the source element.
    sentence_offset : int
        Character offset of this sentence within the source paragraph.
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
    element_role: str = "claim_candidate"
    sentence_offset: int = 0
    artifact_signals: list[str] = field(default_factory=list)


@dataclass
class ClaimGroup:
    """A group of co-located claims from the same paragraph.

    Aggregates sentence-level claims while preserving per-sentence data
    for highlighting. Duck-types as ``Claim`` so downstream modules
    (M1-M7, LLM Judge) work without changes.
    """

    group_id: str
    claims: list[Claim]
    claim_text: str                    # Sentences joined with space
    representative_sentence: str       # Highest-confidence sentence
    representative_confidence: float
    page: int
    section_path: list[str] = field(default_factory=list)
    element_id: str = ""
    element_role: str = "claim_candidate"
    entities: list[dict] = field(default_factory=list)
    quantities: list[dict] = field(default_factory=list)
    full_context: str = ""
    confidence: float = 0.0            # = representative_confidence
    claim_id: str = ""                 # = group_id (duck-typing)
    sentence_offset: int = 0           # From first constituent
    artifact_signals: list[str] = field(default_factory=list)

    @classmethod
    def from_claims(cls, claims: list["Claim"]) -> "ClaimGroup":
        """Create a ClaimGroup from one or more Claims sharing element_id."""
        assert len(claims) >= 1

        # Sort by sentence offset (character position in original paragraph)
        sorted_claims = sorted(claims, key=lambda c: c.sentence_offset)

        # Representative = highest ClimateBERT confidence; tie-break: earliest
        rep = max(sorted_claims, key=lambda c: (c.confidence, -c.sentence_offset))

        # Join claim texts with space
        joined_text = " ".join(c.claim_text.strip() for c in sorted_claims)

        # Merge entities — deduplicate by (text, label)
        seen_ents: set[tuple[str, str]] = set()
        merged_entities: list[dict] = []
        for c in sorted_claims:
            for e in c.entities:
                key = (e["text"], e["label"])
                if key not in seen_ents:
                    seen_ents.add(key)
                    merged_entities.append(e)

        # Merge quantities — deduplicate by (value, unit)
        seen_qty: set[tuple[str, str]] = set()
        merged_quantities: list[dict] = []
        for c in sorted_claims:
            for q in c.quantities:
                key = (q["value"], q["unit"])
                if key not in seen_qty:
                    seen_qty.add(key)
                    merged_quantities.append(q)

        # Merge artifact_signals — union of all constituents
        merged_artifacts: list[str] = []
        seen_artifacts: set[str] = set()
        for c in sorted_claims:
            for sig in getattr(c, "artifact_signals", []):
                if sig not in seen_artifacts:
                    seen_artifacts.add(sig)
                    merged_artifacts.append(sig)

        gid = str(uuid.uuid4())
        first = sorted_claims[0]

        return cls(
            group_id=gid,
            claims=sorted_claims,
            claim_text=joined_text,
            representative_sentence=rep.claim_text,
            representative_confidence=rep.confidence,
            page=first.page,
            section_path=list(first.section_path),
            element_id=first.element_id,
            element_role=first.element_role,
            entities=merged_entities,
            quantities=merged_quantities,
            full_context=first.full_context,
            confidence=rep.confidence,
            claim_id=gid,
            sentence_offset=first.sentence_offset,
            artifact_signals=merged_artifacts,
        )


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


# ---------------------------------------------------------------------------
# Artifact patterns for soft tagging (never delete text, just flag)
# ---------------------------------------------------------------------------
ARTIFACT_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("pipe_nav", re.compile(
        r"(?:^|\n)\s*[\w\s]+(?:\s*\|\s*[\w\s]+){2,}", re.MULTILINE
    )),
    ("embedded_page_num", re.compile(
        r"(?<=[a-z.,;])\s+\d{1,3}\s+(?=[A-Z])"
    )),
    ("nav_keywords", re.compile(
        r"(?i)\b(?:click here|learn more|see page|back to top|table of contents"
        r"|download|visit|homepage|skip to)\b"
    )),
    ("toc_entry", re.compile(
        r"(?m)^[A-Z][\w\s]{5,50}\s{2,}\d{1,3}\s*$"
    )),
    ("concatenated_headers", re.compile(
        r"(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s*){3,}"
    )),
]


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
    # Artifact detection (soft tagging — never deletes text)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_artifacts(text: str) -> list[str]:
        """Detect PDF parsing artifacts in paragraph text.

        Runs on the FULL paragraph (not individual sentences) to catch
        cross-sentence artifacts.  Returns a list of matched pattern
        names (e.g. ``["pipe_nav", "embedded_page_num"]``).

        These are advisory signals — they never cause text deletion.
        """
        found: list[str] = []
        for name, pattern in ARTIFACT_PATTERNS:
            if pattern.search(text):
                found.append(name)
        return found

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_claims(
        self,
        elements: list[DocumentElement],
        return_groups: bool = False,
    ) -> Union[list[Claim], list[ClaimGroup]]:
        """Run the full extraction pipeline on a list of document elements.

        Parameters
        ----------
        elements:
            Parsed paragraphs / sections from the PDF parser.
        return_groups:
            If *True*, aggregate co-located sentence claims into
            ``ClaimGroup`` objects grouped by source paragraph.
            If *False* (default), return individual ``Claim`` objects
            (original behaviour).

        Returns
        -------
        list[Claim] | list[ClaimGroup]
            Claims (or groups) that exceed the detection confidence
            threshold.
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

            # 0. Detect PDF artifacts (soft tagging on full paragraph)
            artifacts = self._detect_artifacts(text)

            # 1. Split paragraph into sentences (with character offsets)
            sentence_pairs = self._split_sentences(text)
            if not sentence_pairs:
                continue

            sentences = [s for s, _ in sentence_pairs]
            offsets = [o for _, o in sentence_pairs]

            # 2. Classify all sentences in one batch
            classified = self._classify_sentences(sentences)

            # 3. Build Claim objects for sentences above the threshold
            for (sentence, confidence), offset in zip(classified, offsets):
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
                    element_role=getattr(element, "element_role", "claim_candidate"),
                    sentence_offset=offset,
                    artifact_signals=list(artifacts),
                )
                claims.append(claim)

        if return_groups:
            return self._group_claims(claims)
        return claims

    # ------------------------------------------------------------------
    # Grouping
    # ------------------------------------------------------------------

    def _group_claims(self, claims: list[Claim]) -> list[ClaimGroup]:
        """Aggregate sentence-level claims by source paragraph.

        Claims sharing the same ``element_id`` (i.e. originating from
        the same ``DocumentElement`` / PDF paragraph) are merged into a
        single ``ClaimGroup``.
        """
        by_element: dict[str, list[Claim]] = defaultdict(list)
        for claim in claims:
            by_element[claim.element_id].append(claim)

        groups: list[ClaimGroup] = []
        for _eid, element_claims in by_element.items():
            groups.append(ClaimGroup.from_claims(element_claims))

        # Sort groups by page then sentence_offset for deterministic order
        groups.sort(key=lambda g: (g.page, g.sentence_offset))
        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _split_sentences(self, text: str) -> list[tuple[str, int]]:
        """Split a paragraph into sentences with character offsets.

        Uses spaCy's built-in sentencizer when available for more
        accurate sentence boundary detection.  Falls back to a
        lightweight regex approach if spaCy is not loaded.

        Sentences shorter than 8 characters are dropped as they are
        unlikely to be meaningful claims.

        Returns
        -------
        list[tuple[str, int]]
            Each tuple is ``(sentence_text, char_offset)`` where
            *char_offset* is the start position in the original *text*.
        """
        if self._nlp is not None:
            doc = self._nlp(text)
            return [
                (sent.text.strip(), sent.start_char)
                for sent in doc.sents
                if len(sent.text.strip()) >= 8
            ]
        # Fallback: regex splitting with offset tracking
        results: list[tuple[str, int]] = []
        boundaries = [0] + [m.end() for m in re.finditer(r"(?<=[.!?])\s+(?=[A-Z])", text)]
        for i, start in enumerate(boundaries):
            end = boundaries[i + 1] if i + 1 < len(boundaries) else len(text)
            sentence = text[start:end].strip()
            if len(sentence) >= 8:
                results.append((sentence, start))
        return results

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
