"""General-purpose PDF parser for ESG reports.

Extracts text (with section hierarchy) and tables from any ESG report PDF
using PyMuPDF (fitz).  Not hard-coded for any specific company layout.

Output is a flat list of ``DocumentElement`` dicts::

    {
        "element_id":   "p{page}_{index}",
        "text":         str,
        "page":         int,
        "element_type": "Title" | "NarrativeText" | "ListItem" | "Table",
        "section_path": [str, ...],
        "table_data":   None | [[col, ...], ...]   # non-None only for Tables
        "element_role": "claim_candidate" | "evidence" | "context" | "skip"
    }

``element_role`` is a lightweight pre-classification that helps downstream
modules decide how to use each element **without** filtering anything out.
All elements are preserved — the role is advisory:

* ``claim_candidate`` — narrative text likely containing verifiable claims.
* ``evidence`` — data tables, third-party statements, or quantitative
  disclosures that can support or refute claims.
* ``context`` — methodology descriptions, legal disclaimers,
  framework references, or other text that provides background
  but is not itself a claim.
* ``skip`` — page numbers, table-of-contents entries, or other
  non-content elements.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Optional

import fitz  # PyMuPDF


# ── tiny helpers ────────────────────────────────────────────────────────

_BULLET_RE = re.compile(
    r"^\s*(?:[•●○◦▪▸►\-–—]|\d{1,3}[.)]\s|[a-zA-Z][.)]\s|[\uf0b7\uf0a7])",
)

_PAGE_NUM_RE = re.compile(r"^\s*\d{1,4}\s*$")

# ── element-role heuristics ────────────────────────────────────────────
# Patterns that identify text whose *role* is context or evidence rather
# than a verifiable claim.  These never delete text — they only tag it.

_LEGAL_DISCLAIMER_RE = re.compile(
    r"(?i)(forward[- ]looking\s+statement|safe\s+harbor|"
    r"this\s+report\s+(does\s+not|should\s+not)|"
    r"no\s+guarantee|except\s+as\s+required\s+by\s+law|"
    r"cautionary\s+(note|statement)|"
    r"actual\s+results\s+may\s+differ|"
    r"risks?\s+and\s+uncertainties)"
)

_METHODOLOGY_RE = re.compile(
    r"(?i)(methodology|reporting\s+boundar|scope\s+of\s+this\s+report|"
    r"data\s+collection|calculation\s+method|"
    r"basis\s+of\s+preparation|reporting\s+period|"
    r"reporting\s+framework|GRI\s+\d{3}|SASB|TCFD|IFRS\s+S[12]|"
    r"materiality\s+(assessment|analysis|matrix)|"
    r"assurance\s+(statement|engagement|report)|"
    r"independent\s+(assurance|verification|auditor))"
)

_TOC_RE = re.compile(
    r"(?i)^(table\s+of\s+contents|contents)\s*$"
)

_TOC_ENTRY_RE = re.compile(
    r"^.{3,80}\s*\.{3,}\s*\d{1,4}\s*$"  # "Section Name ......... 42"
)


def _classify_element_role(
    text: str,
    element_type: str,
    section_path: list[str],
) -> str:
    """Assign an advisory role to a document element.

    Returns one of: ``"claim_candidate"``, ``"evidence"``,
    ``"context"``, ``"skip"``.

    This is intentionally conservative — when in doubt, it returns
    ``"claim_candidate"`` so that nothing is accidentally excluded
    from downstream claim detection.
    """
    # ── skip: TOC entries, page numbers ──
    if _TOC_RE.match(text):
        return "skip"
    if _TOC_ENTRY_RE.match(text):
        return "skip"
    if _PAGE_NUM_RE.match(text):
        return "skip"

    # ── evidence: tables are always evidence ──
    if element_type == "Table":
        return "evidence"

    # ── context: legal disclaimers ──
    if _LEGAL_DISCLAIMER_RE.search(text):
        return "context"

    # ── context: methodology / framework descriptions ──
    if _METHODOLOGY_RE.search(text):
        return "context"

    # ── context: third-party assurance statements ──
    section_lower = " ".join(section_path).lower()
    if any(kw in section_lower for kw in (
        "assurance", "verification", "auditor", "independent",
        "about this report", "reporting framework",
    )):
        # Text under assurance/methodology sections is context, not claims
        # UNLESS it contains first-person claim language
        if not re.search(r"(?i)\b(we|our|the company)\b.{5,30}\b(achiev|reduc|commit|increas|target|goal)", text):
            return "context"

    # ── default: claim candidate ──
    return "claim_candidate"


def _is_bold(flags: int) -> bool:
    """PyMuPDF span flags: bit 4 (16) = bold."""
    return bool(flags & 16)


def _clean(text: str) -> str:
    """Collapse whitespace, strip, normalise dashes/quotes."""
    text = text.replace("\u00a0", " ").replace("\u2002", " ").replace("\u2003", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _merge_hyphens(text: str) -> str:
    """Re-join words that were hyphenated at line breaks."""
    return re.sub(r"(\w)-\n(\w)", r"\1\2", text)


# ── main parser ─────────────────────────────────────────────────────────


class PDFParser:
    """General-purpose ESG-report PDF parser.

    Usage::

        parser = PDFParser()
        elements = parser.parse("report.pdf")
    """

    # Minimum length for a text block to be kept (filters short junk)
    MIN_TEXT_LEN = 15
    # Sidebar detection: blocks in the left/right 15% of the page with short text
    SIDEBAR_MARGIN_RATIO = 0.15
    SIDEBAR_MAX_CHARS = 60

    # A header must be at least this much larger than body text (pts)
    HEADER_SIZE_DELTA = 1.5

    # Maximum length of a header (chars).  Longer text is body.
    MAX_HEADER_LEN = 200

    # How many pages to sample for header/footer detection
    HF_SAMPLE_PAGES = 8

    # Fraction of pages a line must appear on to count as header/footer
    HF_FREQ_THRESHOLD = 0.4

    def __init__(self) -> None:
        pass

    # ── public API ──────────────────────────────────────────────────────

    def parse(self, pdf_path: str) -> list[dict]:
        """Parse *pdf_path* and return a list of DocumentElement dicts."""
        doc = fitz.open(pdf_path)
        try:
            return self._parse_document(doc)
        finally:
            doc.close()

    # ── internal pipeline ───────────────────────────────────────────────

    def _parse_document(self, doc: fitz.Document) -> list[dict]:
        # 1. Collect every span from every page (for global stats).
        pages_raw: list[list[dict]] = []  # per-page list of blocks
        all_spans: list[dict] = []

        for page in doc:
            data = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT, sort=True)
            blocks = data.get("blocks", [])
            pages_raw.append(blocks)
            for blk in blocks:
                if blk.get("type", -1) != 0:
                    continue
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        all_spans.append(span)

        if not all_spans:
            return []

        # 2. Determine the body font size (most common).
        body_size = self._determine_body_font_size(all_spans)

        # 3. Detect repeating headers / footers.
        hf_texts = self._detect_headers_and_footers(pages_raw)

        # 4. Walk pages, build elements.
        elements: list[dict] = []
        section_path: list[str] = []
        idx = 0

        for page_num, blocks in enumerate(pages_raw, start=1):
            page_obj = doc[page_num - 1]

            # 4a. Extract tables first (so we can skip their regions).
            table_elements, table_rects = self._extract_tables(page_obj, page_num)
            for te in table_elements:
                te["element_id"] = f"p{page_num}_{idx}"
                te["section_path"] = list(section_path)
                te["element_role"] = _classify_element_role(
                    te["text"], te["element_type"], section_path,
                )
                elements.append(te)
                idx += 1

            # 4b. Detect columns and sort blocks by column then by Y.
            page_width = page_obj.rect.width
            page_height = page_obj.rect.height
            text_blocks = [b for b in blocks if b.get("type", -1) == 0]
            text_blocks = self._sort_by_columns(text_blocks, page_width, page_height)

            pending_lines: list[str] = []
            pending_is_list = False

            def _flush() -> None:
                nonlocal idx, pending_lines, pending_is_list
                if not pending_lines:
                    return
                raw = "\n".join(pending_lines)
                raw = _merge_hyphens(raw)
                text = _clean(raw)
                if len(text) < self.MIN_TEXT_LEN:
                    pending_lines = []
                    pending_is_list = False
                    return
                etype = "ListItem" if pending_is_list else "NarrativeText"
                role = _classify_element_role(text, etype, section_path)
                elements.append(
                    {
                        "element_id": f"p{page_num}_{idx}",
                        "text": text,
                        "page": page_num,
                        "element_type": etype,
                        "section_path": list(section_path),
                        "table_data": None,
                        "element_role": role,
                    }
                )
                idx += 1
                pending_lines = []
                pending_is_list = False

            for blk in text_blocks:
                # Skip blocks that overlap a detected table rect.
                blk_rect = fitz.Rect(blk["bbox"])
                if any(blk_rect.intersects(tr) for tr in table_rects):
                    continue

                # Skip sidebar/navigation blocks.
                if self._is_sidebar_block(blk, page_width):
                    continue

                # Reassemble block text from spans.
                block_text_parts: list[str] = []
                block_sizes: list[float] = []
                block_bold_ratio = 0.0
                total_chars = 0
                bold_chars = 0

                for line in blk.get("lines", []):
                    line_parts: list[str] = []
                    for span in line.get("spans", []):
                        t = span.get("text", "")
                        if not t.strip():
                            continue
                        line_parts.append(t)
                        block_sizes.append(span["size"])
                        n = len(t.strip())
                        total_chars += n
                        if _is_bold(span.get("flags", 0)):
                            bold_chars += n
                    if line_parts:
                        block_text_parts.append(" ".join(line_parts))

                if not block_text_parts:
                    continue

                full_text = "\n".join(block_text_parts)
                clean_text = _clean(full_text)

                if total_chars > 0:
                    block_bold_ratio = bold_chars / total_chars

                # Filter header/footer repetitions (check full block AND individual lines).
                if self._is_header_footer(clean_text, hf_texts):
                    continue
                # Also check if ALL lines in this block are header/footer text
                block_lines = [l.strip() for l in clean_text.split("\n") if l.strip()]
                if block_lines and all(
                    self._is_header_footer(l, hf_texts) or len(l) < 4
                    for l in block_lines
                ):
                    continue

                # Filter page numbers.
                if _PAGE_NUM_RE.match(clean_text):
                    continue

                # Filter navigation menu text and repeating sidebars.
                if self._is_navigation_text(clean_text):
                    continue
                if self._is_repeating_sidebar(full_text, hf_texts):
                    continue

                # Filter standalone stats/numbers.
                if self._is_standalone_stat(clean_text):
                    continue

                # Determine dominant font size for this block.
                avg_size = sum(block_sizes) / len(block_sizes) if block_sizes else body_size

                # Check if this block is a section header.
                if self._is_header_block(clean_text, avg_size, body_size, block_bold_ratio):
                    _flush()
                    # Collapse newlines in titles to a single space.
                    title_text = re.sub(r"\s+", " ", clean_text).strip()
                    # Update section hierarchy.
                    level = self._header_level(avg_size, body_size)
                    section_path = section_path[: max(0, level - 1)]
                    section_path.append(title_text)

                    elements.append(
                        {
                            "element_id": f"p{page_num}_{idx}",
                            "text": title_text,
                            "page": page_num,
                            "element_type": "Title",
                            "section_path": list(section_path),
                            "table_data": None,
                            "element_role": "context",
                        }
                    )
                    idx += 1
                    continue

                # Is it a list item?
                is_list = bool(_BULLET_RE.match(clean_text))

                # If the element type switches (list vs narrative), flush.
                if pending_lines and is_list != pending_is_list:
                    _flush()

                if is_list:
                    # Each bullet is its own element.
                    _flush()
                    pending_lines = [clean_text]
                    pending_is_list = True
                    _flush()
                else:
                    pending_lines.append(clean_text)

            _flush()

        return elements

    # ── header / footer detection ───────────────────────────────────────

    def _detect_headers_and_footers(self, pages_raw: list[list[dict]]) -> set[str]:
        """Return a set of cleaned text strings that appear on many pages.

        These are likely running headers, footers, or navigation bars and
        should be excluded from the body content.
        """
        total_pages = len(pages_raw)
        if total_pages < 3:
            return set()

        # Sample pages evenly.
        step = max(1, total_pages // self.HF_SAMPLE_PAGES)
        sample_indices = list(range(0, total_pages, step))[: self.HF_SAMPLE_PAGES]
        if total_pages - 1 not in sample_indices:
            sample_indices.append(total_pages - 1)

        # For each sampled page, collect short-ish text blocks near
        # the top or bottom of the page.
        line_counter: Counter[str] = Counter()

        for pi in sample_indices:
            blocks = pages_raw[pi]
            page_texts: set[str] = set()
            for blk in blocks:
                if blk.get("type", -1) != 0:
                    continue
                text_parts = []
                for line in blk.get("lines", []):
                    for span in line.get("spans", []):
                        t = span.get("text", "").strip()
                        if t:
                            text_parts.append(t)
                raw = _clean(" ".join(text_parts))
                if not raw:
                    continue
                # Only consider texts (< 300 chars) — very long paragraphs
                # won't repeat verbatim across pages.
                if len(raw) > 300:
                    continue
                # Strip trailing page numbers for matching.
                normalized = re.sub(r"\s*\d{1,4}\s*$", "", raw).strip()
                if normalized and len(normalized) > 3:
                    page_texts.add(normalized)
            for t in page_texts:
                line_counter[t] += 1

        threshold = max(2, int(len(sample_indices) * self.HF_FREQ_THRESHOLD))
        return {t for t, cnt in line_counter.items() if cnt >= threshold}

    def _is_header_footer(self, text: str, hf_texts: set[str]) -> bool:
        if not hf_texts:
            return False
        normalized = re.sub(r"\s*\d{1,4}\s*$", "", text).strip()
        return normalized in hf_texts

    # ── column detection ───────────────────────────────────────────────

    # Fraction of page height used as header/footer exclusion zone for
    # column detection.  Navigation bars at the top of ESG reports
    # pollute X-position clustering because their menu items fill gaps
    # between content columns.
    COL_HEADER_ZONE = 0.08
    COL_FOOTER_ZONE = 0.92
    # Minimum gap between X-position clusters to count as a new column,
    # expressed as a fraction of page width.
    COL_GAP_RATIO = 0.12

    @staticmethod
    def _sort_by_columns(
        blocks: list[dict],
        page_width: float,
        page_height: float = 0.0,
    ) -> list[dict]:
        """Detect columns and sort blocks: column-by-column, top-to-bottom.

        Uses a two-pass approach:
        1. Detect column boundaries using only *content-area* blocks
           (excluding header/footer zones that contain navigation menus
           whose scattered X positions break gap-based detection).
        2. Assign *all* blocks to the detected columns and sort each
           column top-to-bottom, concatenating columns left-to-right.
        """
        if not blocks:
            return blocks

        # ── Pass 1: detect columns from content-area blocks only ──────
        header_cutoff = page_height * PDFParser.COL_HEADER_ZONE if page_height else 0
        footer_cutoff = page_height * PDFParser.COL_FOOTER_ZONE if page_height else float("inf")
        gap_threshold = page_width * PDFParser.COL_GAP_RATIO

        positioned = []
        content_xs: list[float] = []

        for blk in blocks:
            bbox = blk.get("bbox", [0, 0, page_width, 0])
            lx, ty = bbox[0], bbox[1]
            positioned.append((lx, ty, blk))
            # Only use content-area blocks for X clustering
            if header_cutoff < ty < footer_cutoff:
                content_xs.append(round(lx, -1))

        if not content_xs:
            # No content blocks — fall back to Y sort
            return sorted(blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        left_xs = sorted(set(content_xs))

        if len(left_xs) <= 1:
            return sorted(blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        # Find column boundaries via gap detection
        col_groups: list[list[float]] = [[left_xs[0]]]
        for i in range(1, len(left_xs)):
            gap = left_xs[i] - left_xs[i - 1]
            if gap > gap_threshold:
                col_groups.append([left_xs[i]])
            else:
                col_groups[-1].append(left_xs[i])

        if len(col_groups) <= 1:
            return sorted(blocks, key=lambda b: b.get("bbox", [0, 0])[1])

        # ── Pass 2: assign ALL blocks to nearest column ───────────────
        col_boundaries: list[tuple[float, float]] = []
        for grp in col_groups:
            col_boundaries.append((min(grp), max(grp)))

        result: list[dict] = []
        assigned: set[int] = set()

        for col_min, col_max in col_boundaries:
            col_blocks = []
            for lx, ty, blk in positioned:
                rounded_lx = round(lx, -1)
                if col_min <= rounded_lx <= col_max + 20:
                    col_blocks.append((ty, blk))
                    assigned.add(id(blk))
            col_blocks.sort(key=lambda x: x[0])
            result.extend(blk for _, blk in col_blocks)

        # Safety: add any unassigned blocks at the end (sorted by Y)
        for lx, ty, blk in positioned:
            if id(blk) not in assigned:
                result.append(blk)

        return result

    # ── sidebar / navigation filtering ──────────────────────────────────

    _NAV_PATTERNS = re.compile(
        r"^(message from|introduction|table of contents|contents|"
        r"about this report|appendix|glossary|index|"
        r"people.*diversity|energy.*climate|responsible|sustainability)\s*$",
        re.IGNORECASE,
    )

    def _is_sidebar_block(self, blk: dict, page_width: float) -> bool:
        """Detect sidebar/navigation blocks at page edges with short text."""
        bbox = blk.get("bbox", [0, 0, page_width, 0])
        blk_left, blk_right = bbox[0], bbox[2]
        blk_width = blk_right - blk_left

        # If the block is narrow and in the left margin
        margin = page_width * self.SIDEBAR_MARGIN_RATIO
        if blk_right < margin and blk_width < margin:
            return True

        # Get text length
        text_len = sum(
            len(span.get("text", "").strip())
            for line in blk.get("lines", [])
            for span in line.get("spans", [])
        )

        # Narrow blocks at edges with short text are likely sidebar nav
        if blk_width < page_width * 0.25 and text_len < self.SIDEBAR_MAX_CHARS:
            if blk_left < margin or blk_left > page_width * (1 - self.SIDEBAR_MARGIN_RATIO):
                return True

        return False

    def _is_navigation_text(self, text: str) -> bool:
        """Detect navigation menu text (section names listed vertically)."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 3:
            return False
        # If most lines are very short (< 30 chars) and there are many,
        # it's likely a navigation/TOC sidebar
        short_lines = sum(1 for l in lines if len(l) < 30)
        if short_lines >= len(lines) * 0.7 and len(lines) >= 4:
            return True
        return False

    def _is_repeating_sidebar(self, text: str, hf_texts: set[str]) -> bool:
        """Detect blocks where MOST content matches header/footer patterns.

        Sidebar navigation is often assembled from multiple small blocks
        that are individually detected as repeating, but then concatenated.
        """
        if not hf_texts:
            return False
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 2:
            return False
        matching = 0
        for line in lines:
            normalized = re.sub(r"\s*\d{1,4}\s*$", "", line).strip()
            if normalized in hf_texts or len(normalized) < 4:
                matching += 1
        # If > 60% of lines match known repeating text, it's a sidebar
        return matching >= len(lines) * 0.6

    @staticmethod
    def _is_standalone_stat(text: str) -> bool:
        """Detect standalone numbers/stats that aren't meaningful paragraphs."""
        clean = text.strip().rstrip("%+,.")
        # Pure numbers like "505,000+" or "99.7%"
        if re.match(r"^[\d,.\s%+$€£¥]+$", clean):
            return True
        # Very short text that's mostly numbers
        if len(text) < 20 and sum(c.isdigit() for c in text) > len(text) * 0.5:
            return True
        return False

    # ── font-size analysis ──────────────────────────────────────────────

    def _determine_body_font_size(self, all_spans: list[dict]) -> float:
        """Return the most common font size across the document."""
        size_counter: Counter[float] = Counter()
        for span in all_spans:
            t = span.get("text", "").strip()
            if not t:
                continue
            # Weight by character count so long paragraphs dominate.
            sz = round(span["size"], 1)
            size_counter[sz] += len(t)
        if not size_counter:
            return 10.0
        return size_counter.most_common(1)[0][0]

    # ── header detection ────────────────────────────────────────────────

    def _is_header_block(
        self,
        text: str,
        avg_size: float,
        body_size: float,
        bold_ratio: float,
    ) -> bool:
        """Decide whether a text block is a section header."""
        # Too long to be a header.
        if len(text) > self.MAX_HEADER_LEN:
            return False

        # Very short fragments are not headers either.
        if len(text) < 2:
            return False

        # Must not look like a page number.
        if _PAGE_NUM_RE.match(text):
            return False

        # Size significantly larger than body → header.
        if avg_size >= body_size + self.HEADER_SIZE_DELTA:
            return True

        # Same size but fully bold and short → sub-header.
        if bold_ratio >= 0.9 and len(text) <= 80 and avg_size >= body_size:
            return True

        return False

    def _header_level(self, avg_size: float, body_size: float) -> int:
        """Map font size to nesting level (1 = top, 2 = sub, …)."""
        delta = avg_size - body_size
        if delta > 6:
            return 1
        if delta > 3:
            return 2
        return 3  # bold sub-header at body size

    # ── table extraction ────────────────────────────────────────────────

    def _extract_tables(
        self, page: fitz.Page, page_num: int
    ) -> tuple[list[dict], list[fitz.Rect]]:
        """Extract tables from *page* using PyMuPDF's built-in detector.

        Returns (list_of_table_elements, list_of_table_rects).
        The rects are used to exclude overlapping text blocks.
        """
        table_elements: list[dict] = []
        table_rects: list[fitz.Rect] = []

        try:
            tables_result = page.find_tables()
        except Exception:
            return table_elements, table_rects

        if not tables_result or not tables_result.tables:
            return table_elements, table_rects

        for tbl in tables_result.tables:
            try:
                data = tbl.extract()
            except Exception:
                continue

            if not data:
                continue

            # Clean cell values.
            clean_data: list[list[str]] = []
            for row in data:
                clean_row = []
                for cell in row:
                    if cell is None:
                        clean_row.append("")
                    else:
                        clean_row.append(_clean(str(cell)))
                clean_data.append(clean_row)

            # Skip degenerate tables (single cell or all empty).
            non_empty_cells = sum(
                1 for row in clean_data for c in row if c.strip()
            )
            if non_empty_cells < 2:
                continue

            # Build a plain-text representation for search/RAG.
            text_lines: list[str] = []
            for row in clean_data:
                text_lines.append(" | ".join(row))
            table_text = "\n".join(text_lines)

            table_elements.append(
                {
                    "element_id": "",  # filled in by caller
                    "text": table_text,
                    "page": page_num,
                    "element_type": "Table",
                    "section_path": [],  # filled in by caller
                    "table_data": clean_data,
                }
            )

            # Record the bounding rect so we skip overlapping text.
            try:
                table_rects.append(fitz.Rect(tbl.bbox))
            except Exception:
                pass

        return table_elements, table_rects


# ── CLI convenience ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m pipeline.pdf_parser <path.pdf>")
        sys.exit(1)

    parser = PDFParser()
    elements = parser.parse(sys.argv[1])

    titles = [e for e in elements if e["element_type"] == "Title"]
    narratives = [e for e in elements if e["element_type"] == "NarrativeText"]
    lists = [e for e in elements if e["element_type"] == "ListItem"]
    tables = [e for e in elements if e["element_type"] == "Table"]

    print(f"Total elements: {len(elements)}")
    print(f"  Titles:     {len(titles)}")
    print(f"  Narrative:  {len(narratives)}")
    print(f"  ListItems:  {len(lists)}")
    print(f"  Tables:     {len(tables)}")
    print()

    print("First 10 titles:")
    for t in titles[:10]:
        print(f"  p{t['page']:>3}: {t['text'][:80]}")

    print()
    print("First 5 narrative paragraphs:")
    for n in narratives[:5]:
        sec = " > ".join(n["section_path"]) if n["section_path"] else "(no section)"
        print(f"  p{n['page']:>3} [{sec}]")
        print(f"        {n['text'][:120]}...")

    if tables:
        print()
        print(f"First table (page {tables[0]['page']}):")
        for row in (tables[0]["table_data"] or [])[:5]:
            print(f"  {row}")
