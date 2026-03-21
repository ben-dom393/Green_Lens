"""
PDF Parser for ESG & Annual Reports
Uses a multi-library approach for maximum extraction coverage:
  - PyMuPDF (fitz): fast text + metadata extraction
  - pypdf: reads OCR'd output
  - ocrmypdf + Tesseract: OCR for scanned/image-based pages
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field
from pypdf import PdfReader

try:
    import fitz  # PyMuPDF
except ImportError as exc:
    fitz = None
    FITZ_IMPORT_ERROR = exc
else:
    FITZ_IMPORT_ERROR = None


# ── Pydantic Models ──────────────────────────────────────────────

class PageContent(BaseModel):
    """Extracted content from a single PDF page"""
    page_number: int
    text: str = ""
    is_scanned: bool = False


class ReportMetadata(BaseModel):
    """Metadata extracted from the PDF"""
    title: str = ""
    author: str = ""
    subject: str = ""
    creation_date: str = ""
    page_count: int = 0
    file_size_mb: float = 0.0


class ParsedReport(BaseModel):
    """Complete parsed output of an ESG/Annual report"""
    file_path: str
    metadata: ReportMetadata
    pages: list[PageContent] = Field(default_factory=list)
    full_text: str = ""


# ── Core Parser ──────────────────────────────────────────────────

class PDFParser:
    """
    Multi-library PDF parser optimised for ESG and annual reports.

    Strategy:
      1. PyMuPDF for fast text extraction + metadata
      2. If a page has very little text → OCR it with ocrmypdf/Tesseract
    """

    def __init__(self, ocr_enabled: bool = True, ocr_language: str = "eng"):
        self.ocr_enabled = ocr_enabled
        self.ocr_language = ocr_language

    def parse(self, file_path: str | Path) -> ParsedReport:
        """Parse a PDF file and return structured content"""
        self._require_pymupdf()
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")

        metadata = self._extract_metadata(file_path)
        pages = self._extract_pages(file_path)
        full_text = "\n\n".join(p.text for p in pages if p.text)

        return ParsedReport(
            file_path=str(file_path),
            metadata=metadata,
            pages=pages,
            full_text=full_text,
        )

    def _require_pymupdf(self) -> None:
        """Raise a clear error if PyMuPDF is unavailable."""
        if fitz is None:
            raise ImportError(
                "PyMuPDF is required for PDF parsing. Install dependencies with "
                "`python -m pip install -r requirements.txt`."
            ) from FITZ_IMPORT_ERROR

    # ── Metadata ─────────────────────────────────────────────────

    def _extract_metadata(self, file_path: Path) -> ReportMetadata:
        """Extract PDF metadata using PyMuPDF"""
        doc = fitz.open(str(file_path))
        try:
            meta = doc.metadata or {}
            size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

            return ReportMetadata(
                title=meta.get("title", "") or "",
                author=meta.get("author", "") or "",
                subject=meta.get("subject", "") or "",
                creation_date=self._normalise_pdf_date(meta.get("creationDate", "") or ""),
                page_count=len(doc),
                file_size_mb=size_mb,
            )
        finally:
            doc.close()

    # ── Page-by-page Extraction ──────────────────────────────────

    def _extract_pages(self, file_path: Path) -> list[PageContent]:
        """Extract text from every page"""
        pages = []

        doc = fitz.open(str(file_path))
        try:
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                is_scanned = self._page_needs_ocr(page, text)

                if is_scanned and self.ocr_enabled:
                    ocr_text = self._ocr_page(file_path, i)
                    if ocr_text:
                        text = ocr_text.strip()

                pages.append(PageContent(
                    page_number=i + 1,
                    text=text,
                    is_scanned=is_scanned,
                ))
        finally:
            doc.close()

        return pages
    # ── OCR Fallback ─────────────────────────────────────────────

    def _ocr_page(self, file_path: Path, page_index: int) -> str:
        """OCR a single page using ocrmypdf + Tesseract"""
        if shutil.which("ocrmypdf") is None:
            return ""

        doc = None
        single_page_doc = None
        try:
            doc = fitz.open(str(file_path))
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)

            with tempfile.TemporaryDirectory() as tmp_dir:
                input_path = os.path.join(tmp_dir, "page.pdf")
                output_path = os.path.join(tmp_dir, "page_ocr.pdf")
                single_page_doc.save(input_path)

                subprocess.run(
                    [
                        "ocrmypdf",
                        "--language", self.ocr_language,
                        "--force-ocr",
                        "--skip-big", "50",
                        input_path,
                        output_path,
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=60,
                )

                if os.path.exists(output_path):
                    reader = PdfReader(output_path)
                    if reader.pages:
                        return reader.pages[0].extract_text() or ""

        except subprocess.TimeoutExpired as exc:
            print(f"OCR timed out for page {page_index + 1}: {exc}")
        except subprocess.CalledProcessError as exc:
            stderr = (exc.stderr or "").strip()
            print(f"OCR failed for page {page_index + 1}: {stderr or exc}")
        except Exception as exc:
            print(f"OCR failed for page {page_index + 1}: {exc}")
        finally:
            if single_page_doc is not None:
                single_page_doc.close()
            if doc is not None:
                doc.close()

        return ""

    def _page_needs_ocr(self, page, text: str) -> bool:
        """Heuristic for image-heavy pages that appear to lack extractable text."""
        if len(text) >= 50:
            return False

        has_images = bool(page.get_images(full=True))
        return has_images or len(text) == 0

    def _normalise_pdf_date(self, raw_date: str) -> str:
        """Convert PDF date strings like D:20250321091500 into a readable form."""
        if not raw_date.startswith("D:"):
            return raw_date

        digits = "".join(char for char in raw_date[2:] if char.isdigit())
        if len(digits) < 8:
            return raw_date

        year = digits[0:4]
        month = digits[4:6]
        day = digits[6:8]
        time_parts = []
        if len(digits) >= 10:
            time_parts.append(digits[8:10])
        if len(digits) >= 12:
            time_parts.append(digits[10:12])
        if len(digits) >= 14:
            time_parts.append(digits[12:14])

        if time_parts:
            return f"{year}-{month}-{day} {':'.join(time_parts)}"
        return f"{year}-{month}-{day}"

# ── Convenience function ─────────────────────────────────────────

def parse_report(file_path: str | Path, ocr: bool = True) -> ParsedReport:
    """Quick helper to parse a PDF in one call"""
    parser = PDFParser(ocr_enabled=ocr)
    return parser.parse(file_path)
