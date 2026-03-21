"""
PDF Parser for ESG & Annual Reports
Uses a multi-library approach for maximum extraction coverage:
  - PyMuPDF (fitz): fast text + metadata extraction
  - pypdf: reads OCR'd output
  - ocrmypdf + Tesseract: OCR for scanned/image-based pages
"""

import os
import subprocess
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
from pydantic import BaseModel, Field
from pypdf import PdfReader


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

    # ── Metadata ─────────────────────────────────────────────────

    def _extract_metadata(self, file_path: Path) -> ReportMetadata:
        """Extract PDF metadata using PyMuPDF"""
        doc = fitz.open(str(file_path))
        meta = doc.metadata or {}
        size_mb = round(file_path.stat().st_size / (1024 * 1024), 2)

        report_meta = ReportMetadata(
            title=meta.get("title", "") or "",
            author=meta.get("author", "") or "",
            subject=meta.get("subject", "") or "",
            creation_date=meta.get("creationDate", "") or "",
            page_count=len(doc),
            file_size_mb=size_mb,
        )
        doc.close()
        return report_meta

    # ── Page-by-page Extraction ──────────────────────────────────

    def _extract_pages(self, file_path: Path) -> list[PageContent]:
        """Extract text from every page"""
        pages = []

        doc = fitz.open(str(file_path))

        for i, page in enumerate(doc):
            text = page.get_text("text")
            is_scanned = len(text.strip()) < 50

            # If page looks scanned, try OCR
            if is_scanned and self.ocr_enabled:
                ocr_text = self._ocr_page(file_path, i)
                if ocr_text:
                    text = ocr_text

            pages.append(PageContent(
                page_number=i + 1,
                text=text.strip(),
                is_scanned=is_scanned,
            ))

        doc.close()
        return pages

    # ── OCR Fallback ─────────────────────────────────────────────

    def _ocr_page(self, file_path: Path, page_index: int) -> str:
        """OCR a single page using ocrmypdf + Tesseract"""
        try:
            doc = fitz.open(str(file_path))
            single_page_doc = fitz.open()
            single_page_doc.insert_pdf(doc, from_page=page_index, to_page=page_index)

            with tempfile.TemporaryDirectory() as tmp_dir:
                input_path = os.path.join(tmp_dir, "page.pdf")
                output_path = os.path.join(tmp_dir, "page_ocr.pdf")
                single_page_doc.save(input_path)
                single_page_doc.close()
                doc.close()

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
                    timeout=60,
                )

                if os.path.exists(output_path):
                    reader = PdfReader(output_path)
                    if reader.pages:
                        return reader.pages[0].extract_text() or ""

        except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
            print(f"OCR failed for page {page_index + 1}: {e}")

        return ""


# ── Convenience function ─────────────────────────────────────────

def parse_report(file_path: str | Path, ocr: bool = True) -> ParsedReport:
    """Quick helper to parse a PDF in one call"""
    parser = PDFParser(ocr_enabled=ocr)
    return parser.parse(file_path)
