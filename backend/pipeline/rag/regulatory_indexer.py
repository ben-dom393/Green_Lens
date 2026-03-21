"""Indexes regulatory/standards PDFs into a separate ChromaDB collection and BM25S index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Any

import bm25s
import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

from config import DATA_DIR, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

REGULATORY_DIR = DATA_DIR / "external" / "regulatory"


class RegulatoryIndexer:
    """Indexes regulatory/standards PDFs for RAG retrieval."""

    def __init__(self):
        self._embedding_model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._bm25: bm25s.BM25 | None = None
        self._documents: list[str] = []
        self._metadata: list[dict] = []
        self._indexed: bool = False

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_embedding_model(self) -> SentenceTransformer:
        """Lazy load sentence-transformers model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model for regulatory index: %s", EMBEDDING_MODEL)
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _get_collection(self) -> chromadb.Collection:
        """Return (or create) the in-memory ChromaDB collection for regulatory docs."""
        if self._collection is None:
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name="regulatory_db",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # PDF parsing
    # ------------------------------------------------------------------

    def _parse_pdf_to_chunks(self, pdf_path: Path) -> list[dict]:
        """Extract text chunks from a single PDF.

        Uses PyMuPDF (fitz) to read pages and splits text into paragraphs.
        Each chunk: {"text": str, "source": filename, "page": int, "document_name": str}
        """
        chunks: list[dict] = []
        document_name = pdf_path.stem.replace("_", " ")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception:
            logger.warning("Could not open PDF: %s", pdf_path, exc_info=True)
            return chunks

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text", sort=True)
            if not text or not text.strip():
                continue

            # Split into paragraphs on double newlines
            paragraphs = text.split("\n\n")

            for para in paragraphs:
                para = para.strip()
                # Replace single newlines within a paragraph with spaces
                para = " ".join(line.strip() for line in para.split("\n") if line.strip())

                # Filter very short chunks
                if len(para) < 30:
                    continue

                # If a paragraph is very long (>800 chars), split further
                if len(para) > 800:
                    # Split on sentence boundaries
                    sentences = para.replace(". ", ".\n").split("\n")
                    current = ""
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        if current and len(current) + len(sentence) > 500:
                            if len(current) >= 30:
                                chunks.append({
                                    "text": current.strip(),
                                    "source": pdf_path.name,
                                    "page": page_num + 1,  # 1-indexed
                                    "document_name": document_name,
                                })
                            current = sentence
                        else:
                            current = (current + " " + sentence).strip() if current else sentence
                    if current and len(current) >= 30:
                        chunks.append({
                            "text": current.strip(),
                            "source": pdf_path.name,
                            "page": page_num + 1,
                            "document_name": document_name,
                        })
                else:
                    chunks.append({
                        "text": para,
                        "source": pdf_path.name,
                        "page": page_num + 1,
                        "document_name": document_name,
                    })

        doc.close()
        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ensure_indexed(self) -> None:
        """Build index from all regulatory PDFs (only once)."""
        if self._indexed:
            return

        if not REGULATORY_DIR.exists():
            logger.warning(
                "Regulatory directory not found: %s — skipping regulatory indexing.",
                REGULATORY_DIR,
            )
            self._indexed = True
            return

        # Find all PDFs recursively
        pdf_files = sorted(REGULATORY_DIR.rglob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", REGULATORY_DIR)
            self._indexed = True
            return

        logger.info("Found %d regulatory PDFs to index.", len(pdf_files))

        # Parse all PDFs
        all_chunks: list[dict] = []
        for pdf_path in pdf_files:
            chunks = self._parse_pdf_to_chunks(pdf_path)
            logger.info("  %s: %d chunks", pdf_path.name, len(chunks))
            all_chunks.extend(chunks)

        if not all_chunks:
            logger.warning("No chunks extracted from regulatory PDFs.")
            self._indexed = True
            return

        logger.info("Total regulatory chunks: %d", len(all_chunks))

        # Prepare parallel lists
        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for idx, chunk in enumerate(all_chunks):
            texts.append(chunk["text"])
            ids.append(f"reg_chunk_{idx}")
            metadatas.append({
                "source_file": chunk["source"],
                "page": chunk["page"],
                "document_name": chunk["document_name"],
            })

        # ------ Vector index (ChromaDB) ------
        model = self._load_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection = self._get_collection()
        # Add in batches to avoid exceeding any size limits
        batch_size = 5000
        for i in range(0, len(texts), batch_size):
            end = min(i + batch_size, len(texts))
            collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=texts[i:end],
                metadatas=metadatas[i:end],
            )
        logger.info("Added %d chunks to ChromaDB 'regulatory_db' collection.", len(texts))

        # ------ Sparse index (BM25S) ------
        corpus_tokens = bm25s.tokenize(texts)
        bm25_index = bm25s.BM25()
        bm25_index.index(corpus_tokens)
        self._bm25 = bm25_index
        logger.info("Built BM25 index over %d regulatory chunks.", len(texts))

        # Store for later lookup
        self._documents = texts
        self._metadata = metadatas
        self._indexed = True

        logger.info("Regulatory indexing complete.")

    def get_collection(self) -> chromadb.Collection | None:
        """Return the regulatory ChromaDB collection."""
        return self._collection

    def get_bm25(self) -> bm25s.BM25 | None:
        """Return BM25 index."""
        return self._bm25

    def get_documents(self) -> list[str]:
        """Return indexed document texts."""
        return list(self._documents)

    def get_metadata(self) -> list[dict]:
        """Return metadata parallel to indexed documents."""
        return list(self._metadata)
