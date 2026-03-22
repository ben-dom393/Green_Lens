"""Indexes document elements into ChromaDB (vector) and BM25S (sparse) for hybrid retrieval."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Any

import bm25s
import chromadb
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL

logger = logging.getLogger(__name__)


class DocumentIndexer:
    """Indexes parsed document for RAG retrieval."""

    def __init__(self):
        self._embedding_model: SentenceTransformer | None = None
        self._client: chromadb.ClientAPI | None = None
        self._collection: chromadb.Collection | None = None
        self._bm25: bm25s.BM25 | None = None
        self._documents: list[str] = []  # raw texts for BM25 results
        self._metadata: list[dict] = []  # metadata parallel to documents

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_embedding_model(self) -> SentenceTransformer:
        """Lazy load sentence-transformers model."""
        if self._embedding_model is None:
            logger.info("Loading embedding model: %s", EMBEDDING_MODEL)
            self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    def _get_collection(self) -> chromadb.Collection:
        """Return (or create) the in-memory ChromaDB collection."""
        if self._collection is None:
            self._client = chromadb.Client()
            self._collection = self._client.get_or_create_collection(
                name="esg_report",
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    # ------------------------------------------------------------------
    # Chunking helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _element_text(element: Any) -> str:
        """Extract text from a DocumentElement or plain dict."""
        if isinstance(element, dict):
            return element.get("text", "")
        return getattr(element, "text", "")

    @staticmethod
    def _element_meta(element: Any) -> dict:
        """Extract metadata fields from a DocumentElement or dict."""
        if isinstance(element, dict):
            sp = element.get("section_path", "")
        else:
            sp = getattr(element, "section_path", "")
        # Normalise section_path: list → string
        if isinstance(sp, list):
            sp = " > ".join(sp) if sp else ""

        if isinstance(element, dict):
            return {
                "element_id": element.get("element_id", ""),
                "page": element.get("page", 0),
                "section_path": sp,
            }
        return {
            "element_id": getattr(element, "element_id", ""),
            "page": getattr(element, "page", 0),
            "section_path": sp,
        }

    def _build_chunks(self, elements: list) -> list[dict]:
        """Merge consecutive elements from the same section into ~500-char chunks.

        Returns a list of dicts:
            {"text": str, "element_ids": list[str], "pages": set[int], "section": str}
        """
        chunks: list[dict] = []
        current_text = ""
        current_ids: list[str] = []
        current_pages: set[int] = set()
        current_section = ""

        for elem in elements:
            text = self._element_text(elem).strip()
            if not text:
                continue

            meta = self._element_meta(elem)
            section = meta["section_path"]

            # Start a new chunk when the section changes or adding would exceed ~500 chars
            if current_text and (section != current_section or len(current_text) + len(text) > 500):
                chunks.append(
                    {
                        "text": current_text.strip(),
                        "element_ids": list(current_ids),
                        "pages": set(current_pages),
                        "section": current_section,
                    }
                )
                current_text = ""
                current_ids = []
                current_pages = set()
                current_section = ""

            current_text += (" " if current_text else "") + text
            eid = meta["element_id"]
            if eid:
                current_ids.append(eid)
            current_pages.add(meta["page"])
            current_section = section

        # Flush remaining
        if current_text.strip():
            chunks.append(
                {
                    "text": current_text.strip(),
                    "element_ids": list(current_ids),
                    "pages": set(current_pages),
                    "section": current_section,
                }
            )

        return chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(self, elements: list) -> None:
        """Index document elements into both vector and sparse indexes.

        Args:
            elements: list of DocumentElement (or dicts with text, element_id,
                      page, section_path).
        """
        chunks = self._build_chunks(elements)
        if not chunks:
            logger.warning("No chunks produced from %d elements — nothing to index.", len(elements))
            return

        logger.info("Indexing %d chunks from %d elements.", len(chunks), len(elements))

        # Prepare parallel lists
        texts: list[str] = []
        ids: list[str] = []
        metadatas: list[dict] = []

        for idx, chunk in enumerate(chunks):
            texts.append(chunk["text"])
            ids.append(f"chunk_{idx}")
            metadatas.append(
                {
                    "page": min(chunk["pages"]) if chunk["pages"] else 0,
                    "element_ids": ",".join(chunk["element_ids"]),
                    "section": chunk["section"] if chunk["section"] else "unknown",
                }
            )

        # ------ Vector index (ChromaDB) ------
        model = self._load_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection = self._get_collection()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Added %d chunks to ChromaDB collection.", len(texts))

        # ------ Sparse index (BM25S) ------
        corpus_tokens = bm25s.tokenize(texts)
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self._bm25 = retriever
        logger.info("Built BM25 index over %d chunks.", len(texts))

        # Store for later lookup
        self._documents = texts
        self._metadata = metadatas

    def get_collection(self) -> chromadb.Collection | None:
        """Return ChromaDB collection for querying."""
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
