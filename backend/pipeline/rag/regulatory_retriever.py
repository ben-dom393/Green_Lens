"""Retrieves relevant regulatory paragraphs using hybrid search over the regulatory index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging

import bm25s

from config import RERANKER_MODEL
from pipeline.rag.regulatory_indexer import RegulatoryIndexer

logger = logging.getLogger(__name__)


class RegulatoryRetriever:
    """Retrieves relevant regulatory paragraphs for a given claim."""

    RRF_K = 60  # reciprocal rank fusion constant

    def __init__(self, indexer: RegulatoryIndexer):
        self.indexer = indexer
        self._reranker = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _load_reranker(self):
        """Lazy load cross-encoder reranker."""
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker model for regulatory retriever: %s", RERANKER_MODEL)
                self._reranker = CrossEncoder(RERANKER_MODEL)
            except Exception:
                logger.warning(
                    "Failed to load reranker model — reranking will be skipped.",
                    exc_info=True,
                )
                self._reranker = None

    # ------------------------------------------------------------------
    # Individual retrieval backends
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, n: int = 20) -> list[dict]:
        """Retrieve top-n results from the regulatory ChromaDB collection."""
        collection = self.indexer.get_collection()
        if collection is None or collection.count() == 0:
            return []

        try:
            results = collection.query(
                query_texts=[query], n_results=min(n, collection.count())
            )
        except Exception:
            logger.warning("Regulatory ChromaDB query failed.", exc_info=True)
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        out: list[dict] = []
        for rank, (text, meta) in enumerate(zip(docs, metas)):
            out.append({
                "text": text,
                "source": meta.get("source_file", ""),
                "page": meta.get("page", 0),
                "document_name": meta.get("document_name", ""),
                "rank": rank,
            })
        return out

    def _bm25_search(self, query: str, n: int = 20) -> list[dict]:
        """Retrieve top-n results from the regulatory BM25 index."""
        bm25 = self.indexer.get_bm25()
        if bm25 is None:
            return []

        all_docs = self.indexer.get_documents()
        all_meta = self.indexer.get_metadata()
        if not all_docs:
            return []

        try:
            query_tokens = bm25s.tokenize([query])
            results, scores = bm25.retrieve(query_tokens, k=min(n, len(all_docs)))
        except Exception:
            logger.warning("Regulatory BM25 retrieval failed.", exc_info=True)
            return []

        out: list[dict] = []
        indices = results[0]
        for rank, idx in enumerate(indices):
            idx = int(idx)
            if idx < 0 or idx >= len(all_docs):
                continue
            meta = all_meta[idx] if idx < len(all_meta) else {}
            out.append({
                "text": all_docs[idx],
                "source": meta.get("source_file", ""),
                "page": meta.get("page", 0),
                "document_name": meta.get("document_name", ""),
                "rank": rank,
            })
        return out

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_merge(
        self, vector_results: list[dict], bm25_results: list[dict]
    ) -> list[dict]:
        """Merge two ranked lists using Reciprocal Rank Fusion."""
        k = self.RRF_K
        scored: dict[str, dict] = {}

        for res in vector_results:
            key = res["text"]
            if key not in scored:
                scored[key] = {**res, "rrf_score": 0.0}
            scored[key]["rrf_score"] += 1.0 / (k + res["rank"])

        for res in bm25_results:
            key = res["text"]
            if key not in scored:
                scored[key] = {**res, "rrf_score": 0.0}
            scored[key]["rrf_score"] += 1.0 / (k + res["rank"])

        merged = sorted(scored.values(), key=lambda d: d["rrf_score"], reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Rerank candidates with a cross-encoder."""
        self._load_reranker()
        if self._reranker is None or not candidates:
            return candidates[:top_k]

        try:
            pairs = [(query, doc["text"]) for doc in candidates]
            scores = self._reranker.predict(pairs)

            for doc, score in zip(candidates, scores):
                doc["rerank_score"] = float(score)

            reranked = sorted(
                candidates, key=lambda d: d["rerank_score"], reverse=True
            )
            return reranked[:top_k]
        except Exception:
            logger.warning(
                "Regulatory reranking failed — returning RRF order.", exc_info=True
            )
            return candidates[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        """Retrieve top-k regulatory paragraphs relevant to the query.

        Ensures the regulatory index is built before querying.

        Returns:
            list of dicts with keys:
                text (str), score (float), source (str), page (int),
                document_name (str)
        """
        # Ensure indexed before first query
        self.indexer.ensure_indexed()

        # 1. Semantic search
        vector_results = self._vector_search(query, n=20)

        # 2. Keyword search
        bm25_results = self._bm25_search(query, n=20)

        if not vector_results and not bm25_results:
            logger.debug("No regulatory results for query: %s", query[:80])
            return []

        # 3. Reciprocal Rank Fusion
        merged = self._rrf_merge(vector_results, bm25_results)

        # 4. Cross-encoder reranking
        reranked = self._rerank(query, merged, top_k)

        # 5. Normalise output format
        output: list[dict] = []
        for doc in reranked:
            score = doc.get("rerank_score", doc.get("rrf_score", 0.0))
            output.append({
                "text": doc["text"],
                "score": score,
                "source": doc.get("source", ""),
                "page": doc.get("page", 0),
                "document_name": doc.get("document_name", ""),
            })

        return output
