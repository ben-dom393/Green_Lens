"""Hybrid retriever: ChromaDB (semantic) + BM25S (keyword) with RRF and cross-encoder reranking."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import logging
from typing import Any

import bm25s

from config import RERANKER_MODEL

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Hybrid search: vector + BM25 + cross-encoder reranking."""

    RRF_K = 60  # reciprocal rank fusion constant

    def __init__(self, indexer: Any):
        """
        Args:
            indexer: A DocumentIndexer instance (already populated).
        """
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

                logger.info("Loading reranker model: %s", RERANKER_MODEL)
                self._reranker = CrossEncoder(RERANKER_MODEL)
            except Exception:
                logger.warning("Failed to load reranker model — reranking will be skipped.", exc_info=True)
                self._reranker = None

    # ------------------------------------------------------------------
    # Individual retrieval backends
    # ------------------------------------------------------------------

    def _vector_search(self, query: str, n: int = 20) -> list[dict]:
        """Retrieve top-n results from ChromaDB (semantic search).

        Returns list of {"text", "page", "element_ids", "section", "rank"}.
        """
        collection = self.indexer.get_collection()
        if collection is None or collection.count() == 0:
            return []

        try:
            results = collection.query(query_texts=[query], n_results=min(n, collection.count()))
        except Exception:
            logger.warning("ChromaDB query failed.", exc_info=True)
            return []

        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]

        out: list[dict] = []
        for rank, (text, meta) in enumerate(zip(docs, metas)):
            out.append(
                {
                    "text": text,
                    "page": meta.get("page", 0),
                    "element_ids": [eid for eid in meta.get("element_ids", "").split(",") if eid],
                    "section": meta.get("section", ""),
                    "rank": rank,
                }
            )
        return out

    def _bm25_search(self, query: str, n: int = 20) -> list[dict]:
        """Retrieve top-n results from BM25S (keyword search).

        Returns list of {"text", "page", "element_ids", "section", "rank"}.
        """
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
            logger.warning("BM25 retrieval failed.", exc_info=True)
            return []

        out: list[dict] = []
        # results shape: (1, k) — indices into the corpus
        indices = results[0]
        for rank, idx in enumerate(indices):
            idx = int(idx)
            if idx < 0 or idx >= len(all_docs):
                continue
            meta = all_meta[idx] if idx < len(all_meta) else {}
            out.append(
                {
                    "text": all_docs[idx],
                    "page": meta.get("page", 0),
                    "element_ids": [eid for eid in meta.get("element_ids", "").split(",") if eid],
                    "section": meta.get("section", ""),
                    "rank": rank,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _rrf_merge(self, vector_results: list[dict], bm25_results: list[dict]) -> list[dict]:
        """Merge two ranked lists using Reciprocal Rank Fusion.

        score(d) = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        k = self.RRF_K
        # Use text as the dedup key
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
        """Rerank candidates with a cross-encoder.

        Falls back to RRF ordering if the reranker is unavailable.
        """
        self._load_reranker()
        if self._reranker is None or not candidates:
            return candidates[:top_k]

        try:
            pairs = [(query, doc["text"]) for doc in candidates]
            scores = self._reranker.predict(pairs)

            for doc, score in zip(candidates, scores):
                doc["rerank_score"] = float(score)

            reranked = sorted(candidates, key=lambda d: d["rerank_score"], reverse=True)
            return reranked[:top_k]
        except Exception:
            logger.warning("Reranking failed — returning RRF order.", exc_info=True)
            return candidates[:top_k]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve top-k relevant passages using hybrid search.

        Returns:
            list of dicts with keys:
                text (str), score (float), page (int), element_ids (list[str])
        """
        # 1. Semantic search
        vector_results = self._vector_search(query, n=20)

        # 2. Keyword search
        bm25_results = self._bm25_search(query, n=20)

        # Handle degenerate cases
        if not vector_results and not bm25_results:
            logger.warning("Both retrieval backends returned empty results for query.")
            return []

        # 3. Reciprocal Rank Fusion
        merged = self._rrf_merge(vector_results, bm25_results)

        # 4. Cross-encoder reranking
        reranked = self._rerank(query, merged, top_k)

        # 5. Normalise output format
        output: list[dict] = []
        for doc in reranked:
            score = doc.get("rerank_score", doc.get("rrf_score", 0.0))
            output.append(
                {
                    "text": doc["text"],
                    "score": score,
                    "page": doc.get("page", 0),
                    "element_ids": doc.get("element_ids", []),
                }
            )

        return output
