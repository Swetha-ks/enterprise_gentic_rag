"""
Hybrid Retriever combining BM25 and Dense Vector Search via Reciprocal Rank Fusion (RRF).
Solves vocabulary mismatch and semantic drift in enterprise retrieval.
"""

from typing import List, Dict, Any, Tuple
from ..models.schemas import DocumentChunk, RetrievalResult
from .bm25_retriever import BM25Retriever
from .vector_retriever import DenseVectorRetriever

class HybridRetriever:
    def __init__(
        self,
        bm25_retriever: BM25Retriever,
        vector_retriever: DenseVectorRetriever,
        rrf_k: int = 60,
        sparse_weight: float = 0.4,
        dense_weight: float = 0.6
    ):
        self.bm25 = bm25_retriever
        self.vector = vector_retriever
        self.rrf_k = rrf_k
        self.sparse_weight = sparse_weight
        self.dense_weight = dense_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        enable_hybrid: bool = True
    ) -> List[RetrievalResult]:
        """
        Executes hybrid retrieval using Reciprocal Rank Fusion (RRF).
        """
        # 1. Fetch dense results
        dense_results = self.vector.query(query, top_k=top_k * 2)

        # 2. If hybrid disabled, return pure dense results
        if not enable_hybrid:
            results = []
            for chunk, score in dense_results[:top_k]:
                results.append(RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    dense_score=round(score, 4),
                    rrf_score=round(score, 4)
                ))
            return results

        # 3. Fetch sparse (BM25) results
        bm25_results = self.bm25.query(query, top_k=top_k * 2)

        # 4. Compute RRF Scores
        # Map: chunk_id -> dict with scores and chunk obj
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # Process BM25 ranks
        for rank, (chunk, score) in enumerate(bm25_results):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": chunk,
                    "sparse_score": score,
                    "dense_score": 0.0,
                    "rrf_score": 0.0
                }
            chunk_map[cid]["sparse_score"] = score
            # RRF formula: weight / (k + rank)
            chunk_map[cid]["rrf_score"] += self.sparse_weight / (self.rrf_k + rank + 1)

        # Process Vector ranks
        for rank, (chunk, score) in enumerate(dense_results):
            cid = chunk.chunk_id
            if cid not in chunk_map:
                chunk_map[cid] = {
                    "chunk": chunk,
                    "sparse_score": 0.0,
                    "dense_score": score,
                    "rrf_score": 0.0
                }
            chunk_map[cid]["dense_score"] = score
            chunk_map[cid]["rrf_score"] += self.dense_weight / (self.rrf_k + rank + 1)

        # Sort candidate chunks by RRF score
        sorted_candidates = sorted(
            chunk_map.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        final_results: List[RetrievalResult] = []
        for item in sorted_candidates[:top_k]:
            c: DocumentChunk = item["chunk"]
            final_results.append(RetrievalResult(
                chunk_id=c.chunk_id,
                text=c.text,
                metadata=c.metadata,
                sparse_score=round(item["sparse_score"], 4),
                dense_score=round(item["dense_score"], 4),
                rrf_score=round(item["rrf_score"], 5)
            ))

        return final_results
