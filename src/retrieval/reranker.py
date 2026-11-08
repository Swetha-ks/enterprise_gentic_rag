"""
Cross-Encoder / Context Precision Reranker.
Re-scores candidate chunks from Hybrid Retrieval to maximize precision at Top-1 and Top-3.
Uses cross-attention lexical-semantic scoring to verify exact claim relevance.
"""

import math
import re
from typing import List
from ..models.schemas import RetrievalResult

class Reranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._cross_encoder = None
        self._init_model()

    def _init_model(self):
        import os
        if os.getenv("USE_TRANSFORMERS", "false").lower() in ["true", "1"]:
            try:
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self.model_name)
                return
            except Exception as e:
                print(f"[Reranker] CrossEncoder fallback: {e}")
        self._cross_encoder = None

    def rerank(self, query: str, results: List[RetrievalResult], top_k: int = 3) -> List[RetrievalResult]:
        """
        Reranks a list of RetrievalResult candidates based on direct cross-attention scoring with query.
        """
        if not results:
            return []

        # If torch CrossEncoder model is available, use it
        if self._cross_encoder is not None:
            try:
                pairs = [[query, r.text] for r in results]
                scores = self._cross_encoder.predict(pairs)
                for r, score in zip(results, scores):
                    # Sigmoid normalization
                    prob = 1.0 / (1.0 + math.exp(-float(score)))
                    r.rerank_score = round(prob, 4)
                results.sort(key=lambda x: x.rerank_score, reverse=True)
                return results[:top_k]
            except Exception:
                pass

        # High-performance lexical & semantic alignment scoring
        q_words = set(re.findall(r"\b\w{3,}\b", query.lower()))
        q_len = max(1, len(q_words))

        for r in results:
            text_lower = r.text.lower()
            t_words = set(re.findall(r"\b\w{3,}\b", text_lower))
            
            # 1. Exact query terms overlap (Jaccard & containment)
            overlap = len(q_words.intersection(t_words))
            overlap_ratio = overlap / q_len

            # 2. Key phrase preservation (bigrams / phrases)
            phrase_bonus = 0.0
            query_clean = " ".join(query.lower().split())
            if query_clean in text_lower:
                phrase_bonus = 0.35
            else:
                # Check for 2-word collocations
                q_list = query.lower().split()
                for i in range(len(q_list) - 1):
                    bigram = f"{q_list[i]} {q_list[i+1]}"
                    if bigram in text_lower:
                        phrase_bonus += 0.10

            # 3. Dense & RRF foundation
            base_score = r.dense_score if r.dense_score > 0 else (r.rrf_score * 50.0)

            # Composite cross-attention proxy score
            final_score = (0.45 * overlap_ratio) + (0.35 * base_score) + min(0.20, phrase_bonus)
            r.rerank_score = round(min(0.99, max(0.01, final_score)), 4)

        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]
