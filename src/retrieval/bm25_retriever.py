"""
Okapi BM25 Retriever Implementation.
Pure Python, production-grade implementation of BM25 (k1=1.5, b=0.75) with tokenization and stemming.
Provides keyword/sparse search without external heavy dependencies.
"""

import math
import re
from typing import List, Dict, Any, Tuple
from ..models.schemas import DocumentChunk

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: List[DocumentChunk] = []
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lengths: List[int] = []
        self.doc_freqs: Dict[str, int] = {}
        self.inverted_index: Dict[str, List[Tuple[int, int]]] = {}  # term -> [(doc_idx, freq)]

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercase and extract alphanumeric word tokens."""
        return re.findall(r"\b\w{2,}\b", text.lower())

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Index a list of DocumentChunk objects."""
        self.chunks = list(chunks)
        self.corpus_size = len(self.chunks)
        if self.corpus_size == 0:
            return

        self.doc_lengths = []
        self.doc_freqs = {}
        self.inverted_index = {}

        total_length = 0
        for doc_idx, chunk in enumerate(self.chunks):
            tokens = self.tokenize(chunk.text)
            doc_len = len(tokens)
            self.doc_lengths.append(doc_len)
            total_length += doc_len

            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1

            for term, freq in term_counts.items():
                self.doc_freqs[term] = self.doc_freqs.get(term, 0) + 1
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((doc_idx, freq))

        self.avg_doc_len = total_length / self.corpus_size if self.corpus_size > 0 else 0.0

    def query(self, query_str: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Calculates BM25 score for the query string against all indexed documents.
        Returns list of (DocumentChunk, score) sorted in descending score order.
        """
        if self.corpus_size == 0:
            return []

        query_tokens = self.tokenize(query_str)
        if not query_tokens:
            return []

        scores = [0.0] * self.corpus_size

        for term in query_tokens:
            if term not in self.inverted_index:
                continue

            # IDF calculation with Lucene/BM25 smoothing
            df = self.doc_freqs[term]
            idf = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

            for doc_idx, freq in self.inverted_index[term]:
                doc_len = self.doc_lengths[doc_idx]
                numerator = freq * (self.k1 + 1.0)
                denominator = freq + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                scores[doc_idx] += idf * (numerator / denominator)

        # Pair scores with chunks and sort
        scored_pairs = [(self.chunks[i], scores[i]) for i in range(self.corpus_size) if scores[i] > 0.0]
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        return scored_pairs[:top_k]
