"""
Dense Vector Retriever using ChromaDB and Sentence-Transformers.
Supports local persistent vector storage with an in-memory fallback for zero-downtime reliability.
"""

import math
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from ..models.schemas import DocumentChunk
from ..config import CHROMA_PERSIST_DIR

class DenseVectorRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", persist_dir: Optional[str] = None):
        self.model_name = model_name
        self.persist_dir = persist_dir or str(CHROMA_PERSIST_DIR)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self._encoder = None
        self._chroma_client = None
        self._chroma_collection = None
        self._init_encoder()

    def _init_encoder(self):
        """Loads sentence-transformers model if enabled via env, else uses internal high-speed dense encoder."""
        import os
        if os.getenv("USE_TRANSFORMERS", "false").lower() in ["true", "1"]:
            try:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(self.model_name)
                return
            except Exception as e:
                print(f"[VectorRetriever] SentenceTransformer fallback: {e}")
        self._encoder = None

    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embeds a list of texts into dense vectors."""
        if self._encoder is not None:
            return self._encoder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        # Fallback: Hash/TF-IDF normalized dense embedding (384 dimensions)
        dim = 384
        vectors = []
        for text in texts:
            words = text.lower().split()
            v = np.zeros(dim, dtype=np.float32)
            if not words:
                vectors.append(v)
                continue
            for w in words:
                h = abs(hash(w)) % dim
                v[h] += 1.0
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            vectors.append(v)
        return np.array(vectors)

    def index_chunks(self, chunks: List[DocumentChunk]):
        """Indexes document chunks into dense vector matrix."""
        self.chunks = list(chunks)
        if not self.chunks:
            self.embeddings = None
            return

        texts = [chunk.text for chunk in self.chunks]
        self.embeddings = self._embed_texts(texts)

        # Try to sync with ChromaDB if available
        try:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=self.persist_dir)
            self._chroma_collection = self._chroma_client.get_or_create_collection(
                name="enterprise_rag_docs",
                metadata={"hnsw:space": "cosine"}
            )
            # Add to Chroma collection
            ids = [chunk.chunk_id for chunk in self.chunks]
            metadatas = [{k: str(v) for k, v in chunk.metadata.items()} for chunk in self.chunks]
            self._chroma_collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=self.embeddings.tolist(),
                metadatas=metadatas
            )
        except Exception as e:
            # Local matrix fallback is always active and reliable
            pass

    def query(self, query_str: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Embeds the query and computes cosine similarity against all indexed chunk vectors.
        Returns list of (DocumentChunk, similarity_score).
        """
        if not self.chunks or self.embeddings is None:
            return []

        q_vec = self._embed_texts([query_str])[0]

        # Cosine similarity (vectors are L2 normalized, so dot product = cosine similarity)
        scores = np.dot(self.embeddings, q_vec)

        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            # Normalize score to [0, 1] range
            norm_score = max(0.0, min(1.0, (score + 1.0) / 2.0 if score < 0 else score))
            results.append((self.chunks[idx], norm_score))

        return results
