"""
Unit Tests for Retrieval Modules (BM25, Dense Vector, Hybrid RRF, Reranker).
"""

from src.models.schemas import DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import DenseVectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker

def test_bm25_retrieval():
    retriever = BM25Retriever()
    chunks = [
        DocumentChunk(chunk_id="c1", doc_id="d1", text="Multi-factor authentication MFA is required for all admins."),
        DocumentChunk(chunk_id="c2", doc_id="d2", text="Our company revenue grew 34% year over year in Q3."),
        DocumentChunk(chunk_id="c3", doc_id="d3", text="The cafeteria serves lunch between 12 PM and 2 PM.")
    ]
    retriever.index_chunks(chunks)
    
    results = retriever.query("MFA authentication policy", top_k=1)
    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
    assert results[0][1] > 0

def test_hybrid_rrf_scoring():
    bm25 = BM25Retriever()
    dense = DenseVectorRetriever()
    hybrid = HybridRetriever(bm25, dense)

    chunks = [
        DocumentChunk(chunk_id="c1", doc_id="d1", text="Cloud security protocols mandate AES-256 encryption at rest."),
        DocumentChunk(chunk_id="c2", doc_id="d2", text="Employees receive 20 days of paid time off per calendar year.")
    ]
    bm25.index_chunks(chunks)
    dense.index_chunks(chunks)

    results = hybrid.retrieve("encryption at rest protocols", top_k=2)
    assert len(results) >= 1
    assert results[0].chunk_id == "c1"
    assert results[0].rrf_score > 0

def test_reranker_reordering():
    reranker = Reranker()
    from src.models.schemas import RetrievalResult
    results = [
        RetrievalResult(chunk_id="c1", text="General discussion about security and compliance.", dense_score=0.4),
        RetrievalResult(chunk_id="c2", text="Mandatory AES-256 encryption is strictly enforced for all cloud databases.", dense_score=0.35)
    ]
    reranked = reranker.rerank("What encryption is enforced for cloud databases?", results, top_k=2)
    assert len(reranked) == 2
    assert reranked[0].chunk_id == "c2"
