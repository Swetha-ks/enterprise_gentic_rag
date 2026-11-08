"""
Integration Tests for Master Agentic RAG Workflow.
"""

from src.models.schemas import QueryRequest, DocumentChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import DenseVectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.agent.workflow import AgenticRAGWorkflow

def test_full_agent_workflow():
    bm25 = BM25Retriever()
    dense = DenseVectorRetriever()
    chunks = [
        DocumentChunk(
            chunk_id="chunk_sec_1",
            doc_id="sec_policy",
            text="Severity 0 (P0) incidents require customer notification within 1 hour.",
            metadata={"source": "cloud_security_policy.md", "page": 1}
        )
    ]
    bm25.index_chunks(chunks)
    dense.index_chunks(chunks)

    hybrid = HybridRetriever(bm25, dense)
    reranker = Reranker()
    workflow = AgenticRAGWorkflow(hybrid, reranker)

    req = QueryRequest(
        query="What is the customer notification SLA for a P0 critical incident?",
        top_k=1,
        enable_hybrid=True,
        enable_rerank=True,
        enable_guardrails=True
    )

    response = workflow.execute(req)
    assert response.routed_intent == "ENTERPRISE_RAG"
    assert len(response.sources) >= 1
    assert "1 hour" in response.answer or "P0" in response.answer
    assert response.guardrail_report.is_safe is True
    assert len(response.execution_steps) >= 5
