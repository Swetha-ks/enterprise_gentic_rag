"""
Pydantic Schemas for Enterprise Agentic RAG.
Defines data structures for ingestion, retrieval, agent state, guardrails, and evaluation.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    token_count: int = 0

class RetrievalResult(BaseModel):
    chunk_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sparse_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

class GuardrailReport(BaseModel):
    is_safe: bool = True
    injection_detected: bool = False
    pii_detected: bool = False
    flagged_reasons: List[str] = Field(default_factory=list)
    sanitized_query: str = ""
    faithfulness_score: Optional[float] = None
    faithfulness_passed: bool = True

class AgentStep(BaseModel):
    step_name: str
    description: str
    status: str = "success"  # success, warning, failed
    details: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0

class QueryRequest(BaseModel):
    query: str
    top_k: int = 3
    enable_hybrid: bool = True
    enable_rerank: bool = True
    enable_guardrails: bool = True
    enable_reflection: bool = True

class QueryResponse(BaseModel):
    query: str
    answer: str
    routed_intent: str
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 1.0
    execution_steps: List[AgentStep] = Field(default_factory=list)
    guardrail_report: GuardrailReport
    total_latency_ms: float = 0.0

class IngestResponse(BaseModel):
    status: str
    filename: str
    chunks_indexed: int
    doc_id: str
    message: str

class EvaluationSample(BaseModel):
    id: str
    question: str
    ground_truth_context: str
    ground_truth_answer: str

class BenchmarkResult(BaseModel):
    total_samples: int
    avg_faithfulness: float
    avg_answer_relevance: float
    avg_context_precision: float
    avg_context_recall: float
    avg_latency_ms: float
    detailed_samples: List[Dict[str, Any]] = Field(default_factory=list)
