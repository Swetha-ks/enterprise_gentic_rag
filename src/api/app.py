"""
FastAPI REST API for Enterprise Agentic RAG System.
Provides production endpoints for querying, document ingestion, guardrails, and benchmark evaluation.
"""

import time
import shutil
from pathlib import Path
from typing import List, Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import settings, SAMPLE_DOCS_DIR, EVAL_DIR
from ..models.schemas import (
    QueryRequest, QueryResponse, IngestResponse, BenchmarkResult
)
from ..ingestion.document_loader import DocumentLoader
from ..ingestion.text_splitter import RecursiveTextSplitter
from ..retrieval.bm25_retriever import BM25Retriever
from ..retrieval.vector_retriever import DenseVectorRetriever
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from ..agent.workflow import AgenticRAGWorkflow
from ..evaluation.benchmark import BenchmarkRunner

app = FastAPI(
    title="Enterprise Agentic RAG API",
    description="Production-Grade Agentic Retrieval-Augmented Generation system with Hybrid Search, Guardrails, and Ragas Evaluation.",
    version="1.0.0"
)

# Enable Cross-Origin Resource Sharing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global State Singletons
bm25_retriever = BM25Retriever()
vector_retriever = DenseVectorRetriever(model_name=settings.embedding_model_name)
hybrid_retriever = HybridRetriever(
    bm25_retriever=bm25_retriever,
    vector_retriever=vector_retriever,
    rrf_k=settings.rrf_k,
    sparse_weight=settings.sparse_weight,
    dense_weight=settings.dense_weight
)
reranker = Reranker()
workflow = AgenticRAGWorkflow(
    hybrid_retriever=hybrid_retriever,
    reranker=reranker
)
text_splitter = RecursiveTextSplitter(
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap
)

# In-memory document tracking
indexed_documents: Dict[str, Any] = {}

def index_file_internal(file_path: Path) -> int:
    """Helper to parse, chunk, and index a document."""
    raw_docs = DocumentLoader.load_file(file_path)
    chunks = text_splitter.split_documents(raw_docs)
    
    # Append to existing chunks in retrievers
    all_chunks = list(bm25_retriever.chunks) + chunks
    bm25_retriever.index_chunks(all_chunks)
    vector_retriever.index_chunks(all_chunks)

    indexed_documents[file_path.name] = {
        "file_path": str(file_path),
        "chunk_count": len(chunks),
        "total_chars": sum(len(c.text) for c in chunks)
    }
    return len(chunks)

@app.on_event("startup")
async def startup_event():
    """Auto-indexes sample documents on startup if present."""
    if SAMPLE_DOCS_DIR.exists():
        for doc_file in SAMPLE_DOCS_DIR.glob("*.*"):
            if doc_file.suffix.lower() in [".md", ".pdf", ".txt", ".json"]:
                try:
                    index_file_internal(doc_file)
                except Exception as e:
                    print(f"[Startup Indexing] Error loading {doc_file.name}: {e}")

@app.get("/api/v1/health")
def health_check():
    return {
        "status": "healthy",
        "service": "enterprise-agentic-rag",
        "llm_provider": settings.llm_provider,
        "indexed_documents": len(indexed_documents),
        "total_chunks": len(bm25_retriever.chunks)
    }

@app.post("/api/v1/query", response_model=QueryResponse)
def query_agent(request: QueryRequest):
    """
    Primary endpoint: Routes query through safety guardrails, intent classifier,
    hybrid retrieval, precision reranking, self-reflection, and grounded generation.
    """
    try:
        response = workflow.execute(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query execution failed: {str(e)}")

@app.post("/api/v1/documents/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    """Uploads a PDF, Markdown, or Text file and indexes it immediately."""
    upload_dir = SAMPLE_DOCS_DIR
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest_path = upload_dir / file.filename

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        chunks_count = index_file_internal(dest_path)
        return IngestResponse(
            status="success",
            filename=file.filename,
            chunks_indexed=chunks_count,
            doc_id=file.filename,
            message=f"Successfully indexed {chunks_count} chunks into hybrid vector + BM25 stores."
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process document: {str(e)}")

@app.get("/api/v1/documents/stats")
def get_document_stats():
    return {
        "total_documents": len(indexed_documents),
        "total_chunks": len(bm25_retriever.chunks),
        "documents": indexed_documents
    }

@app.get("/api/v1/evaluate", response_model=BenchmarkResult)
def run_evaluation():
    """Runs automated benchmark evaluation on curated ground truth dataset."""
    eval_file = EVAL_DIR / "ground_truth_dataset.json"
    if not eval_file.exists():
        raise HTTPException(status_code=404, detail="Ground truth evaluation dataset not found.")

    runner = BenchmarkRunner(workflow)
    result = runner.run_benchmark(eval_file)
    return result
