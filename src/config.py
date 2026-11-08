"""
Configuration management for Enterprise Agentic RAG System.
Handles environment variables, model choices, retrieval parameters, and guardrail rules.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DOCS_DIR = DATA_DIR / "sample_docs"
EVAL_DIR = DATA_DIR / "evaluation"
CHROMA_PERSIST_DIR = BASE_DIR / "storage" / "chroma"

class SystemSettings(BaseModel):
    # LLM Settings
    llm_provider: Literal["mock", "openai", "groq", "ollama"] = Field(
        default=os.getenv("LLM_PROVIDER", "mock")
    )
    openai_api_key: str = Field(default=os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = Field(default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    groq_api_key: str = Field(default=os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default=os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile"))
    ollama_base_url: str = Field(default=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = Field(default=os.getenv("OLLAMA_MODEL", "llama3.1"))
    
    # Embedding & Retrieval Settings
    embedding_model_name: str = Field(default="all-MiniLM-L6-v2")
    chunk_size: int = Field(default=600)
    chunk_overlap: int = Field(default=120)
    retrieval_top_k: int = Field(default=5)
    rerank_top_k: int = Field(default=3)
    rrf_k: int = Field(default=60) # Reciprocal Rank Fusion constant
    sparse_weight: float = Field(default=0.4)
    dense_weight: float = Field(default=0.6)
    
    # Guardrails Settings
    enable_guardrails: bool = Field(default=True)
    pii_redaction_enabled: bool = Field(default=True)
    injection_threshold: float = Field(default=0.65)
    faithfulness_threshold: float = Field(default=0.70)
    
    # Agent Settings
    max_reflection_steps: int = Field(default=2)
    similarity_confidence_threshold: float = Field(default=0.45)
    
    class Config:
        arbitrary_types_allowed = True

settings = SystemSettings()
