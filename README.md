# 🛡️ NexRAG Enterprise: Production Agentic RAG Platform

> **An Enterprise-Grade Agentic Retrieval-Augmented Generation (RAG) System featuring Hybrid Search (Dense + BM25), Cross-Encoder Reranking, Adversarial Guardrails, and Automated Ragas-Style Evaluation.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📌 Executive Summary & Architecture

Most naive RAG implementations suffer from **hallucinations, vocabulary mismatch, prompt injections, and lack of quantitative evaluation**. **NexRAG Enterprise** is built to solve these enterprise production challenges:

```
                                      [User Query]
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🛡️ Step 1: Guardrails & PII Scrubber  │
                       │  - Prompt Injection Scanner (Jailbreak)│
                       │  - PII Masking (Email, Card, SSN, Key) │
                       └───────────────────┬────────────────────┘
                                           │ Sanitized Query
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🧭 Step 2: Agent Intent Router        │
                       │  - Direct Answer / Math / RAG Pipeline │
                       └───────────────────┬────────────────────┘
                                           │ ENTERPRISE_RAG
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🔄 Step 3: Query Expansion (HyDE)     │
                       │  - Synonym mapping & Target embeddings │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🔍 Step 4: Hybrid Retrieval (RRF)     │
                       │  ├─ BM25 Okapi (Sparse Lexical, w=0.4) │
                       │  └─ Dense Vector Store (Cosine, w=0.6) │
                       └───────────────────┬────────────────────┘
                                           │ Top-2k Candidates
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🎯 Step 5: Cross-Encoder Reranker     │
                       │  - Cross-attention precision ranking   │
                       └───────────────────┬────────────────────┘
                                           │ Top-k Verified Context
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🪞 Step 6: Self-Reflection Critique   │
                       │  - Evidence sufficiency verification   │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  🧠 Step 7: Grounded LLM Synthesis     │
                       │  - OpenAI / Groq / Ollama / SmartMock  │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                       ┌────────────────────────────────────────┐
                       │  ⚖️ Step 8: Faithfulness Verification  │
                       │  - Hallucination detector & citations  │
                       └───────────────────┬────────────────────┘
                                           │
                                           ▼
                                 [Cited Verified Answer]
```

---


---

## 🚀 Key Technical Features

### 1. Hybrid Search via Reciprocal Rank Fusion (RRF)
Combines keyword-exact sparse matching (Okapi BM25, $k_1=1.5, b=0.75$) with dense semantic embeddings using normalized rank reciprocals:
$$RRF(d) = \sum_{m \in M} \frac{w_m}{k + rank_m(d)}$$

### 2. Cross-Encoder Precision Reranker
Evaluates cross-attention token alignment between the query and candidate passages, filtering out false-positive vector neighbors before prompt injection.

### 3. Comprehensive Guardrail Suite
* **Prompt Injection Defense**: Detects instruction overrides, simulated system roles, and delimiter attacks.
* **PII Redaction**: Automatically scrubs emails, credit cards, SSNs, and API keys before query processing.
* **Faithfulness Verification**: Extracts factual propositions from generated answers and verifies grounding against source context chunks.

### 4. Zero-Cost / Offline Ready
Ships with a built-in `SmartOfflineLLM` engine and local embedding fallback, allowing immediate testing, demonstration, and benchmarking without incurring third-party API costs or requiring internet access. Seamlessly connects to OpenAI (`gpt-4o-mini`), Groq (`llama-3.1-70b`), or local Ollama when keys are configured.

### 5. Automated Evaluation Benchmark (Ragas KPIs)
Includes a CLI evaluation harness that runs over a curated ground truth benchmark and computes:
* **Faithfulness (Grounding)**
* **Answer Relevance**
* **Context Precision**
* **Context Recall**
* **End-to-End Latency**

---

## ⚡ Quickstart Guide

### Option 1: Local Setup (Recommended)

1. **Clone & enter project directory:**
   ```bash
   cd enterprise-agentic-rag
   ```

2. **Install dependencies (or use existing Python environment):**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Unit & Integration Tests:**
   ```bash
   python run_tests.py
   ```

4. **Run the Automated Benchmark:**
   ```bash
   python benchmark_runner.py
   ```

5. **Launch Interactive Streamlit Dashboard:**
   ```bash
   python run_ui.py
   # Opens at http://localhost:8501
   ```

6. **Launch FastAPI REST Backend:**
   ```bash
   python run_server.py
   # API: http://localhost:8000
   # Swagger Docs: http://localhost:8000/docs
   ```

---

### Option 2: Run with Docker Compose

```bash
docker compose up --build
```
* **API Documentation**: `http://localhost:8000/docs`
* **Interactive UI**: `http://localhost:8501`

---

## 🌐 API Reference

### `POST /api/v1/query`
Executes full agentic pipeline:
```json
{
  "query": "What is the mandatory session timeout policy?",
  "top_k": 3,
  "enable_hybrid": true,
  "enable_rerank": true,
  "enable_guardrails": true
}
```
**Response:**
```json
{
  "query": "What is the mandatory session timeout policy?",
  "answer": "Web console sessions automatically terminate after 15 minutes of inactivity.",
  "routed_intent": "ENTERPRISE_RAG",
  "confidence_score": 0.88,
  "sources": [
    {
      "source": "cloud_security_policy.md",
      "page": 1,
      "rerank_score": 0.884,
      "excerpt": "Web console sessions automatically terminate after 15 minutes of inactivity..."
    }
  ],
  "guardrail_report": {
    "is_safe": true,
    "faithfulness_score": 1.0,
    "faithfulness_passed": true
  },
  "total_latency_ms": 3.8
}
```

---

## 📂 Project Directory Structure

```text
enterprise-agentic-rag/
├── README.md                      # Executive documentation & resume guide
├── requirements.txt               # Dependencies
├── pyproject.toml                 # Packaging spec
├── .env.example                   # Environment configuration
├── Dockerfile                     # Production container image
├── docker-compose.yml             # Service orchestration
├── run_server.py                  # Single-command API launcher
├── run_ui.py                      # Single-command Dashboard launcher
├── run_tests.py                   # Automated unit & integration tests
├── benchmark_runner.py            # Evaluation CLI runner
├── benchmark_report.md            # Generated benchmark evaluation report
├── data/
│   ├── sample_docs/               # Enterprise policy & financial docs
│   └── evaluation/                # Ground truth test dataset
├── src/
│   ├── config.py                  # Settings & environment variables
│   ├── models/schemas.py          # Pydantic schemas
│   ├── ingestion/                 # Document loaders & recursive splitter
│   ├── retrieval/                 # BM25, Dense Vector, Hybrid RRF, Reranker
│   ├── agent/                     # Router, HyDE expansion, LLM client, Workflow
│   ├── guardrails/                # Injection scanner, PII redactor, Faithfulness
│   ├── evaluation/                # Metrics calculation & benchmark harness
│   └── api/app.py                 # FastAPI application
├── frontend/dashboard.py          # Streamlit observability UI
└── tests/                         # Test suite
```

---

## 📄 License
MIT License. Free for academic, personal, and commercial portfolio use.
