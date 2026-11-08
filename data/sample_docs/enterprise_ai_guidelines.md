# Enterprise AI Deployment & Governance Guidelines (AI-GOV-2026)

## 1. Acceptable Use of Generative AI
- Employees may utilize authorized LLM endpoints for productivity, documentation drafting, and code optimization, provided that all company proprietary code and customer data are processed through privacy-preserving enterprise accounts with zero data retention for model training.
- **Prohibited Data Inputs**: Submitting unredacted customer Personally Identifiable Information (PII), secret cryptographic keys, raw database connection strings, or non-public financial reports to external unvetted AI models is strictly forbidden and constitutes a P0 security infraction.

## 2. Model Tiering and Production Guardrails
- **Tier 1 (Internal Developer & Advisory Agents)**:
  - Permitted model types: Open-weights SLMs (Llama-3.1 8B, Phi-3, Mistral 7B) hosted in internal VPCs or commercial APIs with enterprise confidentiality SLAs.
  - Required Guardrails: Prompt injection scanner and automated PII masking.
- **Tier 2 (Customer-Facing AI & Autonomous Agents)**:
  - Hallucination Threshold: Must pass automated Faithfulness evaluation with a score of $\ge 0.85$ on the standardized corporate benchmark.
  - Human-in-the-loop (HITL): High-impact actions such as account deletion, billing refunds over $500, or contract modifications require explicit human sign-off.
  - Grounding Mandate: All answers provided by customer-facing agents must carry verifiable citation metadata pointing to indexed corporate knowledge bases.

## 3. Automated Model Evaluation & Monitoring
- RAG pipelines must be evaluated continuously using four canonical metrics: Faithfulness, Answer Relevance, Context Precision, and Context Recall.
- Any model update or prompt modification resulting in a drop greater than **3%** in Context Recall or Faithfulness will trigger an automated deployment rollback.
