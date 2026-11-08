"""
Master Agentic RAG Workflow.
Implements the full agent state machine:
1. Input Guardrails & PII Scrubber
2. Intent Classification & Routing
3. Query Expansion & Decomposition (HyDE)
4. Dense + Sparse Hybrid Retrieval
5. Precision Cross-Encoder Reranker
6. Self-Reflection & Context Sufficiency Critique
7. Grounded LLM Answer Synthesis
8. Output Guardrails & Faithfulness Verification
"""

import time
from typing import List, Dict, Any, Optional
from ..models.schemas import (
    QueryRequest, QueryResponse, AgentStep, GuardrailReport, RetrievalResult
)
from ..guardrails.injection_detector import InjectionDetector
from ..guardrails.pii_masker import PIIMasker
from ..guardrails.faithfulness_check import FaithfulnessChecker
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.reranker import Reranker
from .router import AgentRouter
from .query_rewriter import QueryRewriter
from .llm_client import UnifiedLLMClient

class AgenticRAGWorkflow:
    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: Reranker,
        llm_client: Optional[UnifiedLLMClient] = None
    ):
        self.retriever = hybrid_retriever
        self.reranker = reranker
        self.llm = llm_client or UnifiedLLMClient()
        self.injection_detector = InjectionDetector()
        self.pii_masker = PIIMasker()
        self.faithfulness_checker = FaithfulnessChecker()
        self.router = AgentRouter()

    def execute(self, request: QueryRequest) -> QueryResponse:
        start_time = time.time()
        steps: List[AgentStep] = []
        raw_query = request.query
        
        # ----------------------------------------------------
        # Step 1: Input Guardrails & PII Sanitization
        # ----------------------------------------------------
        t0 = time.time()
        is_injection, risk_score, matched_injections = self.injection_detector.inspect(raw_query)
        sanitized_query, pii_found, detected_pii = self.pii_masker.mask(raw_query)
        
        guardrail_report = GuardrailReport(
            is_safe=not is_injection,
            injection_detected=is_injection,
            pii_detected=pii_found,
            flagged_reasons=matched_injections + detected_pii,
            sanitized_query=sanitized_query
        )

        steps.append(AgentStep(
            step_name="Safety & PII Guardrails",
            description="Scanned query for prompt injection and redacted PII entities.",
            status="failed" if is_injection else ("warning" if pii_found else "success"),
            details={
                "injection_risk_score": risk_score,
                "injection_flags": matched_injections,
                "pii_flags": detected_pii,
                "sanitized_input": sanitized_query
            },
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # Early termination if malicious prompt injection detected
        if is_injection and request.enable_guardrails:
            return QueryResponse(
                query=raw_query,
                answer=(
                    "⚠️ **Security Alert**: The query was blocked by Enterprise Guardrails due to "
                    f"detected adversarial prompt injection patterns: {', '.join(matched_injections)}."
                ),
                routed_intent="SECURITY_VIOLATION",
                sources=[],
                confidence_score=0.0,
                execution_steps=steps,
                guardrail_report=guardrail_report,
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        active_query = sanitized_query if request.enable_guardrails else raw_query

        # ----------------------------------------------------
        # Step 2: Agent Intent Classification & Routing
        # ----------------------------------------------------
        t0 = time.time()
        route, route_rationale = self.router.route(active_query, is_safe=not is_injection)
        steps.append(AgentStep(
            step_name="Intent Routing",
            description=f"Classified query into route: {route}",
            status="success",
            details={"intent": route, "rationale": route_rationale},
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # Handle non-RAG routes immediately
        if route == "DIRECT_CONVERSATION":
            answer = self.llm.complete(
                prompt=f"Respond courteously as an enterprise AI assistant: {active_query}",
                system_prompt="You are an enterprise AI assistant."
            )
            return QueryResponse(
                query=raw_query,
                answer=answer,
                routed_intent=route,
                sources=[],
                confidence_score=0.95,
                execution_steps=steps,
                guardrail_report=guardrail_report,
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        elif route == "STRUCTURED_CALCULATION":
            answer = self.llm.complete(
                prompt=f"Perform the requested mathematical/logical calculation accurately: {active_query}",
                system_prompt="You are a precise calculation tool."
            )
            return QueryResponse(
                query=raw_query,
                answer=answer,
                routed_intent=route,
                sources=[],
                confidence_score=0.98,
                execution_steps=steps,
                guardrail_report=guardrail_report,
                total_latency_ms=round((time.time() - start_time) * 1000, 2)
            )

        # ----------------------------------------------------
        # Step 3: Query Reformulation & Expansion (HyDE)
        # ----------------------------------------------------
        t0 = time.time()
        expanded_queries = QueryRewriter.expand_query(active_query)
        steps.append(AgentStep(
            step_name="Query Expansion",
            description=f"Generated {len(expanded_queries)} search variants using lexical mapping and HyDE.",
            status="success",
            details={"expanded_variants": expanded_queries},
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # ----------------------------------------------------
        # Step 4: Dense + Sparse Hybrid Retrieval (RRF)
        # ----------------------------------------------------
        t0 = time.time()
        retrieved_results: List[RetrievalResult] = []
        for q_variant in expanded_queries:
            variant_results = self.retriever.retrieve(
                query=q_variant,
                top_k=request.top_k * 2,
                enable_hybrid=request.enable_hybrid
            )
            retrieved_results.extend(variant_results)

        # Deduplicate results by chunk_id
        seen_ids = set()
        unique_results: List[RetrievalResult] = []
        for r in retrieved_results:
            if r.chunk_id not in seen_ids:
                seen_ids.add(r.chunk_id)
                unique_results.append(r)

        steps.append(AgentStep(
            step_name="Hybrid Retrieval (BM25 + Dense)",
            description=f"Retrieved {len(unique_results)} candidate chunks via Reciprocal Rank Fusion.",
            status="success",
            details={
                "candidates_retrieved": len(unique_results),
                "top_candidate_scores": [
                    {"id": r.chunk_id, "rrf": r.rrf_score, "dense": r.dense_score, "sparse": r.sparse_score}
                    for r in unique_results[:3]
                ]
            },
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # ----------------------------------------------------
        # Step 5: Cross-Encoder Precision Reranking
        # ----------------------------------------------------
        t0 = time.time()
        if request.enable_rerank and unique_results:
            final_chunks = self.reranker.rerank(
                query=active_query,
                results=unique_results,
                top_k=request.top_k
            )
        else:
            final_chunks = unique_results[:request.top_k]

        steps.append(AgentStep(
            step_name="Cross-Encoder Reranking",
            description=f"Reranked top {len(final_chunks)} chunks for cross-attention alignment.",
            status="success",
            details={
                "reranked_chunks": [
                    {"id": c.chunk_id, "score": c.rerank_score, "source": c.metadata.get("source")}
                    for c in final_chunks
                ]
            },
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # ----------------------------------------------------
        # Step 6: Self-Reflection & Context Sufficiency Critique
        # ----------------------------------------------------
        t0 = time.time()
        top_confidence = final_chunks[0].rerank_score if final_chunks else 0.0
        context_sufficient = top_confidence >= 0.25 and len(final_chunks) > 0

        steps.append(AgentStep(
            step_name="Context Sufficiency Critique",
            description=f"Agent evaluated evidence confidence ({top_confidence:.2f}): {'SUFFICIENT' if context_sufficient else 'LOW_CONFIDENCE'}",
            status="success" if context_sufficient else "warning",
            details={"top_confidence": top_confidence, "is_sufficient": context_sufficient},
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # ----------------------------------------------------
        # Step 7: Grounded LLM Answer Synthesis
        # ----------------------------------------------------
        t0 = time.time()
        context_texts = [c.text for c in final_chunks]
        system_prompt = (
            "You are an enterprise AI assistant. Answer the user's question accurately using ONLY "
            "the provided context chunks. If the answer is not present in the context, explicitly state "
            "that the documentation does not contain this information. Always cite source references."
        )
        formatted_context = "\n\n---\n\n".join([
            f"[Source {i+1}: {c.metadata.get('source', 'doc')} (Page {c.metadata.get('page', 1)})]\n{c.text}"
            for i, c in enumerate(final_chunks)
        ])
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {active_query}\n\nAnswer:"

        answer = self.llm.complete(
            prompt=user_prompt,
            system_prompt=system_prompt,
            context_chunks=context_texts
        )

        steps.append(AgentStep(
            step_name="Answer Synthesis",
            description="Generated context-grounded response with explicit source citations.",
            status="success",
            details={"answer_length_chars": len(answer)},
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # ----------------------------------------------------
        # Step 8: Faithfulness Verification (Hallucination Check)
        # ----------------------------------------------------
        t0 = time.time()
        faith_score, faith_passed, ungrounded_claims = self.faithfulness_checker.evaluate(
            answer=answer,
            context_chunks=context_texts
        )
        guardrail_report.faithfulness_score = faith_score
        guardrail_report.faithfulness_passed = faith_passed

        steps.append(AgentStep(
            step_name="Faithfulness Verification",
            description=f"Calculated answer faithfulness: {faith_score:.2f} ({'PASSED' if faith_passed else 'FLAGGED'})",
            status="success" if faith_passed else "warning",
            details={
                "faithfulness_score": faith_score,
                "passed": faith_passed,
                "ungrounded_claims": ungrounded_claims
            },
            duration_ms=round((time.time() - t0) * 1000, 2)
        ))

        # Format source citations
        sources = [
            {
                "chunk_id": c.chunk_id,
                "source": c.metadata.get("source", "unknown"),
                "page": c.metadata.get("page", 1),
                "rerank_score": c.rerank_score,
                "excerpt": c.text[:220] + "..." if len(c.text) > 220 else c.text
            }
            for c in final_chunks
        ]

        total_latency = round((time.time() - start_time) * 1000, 2)

        return QueryResponse(
            query=raw_query,
            answer=answer,
            routed_intent=route,
            sources=sources,
            confidence_score=round(top_confidence, 2),
            execution_steps=steps,
            guardrail_report=guardrail_report,
            total_latency_ms=total_latency
        )
