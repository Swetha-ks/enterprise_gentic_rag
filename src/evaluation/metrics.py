"""
Evaluation Metrics for RAG Pipelines.
Implements the core Ragas metrics:
- Faithfulness: Grounding of generated claims in retrieved context
- Answer Relevance: Alignment of generated answer with user question
- Context Precision: Signal-to-noise ratio in top retrieved chunks
- Context Recall: Extent to which ground truth facts are captured in context
"""

import re
from typing import List, Dict, Any

class RAGEvaluationMetrics:
    @staticmethod
    def calculate_faithfulness(answer: str, context_chunks: List[str]) -> float:
        """Measures whether the claims in the answer can be inferred from the context."""
        if not answer.strip() or not context_chunks:
            return 0.0

        full_context = " ".join(context_chunks).lower()
        context_words = set(re.findall(r"\b\w{3,}\b", full_context))

        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0

        supported = 0
        for sent in sentences:
            words = set(re.findall(r"\b\w{3,}\b", sent.lower()))
            if not words:
                continue
            overlap = words.intersection(context_words)
            if len(overlap) / len(words) >= 0.35:
                supported += 1

        return round(supported / len(sentences), 4)

    @staticmethod
    def calculate_answer_relevance(question: str, answer: str) -> float:
        """Measures how directly the answer addresses the question."""
        q_words = set(re.findall(r"\b\w{3,}\b", question.lower()))
        a_words = set(re.findall(r"\b\w{3,}\b", answer.lower()))

        if not q_words or not a_words:
            return 0.0

        overlap = q_words.intersection(a_words)
        relevance = len(overlap) / len(q_words)
        # Scale with answer completeness bonus
        length_factor = min(1.0, len(answer.split()) / 15.0)
        score = (0.7 * relevance) + (0.3 * length_factor)
        return round(min(1.0, max(0.1, score)), 4)

    @staticmethod
    def calculate_context_precision(ground_truth_context: str, retrieved_chunks: List[str]) -> float:
        """Measures whether the most relevant chunks are ranked at the top."""
        if not ground_truth_context or not retrieved_chunks:
            return 0.0

        gt_tokens = set(re.findall(r"\b\w{3,}\b", ground_truth_context.lower()))
        precisions = []
        hits = 0

        for k, chunk in enumerate(retrieved_chunks, start=1):
            chunk_tokens = set(re.findall(r"\b\w{3,}\b", chunk.lower()))
            overlap = gt_tokens.intersection(chunk_tokens)
            if len(overlap) / max(1, len(gt_tokens)) >= 0.25:
                hits += 1
                precisions.append(hits / k)

        if not precisions:
            return 0.0

        return round(sum(precisions) / len(precisions), 4)

    @staticmethod
    def calculate_context_recall(ground_truth_context: str, retrieved_chunks: List[str]) -> float:
        """Measures whether all necessary facts from ground truth are present in retrieved chunks."""
        if not ground_truth_context or not retrieved_chunks:
            return 0.0

        gt_tokens = set(re.findall(r"\b\w{3,}\b", ground_truth_context.lower()))
        all_retrieved = " ".join(retrieved_chunks).lower()
        retrieved_tokens = set(re.findall(r"\b\w{3,}\b", all_retrieved))

        recalled = gt_tokens.intersection(retrieved_tokens)
        return round(len(recalled) / max(1, len(gt_tokens)), 4)
