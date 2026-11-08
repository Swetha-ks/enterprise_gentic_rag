"""
Faithfulness and Hallucination Verification Guardrail.
Extracts factual claims from generated answers and verifies their grounding
against retrieved document context chunks.
"""

import re
from typing import List, Tuple, Dict, Any

class FaithfulnessChecker:
    def __init__(self, threshold: float = 0.70):
        self.threshold = threshold

    def evaluate(self, answer: str, context_chunks: List[str]) -> Tuple[float, bool, List[str]]:
        """
        Calculates faithfulness score of answer relative to context chunks.
        Returns: (score: float [0..1], passed: bool, ungrounded_claims: List[str])
        """
        if not answer.strip():
            return 1.0, True, []

        if not context_chunks:
            # If no context was provided, and answer is not a disclaimer, mark as ungrounded
            if "not found" in answer.lower() or "do not have" in answer.lower() or "sorry" in answer.lower():
                return 1.0, True, []
            return 0.20, False, ["Answer generated without supporting context chunks."]

        # Combine all context chunks into a single unified reference body
        full_context = " ".join(context_chunks).lower()
        context_words = set(re.findall(r"\b\w{3,}\b", full_context))

        # Split answer into sentence claims
        sentences = [s.strip() for s in re.split(r"[.!?\n]", answer) if len(s.strip()) > 10]
        if not sentences:
            return 1.0, True, []

        supported_count = 0
        ungrounded_claims = []

        for sent in sentences:
            sent_lower = sent.lower()
            sent_words = set(re.findall(r"\b\w{3,}\b", sent_lower))
            if not sent_words:
                supported_count += 1
                continue

            # Check overlap of substantial claim words against context
            overlap = sent_words.intersection(context_words)
            overlap_ratio = len(overlap) / len(sent_words)

            # Check numbers/dates specifically: if a number is in the claim, it MUST appear in the context
            claim_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", sent_lower))
            number_grounded = True
            if claim_numbers:
                context_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", full_context))
                unseen_numbers = claim_numbers - context_numbers
                if unseen_numbers:
                    number_grounded = False

            if overlap_ratio >= 0.40 and number_grounded:
                supported_count += 1
            else:
                ungrounded_claims.append(sent)

        score = supported_count / len(sentences)
        score = round(min(1.0, max(0.0, score)), 2)
        passed = score >= self.threshold

        return score, passed, ungrounded_claims
