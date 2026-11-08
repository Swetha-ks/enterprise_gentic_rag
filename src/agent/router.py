"""
Agent Router and Intent Classifier.
Routes user requests to the appropriate processing pipeline:
- ENTERPRISE_RAG: Retrieval over private knowledge base
- DIRECT_CONVERSATION: Chit-chat, greetings, or common standard knowledge
- STRUCTURED_CALCULATION: Arithmetic calculation or numeric comparison
- BLOCKED_MALICIOUS: Adversarial injection or forbidden query
"""

import re
from typing import Tuple

class AgentRouter:
    GREETING_PATTERNS = [
        r"^(hi|hello|hey|good\s+morning|good\s+afternoon|good\s+evening|what's\s+up)\b",
        r"^who\s+are\s+you",
        r"^what\s+can\s+you\s+do",
        r"^how\s+are\s+you"
    ]

    CALCULATION_PATTERNS = [
        r"calculate|compute|what\s+is\s+\d+[\+\-\*\/]\d+",
        r"percentage\s+(change|increase|decrease|growth)",
        r"sum\s+of|difference\s+between|average\s+of"
    ]

    def __init__(self):
        self.greeting_regex = [re.compile(p, re.IGNORECASE) for p in self.GREETING_PATTERNS]
        self.calc_regex = [re.compile(p, re.IGNORECASE) for p in self.CALCULATION_PATTERNS]

    def route(self, query: str, is_safe: bool = True) -> Tuple[str, str]:
        """
        Returns: (route_name: str, rationale: str)
        """
        if not is_safe:
            return "BLOCKED_MALICIOUS", "Query violated system safety or injection guardrails."

        cleaned = query.strip()

        # Check for conversational greeting
        for pat in self.greeting_regex:
            if pat.search(cleaned):
                return "DIRECT_CONVERSATION", "Standard greeting or identity inquiry detected; bypassing document retrieval."

        # Check for structured calculation
        for pat in self.calc_regex:
            if pat.search(cleaned):
                return "STRUCTURED_CALCULATION", "Math or numerical calculation request detected; routing to computational tool."

        # Default enterprise intent
        return "ENTERPRISE_RAG", "Domain question requiring enterprise document search and grounded synthesis."
