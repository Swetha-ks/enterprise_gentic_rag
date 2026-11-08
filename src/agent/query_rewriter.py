"""
Query Rewriter and Multi-Query Expansion.
Improves retrieval recall by decomposing complex queries, generating synonyms,
and creating hypothetical document responses (HyDE-style expansion).
"""

import re
from typing import List

class QueryRewriter:
    SYNONYM_MAP = {
        "mfa": "multi-factor authentication 2FA verification",
        "pto": "paid time off vacation leave policy",
        "sla": "service level agreement uptime commitment",
        "retention": "data retention purge deletion schedule",
        "revenue": "revenue ARR quarterly earnings financial performance",
        "breach": "security breach incident response disclosure",
        "ai": "artificial intelligence machine learning LLM models",
        "compliance": "compliance regulatory SOC2 GDPR audit"
    }

    @classmethod
    def expand_query(cls, query: str) -> List[str]:
        """
        Generates expanded search queries to capture lexical and semantic variants.
        """
        queries = [query]
        tokens = query.lower().split()

        # 1. Expand acronyms and synonyms
        added_terms = []
        for token in tokens:
            cleaned = re.sub(r"[^\w]", "", token)
            if cleaned in cls.SYNONYM_MAP:
                added_terms.append(cls.SYNONYM_MAP[cleaned])

        if added_terms:
            queries.append(f"{query} {' '.join(added_terms)}")

        # 2. HyDE-style hypothetical context target query
        hyde_variant = f"Official policy and documented details regarding {query.strip()}"
        queries.append(hyde_variant)

        return queries
