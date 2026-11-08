"""
Prompt Injection and Jailbreak Guardrail.
Detects adversarial prompt engineering attacks, jailbreak attempts, system override patterns,
and indirect delimiter injections.
"""

import re
from typing import Tuple, List

class InjectionDetector:
    # Adversarial jailbreak and override patterns
    INJECTION_PATTERNS = [
        r"(?i)ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
        r"(?i)disregard\s+(the\s+)?(previous|prior|system)\s+rules?",
        r"(?i)you\s+are\s+now\s+(in\s+)?(developer\s+mode|dan\s+mode|unrestricted|god\s+mode)",
        r"(?i)system\s+override",
        r"(?i)print\s+(your\s+)?(initial|system|hidden)\s+prompt",
        r"(?i)reveal\s+(the\s+)?(system\s+instructions?|api\s+keys?)",
        r"(?i)act\s+as\s+an\s+unfiltered\s+ai",
        r"(?i)bypass\s+all\s+(filters|guardrails|safety\s+guidelines)",
        r"(?i)do\s+anything\s+now",
        r"(?i)pretend\s+you\s+have\s+no\s+(morals|filters|restrictions)",
        r"(?i)jailbreak",
        r"(?i)simulated\s+mode\s+enabled",
        r"(?i)delimiter\s+attack|```system|<\|im_start\|>system"
    ]

    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
        self.compiled_regex = [re.compile(p) for p in self.INJECTION_PATTERNS]

    def inspect(self, text: str) -> Tuple[bool, float, List[str]]:
        """
        Inspects text for prompt injection.
        Returns: (is_injection: bool, risk_score: float, matched_rules: List[str])
        """
        matched_rules = []
        score = 0.0

        for pattern in self.compiled_regex:
            match = pattern.search(text)
            if match:
                matched_rules.append(match.group(0))
                score += 0.45

        # Heuristic checks: suspicious repeated delimiters, excessive brackets
        if text.count("```") > 3 or "<|im_start|>" in text:
            score += 0.35
            matched_rules.append("Suspicious structural delimiters detected")

        risk_score = min(1.0, score)
        is_injection = risk_score >= self.threshold

        return is_injection, round(risk_score, 2), matched_rules
