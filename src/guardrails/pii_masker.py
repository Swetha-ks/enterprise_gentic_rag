"""
PII (Personally Identifiable Information) Redaction Guardrail.
Detects and masks sensitive personal and security data in user queries and system responses:
Emails, Phone Numbers, Credit Cards, SSNs, API Keys, and IP Addresses.
"""

import re
from typing import Tuple, List, Dict

class PIIMasker:
    PII_PATTERNS = {
        "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
        "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
        "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "API_KEY": r"\b(?:sk-[a-zA-Z0-9]{20,}|ghp_[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b",
        "IPV4": r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"
    }

    def __init__(self):
        self.compiled = {k: re.compile(v) for k, v in self.PII_PATTERNS.items()}

    def mask(self, text: str) -> Tuple[str, bool, List[str]]:
        """
        Scans text, replaces PII with tokens, and reports types detected.
        Returns: (sanitized_text, pii_detected: bool, pii_types_found: List[str])
        """
        sanitized = text
        detected_types = []

        for pii_type, regex in self.compiled.items():
            matches = regex.findall(sanitized)
            if matches:
                detected_types.append(f"{pii_type} ({len(matches)} instance{'s' if len(matches) > 1 else ''})")
                sanitized = regex.sub(f"[REDACTED_{pii_type}]", sanitized)

        pii_found = len(detected_types) > 0
        return sanitized, pii_found, detected_types
