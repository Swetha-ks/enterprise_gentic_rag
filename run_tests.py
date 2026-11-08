"""
Standalone Test Runner for NexRAG Enterprise.
Runs all unit and integration tests using Python's standard library.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import test functions
from tests.test_retrieval import test_bm25_retrieval, test_hybrid_rrf_scoring, test_reranker_reordering
from tests.test_guardrails import test_injection_detection, test_pii_masking, test_faithfulness_checking
from tests.test_workflow import test_full_agent_workflow

class TestEnterpriseRAG(unittest.TestCase):
    def test_01_bm25(self):
        test_bm25_retrieval()

    def test_02_hybrid_rrf(self):
        test_hybrid_rrf_scoring()

    def test_03_reranker(self):
        test_reranker_reordering()

    def test_04_injection(self):
        test_injection_detection()

    def test_05_pii(self):
        test_pii_masking()

    def test_06_faithfulness(self):
        test_faithfulness_checking()

    def test_07_full_workflow(self):
        test_full_agent_workflow()

def main():
    print("==========================================================")
    print("[*] RUNNING NEXRAG ENTERPRISE TEST SUITE")
    print("==========================================================")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEnterpriseRAG)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n[+] All 7 Unit & Integration Tests Passed Successfully!")
        sys.exit(0)
    else:
        print("\n[-] Tests Failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
