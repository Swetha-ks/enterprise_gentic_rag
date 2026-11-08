"""
CLI Benchmark Runner Script.
Executes the evaluation suite over data/evaluation/ground_truth_dataset.json,
prints an executive summary to terminal, and writes 'benchmark_report.md'.
"""

import sys
import os
from pathlib import Path

# Configure utf-8 encoding for Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import SAMPLE_DOCS_DIR, EVAL_DIR
from src.ingestion.document_loader import DocumentLoader
from src.ingestion.text_splitter import RecursiveTextSplitter
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import DenseVectorRetriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import Reranker
from src.agent.workflow import AgenticRAGWorkflow
from src.evaluation.benchmark import BenchmarkRunner

def main():
    print("==========================================================")
    print("[*] NEXRAG ENTERPRISE - AUTOMATED EVALUATION HARNESS")
    print("==========================================================")

    # 1. Initialize pipeline
    bm25 = BM25Retriever()
    dense = DenseVectorRetriever()
    hybrid = HybridRetriever(bm25, dense)
    reranker = Reranker()
    workflow = AgenticRAGWorkflow(hybrid, reranker)
    splitter = RecursiveTextSplitter()

    # 2. Ingest sample documents
    print("[1/3] Ingesting enterprise sample documents...")
    all_chunks = []
    for doc_file in SAMPLE_DOCS_DIR.glob("*.*"):
        if doc_file.suffix.lower() in [".md", ".pdf", ".txt", ".json"]:
            raw = DocumentLoader.load_file(doc_file)
            chunks = splitter.split_documents(raw)
            all_chunks.extend(chunks)
            print(f"  -> Indexed {len(chunks)} chunks from {doc_file.name}")

    bm25.index_chunks(all_chunks)
    dense.index_chunks(all_chunks)
    print(f"Total corpus size: {len(all_chunks)} chunks indexed into Hybrid Search.\n")

    # 3. Run Benchmark
    print("[2/3] Executing evaluation benchmark against ground truth dataset...")
    eval_dataset = EVAL_DIR / "ground_truth_dataset.json"
    runner = BenchmarkRunner(workflow)
    results = runner.run_benchmark(eval_dataset)

    # 4. Display Results
    print("\n[3/3] Benchmark Results Summary:")
    print("----------------------------------------------------------")
    print(f"  • Total Test Cases:      {results.total_samples}")
    print(f"  • Faithfulness (Grounding): {results.avg_faithfulness * 100:.1f}%")
    print(f"  • Answer Relevance:         {results.avg_answer_relevance * 100:.1f}%")
    print(f"  • Context Precision:        {results.avg_context_precision * 100:.1f}%")
    print(f"  • Context Recall:           {results.avg_context_recall * 100:.1f}%")
    print(f"  • Average Latency:          {results.avg_latency_ms:.1f} ms")
    print("----------------------------------------------------------")

    # Save Markdown report
    report_path = PROJECT_ROOT / "benchmark_report.md"
    report_content = BenchmarkRunner.generate_markdown_report(results)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\n[+] Full Markdown report generated and saved to: {report_path.name}\n")

if __name__ == "__main__":
    main()
