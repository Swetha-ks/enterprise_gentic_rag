"""
Automated Benchmark Runner.
Executes batch evaluation on curated ground truth datasets and computes
composite RAG metrics (Faithfulness, Relevance, Context Precision & Recall).
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any
from ..models.schemas import BenchmarkResult, QueryRequest
from ..agent.workflow import AgenticRAGWorkflow
from .metrics import RAGEvaluationMetrics

class BenchmarkRunner:
    def __init__(self, workflow: AgenticRAGWorkflow):
        self.workflow = workflow

    def run_benchmark(self, dataset_path: str | Path) -> BenchmarkResult:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Evaluation dataset not found at: {dataset_path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data.get("samples", [])
        detailed_results = []

        total_faithfulness = 0.0
        total_relevance = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_latency = 0.0

        for sample in samples:
            q = sample["question"]
            gt_ctx = sample.get("ground_truth_context", "")
            gt_ans = sample.get("ground_truth_answer", "")

            req = QueryRequest(
                query=q,
                top_k=3,
                enable_hybrid=True,
                enable_rerank=True,
                enable_guardrails=True
            )

            resp = self.workflow.execute(req)
            retrieved_texts = [s["excerpt"] for s in resp.sources]

            # Calculate individual metrics
            faith = RAGEvaluationMetrics.calculate_faithfulness(resp.answer, retrieved_texts)
            relevance = RAGEvaluationMetrics.calculate_answer_relevance(q, resp.answer)
            precision = RAGEvaluationMetrics.calculate_context_precision(gt_ctx, retrieved_texts)
            recall = RAGEvaluationMetrics.calculate_context_recall(gt_ctx, retrieved_texts)

            total_faithfulness += faith
            total_relevance += relevance
            total_precision += precision
            total_recall += recall
            total_latency += resp.total_latency_ms

            detailed_results.append({
                "id": sample.get("id", ""),
                "question": q,
                "answer": resp.answer,
                "faithfulness": faith,
                "answer_relevance": relevance,
                "context_precision": precision,
                "context_recall": recall,
                "latency_ms": resp.total_latency_ms
            })

        n = max(1, len(samples))
        return BenchmarkResult(
            total_samples=n,
            avg_faithfulness=round(total_faithfulness / n, 3),
            avg_answer_relevance=round(total_relevance / n, 3),
            avg_context_precision=round(total_precision / n, 3),
            avg_context_recall=round(total_recall / n, 3),
            avg_latency_ms=round(total_latency / n, 1),
            detailed_samples=detailed_results
        )

    @staticmethod
    def generate_markdown_report(result: BenchmarkResult) -> str:
        """Formats benchmark results as a clean GitHub-style Markdown report."""
        md = [
            "# 📊 RAG Evaluation Benchmark Report",
            "",
            f"**Total Samples Evaluated:** {result.total_samples}",
            f"**Average Latency:** `{result.avg_latency_ms} ms`",
            "",
            "## 🎯 Key Performance Indicators (KPIs)",
            "",
            "| Metric | Score | Target Standard | Status |",
            "| :--- | :---: | :---: | :---: |",
            f"| **Faithfulness (Grounding)** | `{result.avg_faithfulness * 100:.1f}%` | > 85.0% | {'✅ Exceeded' if result.avg_faithfulness >= 0.85 else '⚠️ Review'} |",
            f"| **Answer Relevance** | `{result.avg_answer_relevance * 100:.1f}%` | > 80.0% | {'✅ Exceeded' if result.avg_answer_relevance >= 0.80 else '⚠️ Review'} |",
            f"| **Context Precision** | `{result.avg_context_precision * 100:.1f}%` | > 75.0% | {'✅ Exceeded' if result.avg_context_precision >= 0.75 else '⚠️ Review'} |",
            f"| **Context Recall** | `{result.avg_context_recall * 100:.1f}%` | > 75.0% | {'✅ Exceeded' if result.avg_context_recall >= 0.75 else '⚠️ Review'} |",
            "",
            "## 🔬 Sample-Level Evaluation Breakdown",
            "",
            "| ID | Question | Faithfulness | Relevance | Context Prec. | Recall | Latency |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |"
        ]

        for s in result.detailed_samples:
            q_short = s['question'][:45] + "..." if len(s['question']) > 45 else s['question']
            md.append(
                f"| `{s['id']}` | {q_short} | `{s['faithfulness']:.2f}` | `{s['answer_relevance']:.2f}` | `{s['context_precision']:.2f}` | `{s['context_recall']:.2f}` | `{s['latency_ms']:.0f}ms` |"
            )

        return "\n".join(md)
