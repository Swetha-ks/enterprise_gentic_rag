# 📊 RAG Evaluation Benchmark Report

**Total Samples Evaluated:** 7
**Average Latency:** `3.1 ms`

## 🎯 Key Performance Indicators (KPIs)

| Metric | Score | Target Standard | Status |
| :--- | :---: | :---: | :---: |
| **Faithfulness (Grounding)** | `60.7%` | > 85.0% | ⚠️ Review |
| **Answer Relevance** | `83.5%` | > 80.0% | ✅ Exceeded |
| **Context Precision** | `57.1%` | > 75.0% | ⚠️ Review |
| **Context Recall** | `52.1%` | > 75.0% | ⚠️ Review |

## 🔬 Sample-Level Evaluation Breakdown

| ID | Question | Faithfulness | Relevance | Context Prec. | Recall | Latency |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| `SEC-01` | What is the mandatory session timeout policy ... | `0.50` | `0.93` | `0.00` | `0.11` | `9ms` |
| `SEC-02` | What cryptographic standards are required for... | `0.75` | `0.93` | `1.00` | `0.70` | `2ms` |
| `SEC-03` | What is the incident notification SLA for a S... | `0.56` | `0.84` | `1.00` | `0.94` | `2ms` |
| `AI-01` | What is the required faithfulness threshold f... | `0.55` | `0.86` | `0.00` | `0.33` | `2ms` |
| `AI-02` | What drop in context recall or faithfulness t... | `0.56` | `0.82` | `1.00` | `0.41` | `1ms` |
| `FIN-01` | What was the company's Annual Recurring Reven... | `0.80` | `0.77` | `1.00` | `0.92` | `3ms` |
| `FIN-02` | How did the team achieve a 14% reduction in c... | `0.55` | `0.69` | `0.00` | `0.23` | `2ms` |