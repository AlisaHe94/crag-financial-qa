# Beyond Naive RAG: Implementing Probabilistic Corrective Frameworks for Financial and Business Analysis

**Dishen Yang (dy2525), Siwen Chen (sc5552), and Jiayi He (jh5111)**

---

## 1. Introduction & Problem Statement

### 1.1 Context & Background

In high-stakes financial environments such as M&A feasibility analysis, post-investment risk management, and corporate valuation, professional analysts rely heavily on extracting precise metrics from massive, unstructured document repositories. Generative AI and Large Language Models offer immense potential to automate this data extraction and synthesize cross-year performance comparisons.

### 1.2 Problem Statement

Despite the widespread adoption of Baseline RAG to ground LLMs in external knowledge, traditional RAG architectures face critical limitations in precision-critical domains. Standard RAG systems rely heavily on the retrieved documents. When the retrieved documents are wrong, the produced answers are inaccurate due to the lack of self-monitoring mechanisms. These standard systems treat all retrieved documents as equally valid. In the financial sector, if a semantic search mistakenly retrieves wrong information, a standard RAG system will confidently hallucinate an answer based on this erroneous data. This risk is unacceptable for analysts in high-stakes environments. A Corrective RAG (CRAG) would act as a safeguard layer, evaluating the document before it is sent to the model for answer generation. We therefore explore how CRAG reduces hallucinations and improves the efficiency of answer generation under high-stakes financial settings.

### 1.3 Scope Definition

The scope of this project is strictly defined as developing a **Probabilistic CRAG** pipeline. This involves building a custom probabilistic retrieval evaluator using statistical classification methods to assign a continuous confidence score P(Relevant | q, d) ∈ [0, 1] to retrieved financial chunks. We implement a soft-routing mechanism with defined thresholds to either refine internal knowledge or trigger a web-search fallback API when internal data is outdated.

We will **not** train a foundational LLM from scratch, build a full-scale commercial UI, or process real-time high-frequency algorithmic trading data. The focus remains on the design of the algorithmic pipeline and comparative evaluation against a baseline RAG.

### 1.4 Research Question

> To what extent does integrating a continuous probabilistic retrieval evaluator and table-aware document processing into a Corrective RAG architecture reduce hallucination rates and improve numerical reasoning accuracy on complex financial documents, compared to a standard RAG baseline?

We also explore how system latency changes under CRAG compared with a standard RAG baseline.

---

## 2. Literature Review & AI Trends

### 2.1 Current AI Trends

The generative AI landscape is currently shifting from static RAG architectures to dynamic, agentic workflows. A prominent trend in enterprise AI is the implementation of self-correcting mechanisms and confidence thresholds to address hallucinations. Industry leaders are actively utilizing these intelligent routing systems. For instance, IBM Watsonx integrates Granite models with external search APIs to evaluate document quality dynamically, rewriting search queries when internal knowledge is insufficient. Similarly, architectures like CRAG are being adopted across disciplines — from agricultural genomics to financial risk advisory, where cross-validation is used to resolve contradictory client financial data.

### 2.2 Literature Review & Limitations of Existing Approaches

Traditional RAG assumes that retrieved context is always accurate. As noted in recent literature, this leads to catastrophic failures in high-stakes fields when the retrieval corpus contains mixed-quality data.

The CRAG framework (Yan et al., 2024) introduced a retrieval evaluator to assess document relevance. Empirical benchmarks demonstrate that CRAG improves factual accuracy by 8%–36.6% over standard RAG, with the most significant gains observed in mixed-quality corpora — a scenario that mirrors enterprise financial databases. Commercial implementations corroborate these findings: CustomGPT.ai reported a 34% reduction in enterprise hallucinations using confidence thresholds, and Anpu Labs reduced customer service escalations by 22% through automated fact-checking.

Most directly relevant to our work is **MultiFinRAG** (Gondhalekar, Patel, and Yeh, 2025), a multimodal RAG framework purpose-built for financial QA on SEC filings. Using separate FAISS indexes per modality (text, table, image), modality-specific similarity thresholds, and a tiered fallback strategy (text-only → text+table+image), MultiFinRAG achieves 19 percentage points higher accuracy than ChatGPT-4o (free tier) on complex financial QA tasks. Critically, the baseline system in that study — which flattens tables to plain text — scores near 0% on table-based questions, confirming that table-aware parsing is not optional in financial document QA.

While MultiFinRAG demonstrates strong performance, it relies solely on a binary threshold-based routing decision (sufficient / insufficient context). Our project addresses this limitation by introducing a probabilistic continuous evaluator P(Relevant | q, d) ∈ [0, 1] that combines a bi-encoder cosine similarity with a cross-encoder re-ranker score. This outputs a calibrated confidence score used for **three-way routing** (correct / ambiguous / incorrect), providing finer-grained, risk-adjusted control over when to escalate to table retrieval or web search — more appropriate for high-stakes financial analysis where over-confidence in retrieved context is itself a risk.

---

## 3. Proposed Methodology

### 3.1 Data Source

The primary data source is the **EDGAR** public database of the U.S. Securities and Exchange Commission (SEC). EDGAR includes financial documents submitted by listed companies — including 10-K annual reports, 10-Q quarterly reports, 8-K interim reports, and DEF 14A proxy statements — all publicly accessible without fees. These documents are typically long, complex in format, and contain mixed text and table content, making them ideal for evaluating Q&A system performance in real financial scenarios.

### 3.2 Baseline Model

This project first builds a standard RAG pipeline as a baseline. The baseline adopts a common, low-cost text-based RAG architecture with three stages:

- **Document processing:** Extract and preprocess text from financial filings; apply fixed-size chunking to construct retrievable text units.
- **Vector retrieval:** Vectorize text chunks using a pre-trained embedding model and build a FAISS vector index. At query time, convert the question to a vector and return the most similar text chunks.
- **Answer generation:** Provide the retrieved chunks and the user's question to the LLM, which generates the final answer based on the given context.

### 3.3 Proposed Corrective RAG Architecture

Standard RAG enters generation directly after retrieval, lacking any judgment on whether results are sufficient or reliable. We insert a probabilistic retrieval evaluator between retrieval and generation, and adopt the tiered fallback strategy demonstrated by MultiFinRAG (Gondhalekar et al., 2025) to handle the multi-modality structure of SEC filings.

#### Tiered Retrieval

Retrieval proceeds in up to three tiers:

1. **Tier 1 — Text-only:** Retrieve text chunks above similarity threshold θ_text = 0.70. If at least n = 6 qualifying chunks are found, proceed to generation.
2. **Tier 2 — Table fallback:** If text hits are sparse (< 6), automatically fetch the top m = 4 table chunks above θ_table = 0.65 and combine them with the text context. This directly addresses the failure mode identified by Gondhalekar et al., where baseline RAG scores near 0% on table questions.
3. **Tier 3 — Web search fallback:** If the overall probabilistic confidence score falls below τ_low, a Tavily web search API call replaces the internal context entirely, handling queries that fall outside the document corpus.

#### Probabilistic Evaluator

Unlike MultiFinRAG's binary threshold router, we add a continuous confidence score P(Relevant | q, d) ∈ [0, 1] combining (i) cosine similarity from the FAISS bi-encoder and (ii) a cross-encoder re-ranker (MiniLM fine-tuned on MS-MARCO):

```
final_score = α · cosine_score + (1 − α) · sigmoid(cross_encoder_logit)
```

This aggregate score drives a three-way soft router with two learnable thresholds τ_high and τ_low:

| Score range | Decision | Action |
|---|---|---|
| score ≥ τ_high | **Correct** | Pass chunks to generation |
| τ_low ≤ score < τ_high | **Ambiguous** | Rewrite query, re-retrieve |
| score < τ_low | **Incorrect** | Trigger web-search fallback |

The thresholds τ_high and τ_low are hyperparameters tuned on a held-out validation split of annotated (query, chunk, relevance-label) pairs. The confidence score is surfaced to the user alongside every answer to support auditable, analyst-facing outputs.

#### Modality-Aware FAISS Indexes

Following MultiFinRAG, we maintain **separate FAISS indexes** for text and table chunks (embedding model: BAAI/bge-base-en-v1.5), with modality-specific similarity thresholds calibrated on a held-out development set. This avoids the precision loss that occurs when table and text embeddings compete within a single shared index.

### 3.4 Additional Improvement Modules

Beyond the CRAG evaluator, we plan two further improvements:

**Module I — Table-aware document processing.** Financial documents contain critical information in tables (balance sheets, income statements, key indicators). The baseline flattens tables to plain text, losing row-column correspondence. We introduce a table detection and parsing module that converts identified tables into structured Markdown or JSON representations and incorporates them into the retrieval process via the dedicated table FAISS index.

**Module II — Semantic chunking strategy.** Fixed-length chunking interrupts semantically complete paragraphs. We implement the MultiFinRAG semantic chunking algorithm (Gondhalekar et al., 2025 §3.3.2): embed each sentence, compute cosine distances between adjacent sentences, mark breakpoints at the 95th-percentile distance, and greedily merge near-duplicate chunks (cosine similarity > 0.85) to reduce redundancy.

### 3.5 Innovation Points

This project focuses specifically on financial document scenarios where data includes both body content and dense table information. Our key innovations relative to existing CRAG and MultiFinRAG systems are:

- A **continuous probabilistic evaluator** (vs. binary thresholding) enabling risk-adjusted routing tunable to analyst requirements.
- A **three-way routing mechanism** that distinguishes ambiguous retrievals (triggering query rewriting) from clearly incorrect ones (triggering web search).
- A **combined tiered + probabilistic** pipeline that integrates MultiFinRAG's proven modality-aware retrieval architecture with CRAG's self-correction logic.

During development we may also explore dynamic routing or self-reflection tokens to further optimize the fallback mechanism.

---

## 4. Implementation & Real-World Value

### 4.1 Real-World Applications

This system is designed for quantitative analysts, business analysts, and financial strategy analysts who extract precise information from dense financial documents. Annual reports, quarterly reports, financial disclosures, and investor materials are long, information-scattered, and present key figures in both text and tables. Standard RAG models are prone to temporal hallucinations on such documents, risking catastrophic consequences for business decisions. The CRAG architecture acts as a safeguard, actively calculating a confidence probability to prevent the model from outputting answers from the wrong documents.

Use cases include financial research, report reading assistance, and information screening. Users can query a financial report in natural language to obtain indicator explanations, disclosure summaries, or locate specific document sections. The system improves information retrieval efficiency, reduces reading cost, and provides clearer references for downstream analysis.

### 4.2 Live Demo

Our primary demo plan is an interactive web interface built with **Streamlit** showcasing a side-by-side comparison of Baseline RAG vs. Probabilistic CRAG. We will use **LangChain** or **LlamaIndex** to connect the chatbot with the interface and the vector database, with **Meta Llama-3.1-8B-Instant served via the Groq LPU API** as the default generator — Groq's smallest production-hosted open Llama (the 3B preview slot was decommissioned in late 2025) and a strong small-model baseline. Commercial APIs from OpenAI or Anthropic are retained as drop-in fallbacks. The demo will include a local small-scale FAISS vector database.

The key demo moment is a **trick question** (e.g., "What is the current federal funds rate?") — the audience watches the system try the internal corpus, receive a low confidence score, and gracefully pivot to a live web search. This directly demonstrates the "edge case handling" rubric requirement.

### 4.3 Scalability & Resource Efficiency

By filtering out irrelevant documents before sending context to the LLM, our architecture reduces API token expenditure, improving efficiency and reducing costs for enterprise-level scaling. MultiFinRAG reported a >60% reduction in token usage through semantic chunk merging alone; we expect similar gains.

---

## 5. Evaluation & Metrics

### 5.1 Quantitative Metrics

We adopt the **four question-type taxonomy** introduced by MultiFinRAG (Gondhalekar et al., 2025) to ensure evaluation covers all modalities present in SEC filings:

| Type | Description |
|---|---|
| Type 1 | Text-based: answerable from narrative paragraphs only |
| Type 2 | Image-based: requiring chart/graph interpretation (partial coverage via table fallback) |
| Type 3 | Table-based: requiring correct row/column lookup from financial tables |
| Type 4 | Text + Table combined: requiring synthesis across both modalities |

**Primary metrics:**

- **Keyword hit rate** (proxy accuracy): fraction of expected answer keywords present in the generated answer.
- **Routing precision** (CRAG only): fraction of queries where the routing decision (correct / ambiguous / incorrect) matches whether the answer was in the corpus.
- **Latency (ms)**: average response time per query, compared across all four experimental conditions.

**Evaluation dimensions:**

- **Baseline comparison:** same question set across all four ablation conditions.
- **Ablation study:** four conditions isolating the contribution of each module independently:
  1. `baseline_no_tables` — naive chunking, text-only retrieval, no CRAG
  2. `baseline_tables` — table-aware chunking, text-only retrieval, no CRAG
  3. `crag_no_tables` — naive chunking, CRAG probabilistic router
  4. `crag_tables` — semantic chunking + table-aware + CRAG *(full system)*
- **Analysis by question type:** accuracy breakdown across Types 1–4 to identify per-modality strengths and weaknesses.

### 5.2 User Experience Improvements

By surfacing the confidence score P(Relevant | q, d) alongside every answer, analysts instantly see the reliability of the model's response, avoiding hidden hallucinations and increasing the practical trustworthiness of the system in high-stakes environments.

---

## 6. Project Plan & Risk Management

### 6.1 Realistic Timeline & Milestones

| Week | Tasks | Milestone |
|---|---|---|
| 1 | Download and organize SEC EDGAR documents; set up development environment; build and test baseline RAG pipeline. | Baseline system running end-to-end. |
| 2 | Develop table detection and parsing module; begin CRAG architecture; build initial evaluation question set (Types 1–4); run preliminary table-question tests. | Table-aware module operational on sample filings. |
| 3–4 | Integrate probabilistic evaluator and semantic chunking; build full pipeline; run baseline comparison experiments and ablation study. | Full system running; primary evaluation data collected. |
| 5 | Build Streamlit chatbot interface; organize experimental results; design demo questions. | Interactive demo interface complete. |
| 6 | Write final report, prepare presentation slides, organize GitHub repository. | All deliverables complete per rubric. |

### 6.2 Required Resources & Availability

- **Computing:** Google Colab GPU (T4/A100) or equivalent cloud compute — free tier sufficient for development; paid tier available if needed.
- **API services:** Groq LPU API serving Meta **Llama-3.1-8B-Instant** as the default generator (free tier sufficient for development); OpenAI or Anthropic APIs retained as commercial fallbacks; Tavily Search API for web fallback.
- **Data:** SEC EDGAR financial documents — publicly available, no permissions or fees required.
- **Development tools:**
  - Vector search: FAISS
  - Text embedding: SentenceTransformers (BAAI/bge-base-en-v1.5)
  - Table detection and parsing: pdfplumber, Camelot, or Docling (selected based on performance)
  - Cross-encoder re-ranker: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Interactive interface: Streamlit
  - Code management: GitHub

### 6.3 Challenges & Mitigation Strategies

**Challenge 1 — Table parsing accuracy is unstable.**
Financial documents contain tables with complex, nested structures. Open-source parsing tools may produce inconsistent output quality.

*Mitigation:* Benchmark multiple tools (pdfplumber, Camelot, Docling) and select the best performer. If automatic parsing quality is insufficient, apply a semi-automatic fallback that manually proofreads a small number of key tables in the evaluation set.

**Challenge 2 — Retrieval evaluator may mis-route.**
CRAG-style quality evaluation depends on the model's ability to judge relevance. Inaccurate evaluation scores could trigger false alarms or fail to catch insufficient evidence.

*Mitigation:* If the cross-encoder approach proves unstable, fall back to a rule-based threshold on FAISS cosine similarity alone. This is simpler, more predictable, and still captures the most egregious retrieval failures.

**Challenge 3 — Open-source model performance may be limited on financial text.**
An 8B-parameter open model such as Llama-3.1-8B-Instant may struggle with financial terminology, numerical reasoning over tables, and very long context windows.

*Mitigation:* All LLM calls are encapsulated in an independent module (`_build_llm_client` in `rag_baseline.py`), so the generator can be swapped via a single environment variable. If Llama-3.1-8B underperforms on Type-3/Type-4 numerical questions, we can fall back to a larger Groq-hosted model (e.g., `llama-3.3-70b-versatile`) or to a commercial API (OpenAI / Anthropic) without touching the retrieval, evaluator, or interface code.

**Minimum deliverable:** At minimum, this project delivers (1) an operational baseline RAG Q&A system, (2) at least one implemented improvement module (table-aware parsing or CRAG evaluator), (3) comparative experimental results, and (4) a working Streamlit chatbot interface.

---

## 7. References

1. Chen, Z., Chen, W., Smiley, C., Shah, S., Borova, I., Langdon, D., ... & Wang, W. Y. (2021). FinQA: A dataset of numerical reasoning over financial data. *EMNLP 2021* (pp. 3697–3711). https://doi.org/10.48550/arXiv.2109.00122

2. Gondhalekar, C., Patel, U., & Yeh, F.-C. (2025). MultiFinRAG: An optimized multimodal retrieval-augmented generation (RAG) framework for financial question answering. *arXiv:2506.20821*. https://arxiv.org/abs/2506.20821

3. Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *NeurIPS 33*, 9459–9474. https://doi.org/10.48550/arXiv.2005.11401

4. Sahin, S. (2024). CRAG (Corrective Retrieval-Augmented Generation) in LLM: What it is & how it works. *Medium*. https://medium.com/@sahin.samia/crag-corrective-retrieval-augmented-generation-in-llm-what-it-is-and-how-it-works-ce24db3343a7

5. Smock, B., Pesala, R., & Abraham, R. (2022). PubTables-1M: Towards comprehensive table extraction from unstructured documents. *CVPR 2022* (pp. 4634–4642). https://doi.org/10.48550/arXiv.2110.00061

6. Yan, S.-Q., Gu, J.-C., Zhu, Y., & Ling, Z.-H. (2024). Corrective retrieval augmented generation. *arXiv:2401.15884*. https://arxiv.org/abs/2401.15884

7. Zhu, F., Lei, W., Huang, Y., Wang, C., Zhang, S., Lv, J., ... & Chua, T. S. (2021). TAT-QA: A question answering benchmark on a hybrid of tabular and textual content in finance. *ACL 2021* (pp. 3277–3287). https://doi.org/10.48550/arXiv.2105.07624

---

*We used ChatGPT and Gemini for idea refinement, information gathering, and structuring of this proposal.*
