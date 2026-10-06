# 📚 ScholarRAG: Production-Grade Hybrid RAG for Scientific & Technical Papers

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![FastEmbed](https://img.shields.io/badge/Embeddings-FastEmbed_(BGE)-orange.svg)](https://qdrant.github.io/fastembed/)
[![FlashRank](https://img.shields.io/badge/Re--Ranker-FlashRank_(Cross--Encoder)-red.svg)](https://github.com/PrithivirajDamodaran/FlashRank)
[![Gemini](https://img.shields.io/badge/LLM-Google_Gemini-blue.svg)](https://aistudio.google.com/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)

An intermediate, production-ready **Retrieval-Augmented Generation (RAG)** system specifically engineered for technical documentation and academic research papers. Built from first principles to overcome the fundamental failure modes of naive RAG pipelines.

---

## 🎯 Key Architectural Differences: Naive RAG vs. ScholarRAG

| Feature | Naive / Tutorial RAG | **ScholarRAG (This Project)** |
| :--- | :--- | :--- |
| **Document Ingestion** | Blind character splitting (`chunk_size=500`) | **Layout-Aware PDF & Markdown Parsing** with page-level tracking |
| **Retrieval Strategy** | Dense Vector Search Only (Cosine Similarity) | **Hybrid Search (Dense Bi-Encoder + Sparse Okapi BM25)** |
| **Score Fusion** | Ad-hoc score weighting | **Reciprocal Rank Fusion (RRF, $k=60$)** |
| **Re-Ranking** | None (feeds raw top-k to LLM) | **Cross-Encoder Re-Ranking (FlashRank)** for token-level query-doc attention |
| **Query Alignment** | Raw user question embedding | **Hypothetical Document Embeddings (HyDE)** |
| **LLM Grounding** | Generic QA prompt | **Strict Grounding Guardrails with In-Text Citations `[Source, Page]`** |
| **Observability** | Terminal print statements | **Visual Side-by-Side Retrieval Strategy Inspector UI** |

---

## 🏗️ System Architecture

```
                               ┌───────────────────────────┐
                               │  Research Paper (PDF/MD)  │
                               └─────────────┬─────────────┘
                                             │
                                ┌────────────▼────────────┐
                                │   Layout-Aware Parser   │
                                │   & Recursive Chunker   │
                                └────────────┬────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
            ┌──────────▼──────────┐                     ┌──────────▼──────────┐
            │ Dense Embeddings    │                     │ Sparse Inverted     │
            │ (FastEmbed BGE /    │                     │ Index (Okapi BM25)  │
            │  ChromaDB)          │                     │                     │
            └──────────┬──────────┘                     └──────────┬──────────┘
                       │                                           │
                       │          User Query (with HyDE)           │
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                ┌────────────▼────────────┐
                                │ Reciprocal Rank Fusion  │
                                │ (RRF Score Combination) │
                                └────────────┬────────────┘
                                             │ (Top-15 Candidates)
                                ┌────────────▼────────────┐
                                │ Cross-Encoder Re-Ranker │
                                │ (FlashRank ONNX)        │
                                └────────────┬────────────┘
                                             │ (Top-4 Clean Chunks)
                                ┌────────────▼────────────┐
                                │ Grounded Generation     │
                                │ (Gemini 2.5 Flash)      │
                                └────────────┬────────────┘
                                             │
                                ┌────────────▼────────────┐
                                │   Streamlit Studio &    │
                                │   Retrieval Inspector   │
                                └─────────────────────────┘
```

---

## 🧠 Core Engineering Principles You Will Learn

### 1. The Power of Hybrid Search (Dense + Sparse)
- **Dense Retrieval (Bi-Encoders):** Encodes conceptual semantics into high-dimensional space ($D=384$). Excellent for questions like *"How do transformers handle order without recurrence?"*
- **Sparse Retrieval (BM25):** Calculates term frequency and inverse document frequency. Essential for specific technical keywords, exact hyperparameters (e.g. *"Adam $\beta_2=0.98$"*), and author names.

### 2. Reciprocal Rank Fusion (RRF)
Combining cosine similarity ($\in [0, 1]$) with unbounded BM25 scores ($\in [0, \infty)$) using rank distributions:
$$RRF(d) = \sum_{m \in M} \frac{1}{k + \text{rank}_m(d)}$$
where $k=60$. Chunks appearing near the top of both retrieval algorithms receive exponential priority without score calibration issues.

### 3. Cross-Encoder Re-Ranking (Two-Stage Retrieval)
- **Bi-Encoders** independently compress query and document into separate vectors ($O(1)$ search, but coarse).
- **Cross-Encoders** feed $(Q, D)$ jointly through full multi-head self-attention layers to compute true cross-token interactions.
- We implement a two-stage pipeline: Stage 1 retrieves top-15 candidates; Stage 2 cross-encoder delivers the top-4 cleanest passages.

### 4. Hypothetical Document Embeddings (HyDE)
Queries are short and question-shaped; documents are long and answer-shaped. HyDE uses the LLM to hypothesize an academic excerpt, embedding the answer passage to land directly in target embedding clusters.

---

## 🚀 Quickstart Guide

### 1. Clone & Set Up Environment
```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Key
Create a `.env` file (or enter it dynamically in the UI):
```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 3. Run Automated Tests
```bash
python tests/test_pipeline.py
```

### 4. Launch the Interactive Studio UI
```bash
streamlit run app.py
```

---

## 💼 Resume Bullet Points (Copy & Paste for Your Resume)

- **ScholarRAG — Production-Grade Hybrid RAG Engine for Technical Papers**
  - Engineered an end-to-end two-stage RAG pipeline combining **Dense Vector Search (ChromaDB + BGE Embeddings)** and **Sparse Keyword Search (Okapi BM25)** via **Reciprocal Rank Fusion (RRF)**.
  - Implemented **Cross-Encoder Re-ranking (FlashRank)** to eliminate retrieval noise, improving context precision for top-4 passages prior to LLM generation.
  - Integrated **Hypothetical Document Embeddings (HyDE)** and structured PDF parsing with page-level metadata tracking to guarantee verifiable in-text citations `[Source, Page]`.
  - Built an interactive **Retrieval Strategy Inspector** in Streamlit to benchmark dense, sparse, hybrid, and re-ranked results side-by-side with latency profiling.

---

## 🎤 Key Interview Questions & Talking Points

1. **"Why did you choose Hybrid Search over standard Vector Search?"**
   > *Answer:* Dense embeddings excel at high-level semantic themes but fail on exact technical keywords, numbers, and acronyms. BM25 provides exact lexical matching. Combining both with Reciprocal Rank Fusion (RRF) gives the best of both worlds.

2. **"What is the difference between Bi-Encoders and Cross-Encoders?"**
   > *Answer:* Bi-encoders embed query and document separately into vectors for fast index lookup, but token-to-token cross-attention is lost. Cross-encoders process query and passage jointly through transformer layers for exact relevance scoring, making them ideal as a second-stage re-ranker.

3. **"How do you prevent hallucinations in technical QA?"**
   > *Answer:* Through three layers: (1) Re-ranking to strip irrelevant context, (2) Strict grounding system prompt with explicit instruction to decline when context is missing, and (3) Mandatory inline citation format `[Source, Page]` mapped directly to chunk metadata.

---

## 📁 Repository Structure
```
├── app.py                     # Interactive Streamlit Studio & Strategy Inspector
├── requirements.txt           # Project dependencies
├── .env.example               # Environment variables template
├── scholar_rag/
│   ├── __init__.py
│   ├── config.py              # Central hyperparameters & model configuration
│   ├── document_loader.py     # Layout-aware PDF and text parsing with page metadata
│   ├── chunker.py             # Recursive chunker with overlap and token tracking
│   ├── embeddings.py          # FastEmbed & Gemini embedding wrappers
│   ├── vector_store.py        # ChromaDB persistent vector database
│   ├── bm25_retriever.py      # Okapi BM25 sparse keyword retriever
│   ├── hybrid_retriever.py    # Reciprocal Rank Fusion (RRF) engine
│   ├── reranker.py            # FlashRank cross-encoder re-ranking module
│   ├── query_transform.py     # HyDE & query expansion module
│   ├── generator.py           # Grounded Gemini generation with strict citations
│   └── pipeline.py            # Master orchestrator combining all modules
├── sample_data/
│   └── sample_transformer_paper.md
└── tests/
    └── test_pipeline.py       # Automated benchmark & verification test suite
```
