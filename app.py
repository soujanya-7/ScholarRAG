"""
ScholarRAG: Interactive Research Paper QA & Retrieval Studio.
Built with Streamlit, ChromaDB, FastEmbed, BM25, FlashRank, and Google Gemini.
"""

from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path
import streamlit as st

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from scholar_rag.config import config
from scholar_rag.pipeline import ScholarRAGPipeline
from scholar_rag.generator import GroundedGenerator

# Page configuration
st.set_page_config(
    page_title="ScholarRAG - Research Paper Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #4F46E5, #06B6D4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .chunk-card {
        background-color: #F8FAFC;
        border-left: 4px solid #4F46E5;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
    .badge-dense { background-color: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-bm25 { background-color: #FEF3C7; color: #92400E; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-hybrid { background-color: #DCFCE7; color: #166534; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
    .badge-rerank { background-color: #F3E8FF; color: #6B21A8; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline(api_key: str = ""):
    """Initializes and caches the ScholarRAG pipeline instance."""
    return ScholarRAGPipeline(api_key=api_key or None)


# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "sample_indexed" not in st.session_state:
    st.session_state.sample_indexed = False


# Sidebar
with st.sidebar:
    st.title("⚙️ ScholarRAG Studio")

    # API Key Configuration
    api_key_input = st.text_input(
        "Google Gemini API Key",
        type="password",
        value=config.gemini_api_key,
        help="Get your free API key at: https://aistudio.google.com/app/apikey"
    )
    if api_key_input != config.gemini_api_key:
        config.gemini_api_key = api_key_input
        os.environ["GEMINI_API_KEY"] = api_key_input

    pipeline = get_pipeline(api_key=api_key_input)

    st.markdown("---")
    st.subheader("📄 Document Ingestion")

    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload Research Papers",
        type=["pdf", "md", "txt"],
        accept_multiple_files=True,
        help="Upload academic papers or technical documentation."
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            if st.button(f"📥 Index '{uploaded_file.name}'", key=f"btn_{uploaded_file.name}"):
                with st.spinner(f"Parsing & indexing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp:
                        tmp.write(uploaded_file.getbuffer())
                        tmp_path = tmp.name

                    # Index document
                    res = pipeline.index_document(tmp_path)
                    st.success(f"Indexed {res['chunks_created']} chunks ({res['total_pages']} pages) in {res['time_seconds']}s!")

    # Preloaded sample paper button
    sample_path = Path(__file__).resolve().parent / "sample_data" / "sample_transformer_paper.md"
    if sample_path.exists() and not st.session_state.sample_indexed:
        if st.button("✨ Load Sample Paper ('Attention Is All You Need')"):
            with st.spinner("Indexing Transformer sample paper..."):
                res = pipeline.index_document(sample_path)
                st.session_state.sample_indexed = True
                st.success("Sample paper indexed successfully!")

    st.markdown("---")
    st.subheader("🎛️ Retrieval Settings")

    retrieval_mode = st.selectbox(
        "Retrieval Strategy",
        options=["hybrid_rerank", "hybrid", "dense", "bm25"],
        format_func=lambda x: {
            "hybrid_rerank": "🌟 Hybrid + Cross-Encoder Rerank (Best)",
            "hybrid": "⚡ Hybrid Search (Dense + BM25 RRF)",
            "dense": "🔵 Dense Vector Search Only",
            "bm25": "🟠 BM25 Sparse Keyword Only"
        }[x],
        index=0,
        help="Choose between naive single-strategy retrieval or intermediate multi-stage hybrid retrieval."
    )

    top_k = st.slider("Top Candidates (Stage 1)", min_value=2, max_value=20, value=8)
    top_n_rerank = st.slider("Context Chunks to LLM (Stage 2)", min_value=1, max_value=8, value=4)
    use_hyde = st.toggle("Enable HyDE (Hypothetical Doc Embeddings)", value=False, help="Generates a hypothetical answer before embedding to bridge query-document semantic gap.")

    st.markdown("---")
    stats = pipeline.get_stats()
    st.markdown(f"**📚 Corpus Stats:** `{stats['total_documents']} Papers` | `{stats['total_chunks']} Chunks`")

    if st.button("🗑️ Reset / Clear Knowledge Base"):
        pipeline.clear()
        st.session_state.sample_indexed = False
        st.session_state.messages = []
        st.info("Corpus cleared!")
        st.rerun()


# Main Header
st.markdown('<div class="main-header">ScholarRAG: Research & Technical QA Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Production-Grade Hybrid RAG with Reciprocal Rank Fusion, Cross-Encoder Re-Ranking, and Strict Citations.</div>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💬 Grounded Chat",
    "🔬 Retrieval Strategy Inspector",
    "📑 Chunk Explorer",
    "🎓 How It Works (Concepts)"
])


# TAB 1: Grounded Chat
with tab1:
    col_chat, col_info = st.columns([3, 1.2])

    with col_chat:
        st.subheader("Ask Questions About Your Papers")

        # Example prompt buttons
        example_cols = st.columns(3)
        sample_questions = [
            "What optimizer and warmup steps were used?",
            "How does positional encoding work?",
            "What BLEU score was achieved on WMT 2014 En-De?"
        ]
        chosen_sample = None
        for i, q in enumerate(sample_questions):
            if example_cols[i].button(q, key=f"ex_{i}"):
                chosen_sample = q

        # Display message history
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "sources" in msg and msg["sources"]:
                    with st.expander("📍 Verifiable Sources & Page Numbers"):
                        for s in msg["sources"]:
                            st.markdown(f"- **Document:** `{s.get('source')}` | **Page:** `{s.get('page_number')}`")

        # User Input
        user_query = st.chat_input("Ask a question about the indexed papers...")
        query_to_run = chosen_sample or user_query

        if query_to_run:
            # Append user message
            st.session_state.messages.append({"role": "user", "content": query_to_run})
            with st.chat_message("user"):
                st.markdown(query_to_run)

            # Generate response
            with st.chat_message("assistant"):
                if not config.gemini_api_key:
                    st.warning("⚠️ Please provide a Gemini API Key in the sidebar to generate answers. In the meantime, you can test the retrieval pipeline in Tab 2!")
                else:
                    with st.spinner("Retrieving relevant context and synthesizing grounded answer..."):
                        try:
                            # 1. Retrieve
                            retrieval_res = pipeline.retrieve(
                                query=query_to_run,
                                mode=retrieval_mode,
                                top_k=top_k,
                                top_n_rerank=top_n_rerank,
                                use_hyde=use_hyde
                            )
                            chunks = retrieval_res["final_chunks"]

                            # 2. Generate
                            generator = GroundedGenerator(api_key=config.gemini_api_key)
                            response = generator.generate(query=query_to_run, chunks=chunks)

                            st.markdown(response.answer)

                            if response.sources_cited:
                                with st.expander("📍 Grounded Source Citations"):
                                    for s in response.sources_cited:
                                        st.markdown(f"- **Document:** `{s.get('source')}` | **Page:** `{s.get('page_number')}`")

                            # Append to session state
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response.answer,
                                "sources": response.sources_cited
                            })

                        except Exception as e:
                            st.error(f"Error generating answer: {str(e)}")

    with col_info:
        st.subheader("💡 Active Pipeline Info")
        st.markdown(f"""
        - **Retrieval Engine:** `{retrieval_mode.upper()}`
        - **Dense Embeddings:** `BAAI/bge-small-en-v1.5`
        - **Sparse Search:** `Okapi BM25`
        - **Re-ranker:** `FlashRank (Cross-Encoder)`
        - **LLM Generator:** `Gemini 2.5 Flash`
        - **HyDE Enabled:** `{'Yes' if use_hyde else 'No'}`
        """)
        st.info("ScholarRAG enforces strict citation grounding. If a fact is not in the text, the model will decline rather than hallucinate.")


# TAB 2: Visual Chunk & Strategy Inspector (Side-by-Side Comparison)
with tab2:
    st.subheader("🔬 Side-by-Side Retrieval Strategy Inspector")
    st.markdown("Compare how different retrieval algorithms rank and score document chunks for the exact same query.")

    inspector_query = st.text_input(
        "Enter a test query for inspection:",
        value="What is the formula for scaled dot-product attention and why is scaling applied?"
    )

    if inspector_query:
        if st.button("🚀 Run Multi-Strategy Comparison"):
            with st.spinner("Executing Dense, BM25, Hybrid RRF, and Cross-Encoder pipelines..."):
                dense_res = pipeline.retrieve(inspector_query, mode="dense", top_k=5)
                bm25_res = pipeline.retrieve(inspector_query, mode="bm25", top_k=5)
                hybrid_res = pipeline.retrieve(inspector_query, mode="hybrid", top_k=5)
                rerank_res = pipeline.retrieve(inspector_query, mode="hybrid_rerank", top_k=8, top_n_rerank=5)

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.markdown("#### 🔵 Dense Vector Only")
                    st.caption(f"Time: {dense_res['retrieval_time_seconds']:.4f}s")
                    for i, c in enumerate(dense_res["final_chunks"], 1):
                        st.markdown(f"""
                        <div class="chunk-card">
                            <span class="badge-dense">Rank #{i} | Cosine: {c['score']:.4f}</span><br>
                            <small><b>Page {c['page_number']}</b> ({c['source']})</small><br>
                            <small>{c['content'][:150]}...</small>
                        </div>
                        """, unsafe_allow_html=True)

                with col2:
                    st.markdown("#### 🟠 BM25 Sparse Only")
                    st.caption(f"Time: {bm25_res['retrieval_time_seconds']:.4f}s")
                    for i, c in enumerate(bm25_res["final_chunks"], 1):
                        st.markdown(f"""
                        <div class="chunk-card">
                            <span class="badge-bm25">Rank #{i} | BM25: {c['score']:.4f}</span><br>
                            <small><b>Page {c['page_number']}</b> ({c['source']})</small><br>
                            <small>{c['content'][:150]}...</small>
                        </div>
                        """, unsafe_allow_html=True)

                with col3:
                    st.markdown("#### 🟢 Hybrid (RRF)")
                    st.caption(f"Time: {hybrid_res['retrieval_time_seconds']:.4f}s")
                    for i, c in enumerate(hybrid_res["final_chunks"], 1):
                        st.markdown(f"""
                        <div class="chunk-card">
                            <span class="badge-hybrid">Rank #{i} | RRF: {c['score']:.6f}</span><br>
                            <small><b>Dense Rank:</b> {c.get('dense_rank', 'N/A')} | <b>BM25 Rank:</b> {c.get('sparse_rank', 'N/A')}</small><br>
                            <small>{c['content'][:150]}...</small>
                        </div>
                        """, unsafe_allow_html=True)

                with col4:
                    st.markdown("#### 🌟 Hybrid + Re-Ranker")
                    st.caption(f"Time: {rerank_res['retrieval_time_seconds']:.4f}s")
                    for i, c in enumerate(rerank_res["final_chunks"], 1):
                        st.markdown(f"""
                        <div class="chunk-card">
                            <span class="badge-rerank">Rank #{i} | Cross-Enc: {c['score']:.4f}</span><br>
                            <small><b>Page {c['page_number']}</b> ({c['source']})</small><br>
                            <small>{c['content'][:150]}...</small>
                        </div>
                        """, unsafe_allow_html=True)


# TAB 3: Chunk Explorer
with tab3:
    st.subheader("📑 Indexed Corpus Chunks Explorer")
    if not pipeline.all_chunks:
        st.info("No chunks currently indexed. Upload a document in the sidebar or load the sample paper.")
    else:
        st.write(f"Total Chunks in Memory: **{len(pipeline.all_chunks)}**")
        for i, chunk in enumerate(pipeline.all_chunks, 1):
            with st.expander(f"Chunk #{i} — [{chunk.source} | Page {chunk.page_number} | ID: {chunk.chunk_id}]"):
                st.code(chunk.content, language="markdown")
                st.json(chunk.metadata)


# TAB 4: How It Works & Theory
with tab4:
    st.subheader("🎓 Master RAG Architecture & Core Concepts")
    st.markdown("""
    ### Why Naive RAG Fails in Production
    - **Naive RAG** splits text into arbitrary chunks, indexes them into a vector DB with a single embedding model, and feeds the top-3 closest vectors to the LLM.
    - **Failure Modes:** Misses exact technical keywords/numbers, fails on complex queries, pulls irrelevant text that distracts the LLM, and lacks verifiable source citations.

    ---

    ### 1. Hybrid Search (Dense + Sparse)
    - **Dense Vector Search (Bi-Encoder):** Projects text into a continuous semantic vector space. Great for conceptual queries (*"How do transformers avoid recurrence?"*).
    - **Sparse Search (Okapi BM25):** Calculates exact term frequency / inverse document frequency. Great for specific keywords (*"Adam $\\beta_2 = 0.98$"*, author names, paper acronyms).

    ---

    ### 2. Reciprocal Rank Fusion (RRF)
    BM25 produces unbounded positive numbers ($[0, \infty)$), while cosine similarity produces scores $\in [0, 1]$. To combine them without fragile normalization, we use **Reciprocal Rank Fusion**:
    $$RRF(d) = \\sum_{m \\in M} \\frac{1}{k + \\text{rank}_m(d)}$$
    where $k = 60$. Chunks appearing near the top of both algorithms receive the highest priority.

    ---

    ### 3. Cross-Encoder Re-Ranking (Two-Stage Retrieval)
    - **Bi-Encoders:** Encode query $Q$ and doc $D$ separately. Fast ($O(1)$ search) but coarse.
    - **Cross-Encoders:** Feed $(Q, D)$ jointly into transformer self-attention layers simultaneously. Computes deep cross-token interactions.
    - **Pipeline:** Hybrid Search generates Top-15 candidates, and FlashRank Cross-Encoder re-scores them to deliver the Top-4 cleanest chunks to the LLM.

    ---

    ### 4. Hypothetical Document Embeddings (HyDE)
    Users ask questions (*"What is the learning rate?"*), but documents contain statements (*"We used a warmup schedule with..."*). HyDE asks the LLM to write a hypothetical excerpt first, embedding the answer passage to find true semantic matches.
    """)
