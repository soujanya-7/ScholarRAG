"""
Configuration settings for ScholarRAG.
Handles environment variables, model names, chunking parameters, and retrieval hyperparameters.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

class RAGConfig(BaseModel):
    # LLM Settings
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    llm_model: str = Field(default="gemini-2.5-flash")
    llm_temperature: float = Field(default=0.2)
    max_output_tokens: int = Field(default=2048)

    # Embedding Settings
    embedding_provider: str = Field(default_factory=lambda: os.getenv("EMBEDDING_PROVIDER", "local"))
    local_embedding_model: str = Field(default="BAAI/bge-small-en-v1.5")
    gemini_embedding_model: str = Field(default="models/text-embedding-004")

    # Document & Chunking Settings
    chunk_size: int = Field(default=700)  # Characters per chunk
    chunk_overlap: int = Field(default=120)  # Overlap between consecutive chunks

    # Vector Storage Settings
    chroma_persist_dir: str = Field(
        default_factory=lambda: os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma_db"))
    )
    collection_name: str = Field(default="research_papers")

    # Retrieval & Re-ranking Hyperparameters
    top_k_dense: int = Field(default=8)  # Retrieved from vector DB
    top_k_sparse: int = Field(default=8)  # Retrieved from BM25
    rrf_k: int = Field(default=60)  # Reciprocal Rank Fusion constant
    top_n_rerank: int = Field(default=4)  # Final chunks passed to LLM after reranking
    reranker_model: str = Field(default="ms-marco-TinyBERT-L-2-v2")

    # HyDE (Hypothetical Document Embeddings)
    enable_hyde: bool = Field(default=False)

# Global default config instance
config = RAGConfig()
