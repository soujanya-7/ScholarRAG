"""
ScholarRAG: FastAPI Backend Server.
Serves the modern bespoke Web UI and exposes REST APIs for indexing, multi-strategy retrieval,
and grounded generation.
"""

from __future__ import annotations
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, Body
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from scholar_rag.config import config
from scholar_rag.pipeline import ScholarRAGPipeline
from scholar_rag.generator import GroundedGenerator

app = FastAPI(title="ScholarRAG API", version="1.0.0")

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline_instance = ScholarRAGPipeline()

# Request Models
class SearchRequest(BaseModel):
    query: str
    mode: str = "hybrid_rerank"
    top_k: int = 8
    top_n_rerank: int = 4
    use_hyde: bool = False

class ChatRequest(BaseModel):
    query: str
    mode: str = "hybrid_rerank"
    top_k: int = 8
    top_n_rerank: int = 4
    use_hyde: bool = False
    api_key: Optional[str] = None

class ConfigUpdateRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    chunk_size: Optional[int] = None
    chunk_overlap: Optional[int] = None


@app.get("/api/stats")
async def get_stats():
    """Returns current indexing statistics and configuration."""
    stats = pipeline_instance.get_stats()
    return {
        "status": "success",
        "total_documents": stats["total_documents"],
        "total_chunks": stats["total_chunks"],
        "documents": stats["documents"],
        "has_api_key": bool(config.gemini_api_key),
        "embedding_provider": config.embedding_provider,
        "llm_model": config.llm_model,
        "reranker_model": config.reranker_model,
    }


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads and indexes a PDF, Markdown, or TXT file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in [".pdf", ".md", ".txt", ".markdown"]:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Index document
        res = pipeline_instance.index_document(tmp_path)
        # Fix source name to match original filename
        res["source"] = file.filename
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-sample")
async def load_sample_paper():
    """Indexes the built-in Attention Is All You Need sample paper."""
    sample_path = Path(__file__).resolve().parent / "sample_data" / "sample_transformer_paper.md"
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample paper file not found")

    try:
        res = pipeline_instance.index_document(sample_path)
        return {"status": "success", "result": res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/benchmark-search")
async def benchmark_search(req: SearchRequest):
    """
    Executes a multi-strategy comparison query across:
    1. Dense Vector Search
    2. BM25 Sparse Search
    3. Hybrid RRF Search
    4. Cross-Encoder Re-Ranked Search
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    try:
        dense_res = pipeline_instance.retrieve(req.query, mode="dense", top_k=req.top_k, use_hyde=req.use_hyde)
        bm25_res = pipeline_instance.retrieve(req.query, mode="bm25", top_k=req.top_k)
        hybrid_res = pipeline_instance.retrieve(req.query, mode="hybrid", top_k=req.top_k, use_hyde=req.use_hyde)
        rerank_res = pipeline_instance.retrieve(
            req.query,
            mode="hybrid_rerank",
            top_k=req.top_k,
            top_n_rerank=req.top_n_rerank,
            use_hyde=req.use_hyde
        )

        return {
            "status": "success",
            "query": req.query,
            "strategies": {
                "dense": dense_res,
                "bm25": bm25_res,
                "hybrid": hybrid_res,
                "hybrid_rerank": rerank_res
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat_generate(req: ChatRequest):
    """
    Retrieves context and generates a grounded response with Gemini.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    api_key = req.api_key or config.gemini_api_key
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="Gemini API key is required. Please set it in Settings or .env"
        )

    try:
        start_time = time.time()
        # 1. Retrieve
        retrieval_res = pipeline_instance.retrieve(
            query=req.query,
            mode=req.mode,
            top_k=req.top_k,
            top_n_rerank=req.top_n_rerank,
            use_hyde=req.use_hyde
        )
        chunks = retrieval_res["final_chunks"]
        retrieval_time = round(time.time() - start_time, 3)

        # 2. Generate with GroundedGenerator
        gen_start = time.time()
        generator = GroundedGenerator(api_key=api_key)
        response = generator.generate(query=req.query, chunks=chunks)
        gen_time = round(time.time() - gen_start, 3)

        return {
            "status": "success",
            "answer": response.answer,
            "sources_cited": response.sources_cited,
            "retrieved_chunks": chunks,
            "stats": {
                "retrieval_time_s": retrieval_time,
                "generation_time_s": gen_time,
                "total_time_s": round(retrieval_time + gen_time, 3),
                "chunks_used": len(chunks),
                "mode": req.mode,
                "hyde_used": req.use_hyde
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/chunks")
async def get_all_chunks():
    """Returns all chunks currently stored in memory."""
    chunks = pipeline_instance.all_chunks
    return {
        "status": "success",
        "total_chunks": len(chunks),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "source": c.source,
                "page_number": c.page_number,
                "chunk_index": c.chunk_index,
                "char_count": c.char_count,
                "content": c.content,
                "metadata": c.metadata
            }
            for c in chunks
        ]
    }


@app.post("/api/clear")
async def clear_corpus():
    """Clears all indexed data."""
    try:
        pipeline_instance.clear()
        return {"status": "success", "message": "Corpus cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/update-config")
async def update_config(req: ConfigUpdateRequest):
    """Updates API key and runtime configurations."""
    if req.gemini_api_key is not None:
        config.gemini_api_key = req.gemini_api_key
        os.environ["GEMINI_API_KEY"] = req.gemini_api_key
    return {"status": "success", "message": "Configuration updated"}


# Serve static files for the modern frontend
static_dir = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def serve_index():
    """Serves the main HTML application."""
    index_file = static_dir / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "ScholarRAG API Server Running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
