"""
Generation Module for ScholarRAG using Google Gemini.
Enforces strict citation grounding, hallucination guardrails, and source attribution.
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional, Iterator
from google import genai

from google.genai import types
from pydantic import BaseModel, Field
from scholar_rag.config import config


class RAGResponse(BaseModel):
    """Structured response from ScholarRAG."""
    answer: str
    sources_cited: List[Dict[str, Any]] = Field(default_factory=list)
    retrieved_chunks: List[Dict[str, Any]] = Field(default_factory=list)
    query: str
    model: str


SYSTEM_PROMPT = """You are ScholarRAG, an authoritative and precise scientific research assistant.
Your goal is to answer technical and academic questions accurately based ONLY on the provided context excerpts from research papers.

RULES YOU MUST STRICTLY FOLLOW:
1. Grounding: Rely strictly on facts stated directly in the provided context. Do NOT extrapolate or assume information not present.
2. In-text Citations: Whenever you state a fact, metric, formula, or finding, include an inline citation formatted exactly as: `[Source: <filename>, Page: <page_number>]`.
3. Honesty & Boundaries: If the provided excerpts do not contain enough information to answer the question completely, state clearly: "Based on the provided documents, there is insufficient information to answer..." Do NOT fabricate facts.
4. Technical Precision: Preserve mathematical notation, hyperparameters, equations, and acronyms exactly as presented in the source.
5. Organization: Use clean Markdown with headers, bullet points, and code/math blocks where appropriate.
"""


class GroundedGenerator:
    """Handles prompt construction, citation formatting, and LLM generation via Gemini."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved chunks into a clean, numbered context block with clear metadata boundaries."""
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            source = chunk.get("source", "Unknown")
            page = chunk.get("page_number", 1)
            score = chunk.get("score", 0.0)
            content = chunk.get("content", "").strip()

            block = (
                f"--- EXCERPT [{i}] ---\n"
                f"Source Document: {source}\n"
                f"Page: {page}\n"
                f"Relevance Score: {score}\n"
                f"Content:\n{content}\n"
            )
            context_blocks.append(block)

        return "\n".join(context_blocks)

    def generate(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> RAGResponse:
        """Generates a grounded answer with citations for the user query."""
        if not self.client:
            raise ValueError(
                "Gemini API Key is not configured. Please set GEMINI_API_KEY in your .env file."
            )

        if not chunks:
            return RAGResponse(
                answer="No relevant documents or chunks were found to answer your query. Please upload or index relevant research papers first.",
                sources_cited=[],
                retrieved_chunks=[],
                query=query,
                model=config.llm_model
            )

        formatted_context = self._format_context(chunks)
        user_message = f"""CONTEXT FROM RESEARCH PAPERS:
{formatted_context}

---
USER QUESTION:
{query}

Please provide a comprehensive, well-structured, and strictly cited answer."""

        sys_prompt = system_prompt or SYSTEM_PROMPT

        response = self.client.models.generate_content(
            model=config.llm_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=config.llm_temperature,
                max_output_tokens=config.max_output_tokens,
            )
        )

        answer_text = response.text.strip() if response.text else "No response generated."

        # Extract unique sources cited
        unique_sources = []
        seen = set()
        for c in chunks:
            key = (c.get("source"), c.get("page_number"))
            if key not in seen:
                seen.add(key)
                unique_sources.append({
                    "source": c.get("source"),
                    "page_number": c.get("page_number"),
                    "chunk_id": c.get("chunk_id")
                })

        return RAGResponse(
            answer=answer_text,
            sources_cited=unique_sources,
            retrieved_chunks=chunks,
            query=query,
            model=config.llm_model
        )

    def generate_stream(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        system_prompt: Optional[str] = None
    ) -> Iterator[str]:
        """Streams generated tokens from Gemini in real-time for UI responsiveness."""
        if not self.client:
            yield "Gemini API Key is not configured. Please set GEMINI_API_KEY in your .env file."
            return

        if not chunks:
            yield "No relevant document chunks found."
            return

        formatted_context = self._format_context(chunks)
        user_message = f"""CONTEXT FROM RESEARCH PAPERS:
{formatted_context}

---
USER QUESTION:
{query}

Please provide a comprehensive, well-structured, and strictly cited answer."""

        sys_prompt = system_prompt or SYSTEM_PROMPT

        stream = self.client.models.generate_content_stream(
            model=config.llm_model,
            contents=user_message,
            config=types.GenerateContentConfig(
                system_instruction=sys_prompt,
                temperature=config.llm_temperature,
                max_output_tokens=config.max_output_tokens,
            )
        )

        for chunk in stream:
            if chunk.text:
                yield chunk.text
