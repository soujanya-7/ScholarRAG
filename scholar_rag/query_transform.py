"""
Query Transformation Module for ScholarRAG.
Implements:
1. Hypothetical Document Embeddings (HyDE)
2. Multi-Query Expansion & Technical Term Disambiguation
"""

from __future__ import annotations
from typing import List, Optional
from google import genai

from scholar_rag.config import config


class QueryTransformer:
    """Transforms user queries to optimize vector space alignment and retrieval recall."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.gemini_api_key
        self.client = None
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)

    def generate_hyde_passage(self, query: str) -> str:
        """
        Generates a hypothetical technical passage that answers the query.
        Used for dense vector embedding alignment (HyDE).
        """
        if not self.client:
            # Fallback if no LLM key provided
            return query

        prompt = f"""You are an expert scientific researcher. Write a concise, hypothetical paragraph from a technical academic paper that directly and accurately answers this question. Use formal academic language and technical terminology. Do not say "Here is a paragraph", just output the passage.

Question: {query}
Hypothetical Paper Excerpt:"""

        try:
            response = self.client.models.generate_content(
                model=config.llm_model,
                contents=prompt,
            )
            return response.text.strip() if response.text else query
        except Exception:
            return query

    def expand_query(self, query: str) -> List[str]:
        """
        Generates 2-3 alternative technical queries/synonyms to maximize BM25 and vector recall.
        """
        if not self.client:
            return [query]

        prompt = f"""Given the following technical question, generate 2 alternative academic search queries. Include potential acronyms, technical synonyms, or alternative phrasing.
Output each query on a new line with no numbers or bullets.

Original Question: {query}"""

        try:
            response = self.client.models.generate_content(
                model=config.llm_model,
                contents=prompt,
            )
            lines = [line.strip() for line in response.text.split("\n") if line.strip()]
            queries = [query] + lines[:2]
            return list(dict.fromkeys(queries))  # Deduplicate while preserving order
        except Exception:
            return [query]
