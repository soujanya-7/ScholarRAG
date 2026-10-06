"""
Document Loader Module for ScholarRAG.
Extracts text and page-level metadata from PDFs, Markdown, and TXT files.
"""

from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Union

import fitz  # PyMuPDF
from pydantic import BaseModel, Field


class DocumentPage(BaseModel):
    """Represents a single page or segment from a document with rich metadata."""
    content: str
    page_number: int
    source: str
    total_pages: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentLoader:
    """Handles loading and parsing of research papers (PDF, Markdown, TXT)."""

    @staticmethod
    def load_pdf(file_path: str | Path) -> List[DocumentPage]:
        """
        Parses a PDF file page-by-page using PyMuPDF (fitz).
        Extracts clean text while preserving page number references for citations.
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found at: {path}")

        doc = fitz.open(path)
        pages: List[DocumentPage] = []
        total_pages = len(doc)

        for page_idx in range(total_pages):
            page = doc[page_idx]
            # Extract plain text with layout preservation
            text = page.get_text("text").strip()
            
            # If the page has meaningful text, store it
            if text:
                pages.append(
                    DocumentPage(
                        content=text,
                        page_number=page_idx + 1,  # 1-indexed for human citations
                        source=path.name,
                        total_pages=total_pages,
                        metadata={
                            "file_path": str(path.resolve()),
                            "file_name": path.name,
                            "file_size": path.stat().st_size,
                        }
                    )
                )

        doc.close()
        return pages

    @staticmethod
    def load_text(file_path: str | Path) -> List[DocumentPage]:
        """Loads a plain text or Markdown file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found at: {path}")

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        return [
            DocumentPage(
                content=text,
                page_number=1,
                source=path.name,
                total_pages=1,
                metadata={
                    "file_path": str(path.resolve()),
                    "file_name": path.name,
                    "file_size": path.stat().st_size,
                }
            )
        ]

    @classmethod
    def load(cls, file_path: str | Path) -> List[DocumentPage]:
        """Auto-detects file format and parses accordingly."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return cls.load_pdf(path)
        elif suffix in [".txt", ".md", ".markdown"]:
            return cls.load_text(path)
        else:
            raise ValueError(f"Unsupported file format: {suffix}. Supported formats: .pdf, .md, .txt")
