"""
Chunking Module for ScholarRAG.
Implements Recursive Text Splitting with metadata propagation, section awareness, and token estimation.
"""

from __future__ import annotations
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_text_splitters import RecursiveCharacterTextSplitter
from scholar_rag.document_loader import DocumentPage


class Chunk(BaseModel):
    """Represents an atomic text chunk ready for vector embedding and BM25 indexing."""
    chunk_id: str
    content: str
    source: str
    page_number: int
    chunk_index: int
    char_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class DocumentChunker:
    """
    Splits documents recursively based on natural linguistic boundaries (paragraphs -> sentences -> words)
    while strictly preserving page numbers and source metadata for verifiable citations.
    """

    def __init__(self, chunk_size: int = 700, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Recursive splitter separates by paragraphs, then lines, then words
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", ". ", "; ", " ", ""],
            keep_separator=True,
            length_function=len,
        )

    def chunk_pages(self, pages: List[DocumentPage]) -> List[Chunk]:
        """
        Chunks a list of DocumentPages, assigning unique IDs and attaching page metadata to each chunk.
        """
        chunks: List[Chunk] = []
        global_chunk_idx = 0

        for page in pages:
            raw_text = page.content.strip()
            if not raw_text:
                continue

            # Split text of the individual page
            splits = self.splitter.split_text(raw_text)

            for split_idx, split_content in enumerate(splits):
                cleaned_text = split_content.strip()
                if not cleaned_text:
                    continue

                chunk_id = f"{page.source}_p{page.page_number}_c{split_idx}"
                
                chunk_obj = Chunk(
                    chunk_id=chunk_id,
                    content=cleaned_text,
                    source=page.source,
                    page_number=page.page_number,
                    chunk_index=global_chunk_idx,
                    char_count=len(cleaned_text),
                    metadata={
                        **page.metadata,
                        "chunk_id": chunk_id,
                        "page_number": page.page_number,
                        "source": page.source,
                        "split_idx": split_idx,
                        "total_pages": page.total_pages,
                    }
                )
                chunks.append(chunk_obj)
                global_chunk_idx += 1

        return chunks
