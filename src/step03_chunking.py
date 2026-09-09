"""Step 3: Document Chunking with Metadata Preservation and Deterministic Hash IDs."""

import hashlib
import sys
import warnings
from pathlib import Path
from typing import List, Optional, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import Settings, logger
from src.step02_document_loader import KnowledgeDocumentLoader


class KnowledgeChunker:
    """Handles recursive text chunking and generates deterministic chunk IDs."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        settings: Optional[Settings] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.settings = settings or Settings()
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    @staticmethod
    def generate_chunk_id(source_file: str, page_or_row: Any, index: int, content: str) -> str:
        """Generates a deterministic SHA-256 unique ID for a chunk."""
        raw_key = f"{source_file}::p{page_or_row}::idx{index}::{content[:100]}"
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()[:16]

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Splits a list of documents into chunked documents with updated metadata."""
        if not documents:
            logger.warning("No documents provided to chunker.")
            return []

        logger.info(f"Chunking {len(documents)} document pages/sections...")
        chunked_docs: List[Document] = []

        for doc_idx, doc in enumerate(documents):
            sub_chunks = self.splitter.split_text(doc.page_content)

            # Ensure source and file_name exist on parent metadata
            source = doc.metadata.get("source", "")
            file_name = doc.metadata.get("file_name") or (Path(source).name if source else "sample_data.csv")
            page_or_row = doc.metadata.get("page", doc_idx)

            for chunk_idx, text in enumerate(sub_chunks):
                chunk_id = self.generate_chunk_id(file_name, page_or_row, chunk_idx, text)

                chunk_metadata = dict(doc.metadata)
                chunk_metadata.update({
                    "file_name": str(file_name),
                    "source": str(source),
                    "page": page_or_row,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_idx,
                    "chunk_size": len(text)
                })

                chunked_docs.append(
                    Document(
                        page_content=text,
                        metadata=chunk_metadata
                    )
                )

        logger.info(f"Generated {len(chunked_docs)} chunks from {len(documents)} documents.")
        return chunked_docs


def main() -> None:
    """Runs Step 3 document chunking verification."""
    print("==================================================")
    print(" Starting Step 3 Document Chunking Verification")
    print("==================================================")

    settings = Settings()
    loader = KnowledgeDocumentLoader(settings)
    chunker = KnowledgeChunker(chunk_size=200, chunk_overlap=20, settings=settings)

    loaded_docs = loader.load_directory(settings.documents_dir)
    if not loaded_docs:
        print("[!] No documents found to chunk. Ensure Step 2 created test documents.")
        return

    chunks = chunker.split_documents(loaded_docs)

    print(f"\n[+] Total Chunks Generated: {len(chunks)}")
    print("\n[+] Sample Chunk Metadata & Content:")
    for chunk in chunks[:3]:
        print(f"  Chunk ID: {chunk.metadata['chunk_id']}")
        print(f"  Source File: {chunk.metadata['file_name']}")
        print(f"  Page/Row: {chunk.metadata['page']}")
        print(f"  Content Preview: {chunk.page_content[:100]}...\n---")

    print("[SUCCESS] Step 3 Document Chunking complete!")


if __name__ == "__main__":
    main()