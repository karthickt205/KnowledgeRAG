"""Step 6: Lexical BM25 Search Indexing and Retrieval mapping to Neo4j Chunk IDs."""

import re
import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("rank_bm25 package is required. Install it using: pip install rank-bm25")

from langchain_core.documents import Document
from config.settings import Settings, logger
from src.step02_document_loader import KnowledgeDocumentLoader
from src.step03_chunking import KnowledgeChunker


class BM25RetrieverIndex:
    """Manages tokenization, BM25 Index creation, and lexical retrieval."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.bm25_index: Optional[BM25Okapi] = None
        self.chunks: List[Document] = []
        self.chunk_ids: List[str] = []

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Lowercases text and extracts alphanumeric terms as tokens."""
        return re.findall(r"\w+", text.lower())

    def build_index(self, chunks: List[Document]) -> None:
        """Tokenizes chunks and builds the in-memory BM25 index."""
        if not chunks:
            logger.warning("No chunks provided to build BM25 index.")
            return

        self.chunks = chunks
        self.chunk_ids = [chunk.metadata.get("chunk_id", f"idx_{i}") for i, chunk in enumerate(chunks)]
        
        logger.info(f"Tokenizing and indexing {len(chunks)} chunks for BM25 search...")
        corpus_tokens = [self.tokenize(chunk.page_content) for chunk in chunks]
        self.bm25_index = BM25Okapi(corpus_tokens)
        logger.info("BM25 index successfully built.")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Queries the BM25 index and returns top-k matching chunks with scores."""
        if not self.bm25_index or not self.chunks:
            logger.error("BM25 index has not been built yet. Call build_index() first.")
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Get scores for all chunks in corpus
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Rank scores descending
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in ranked_indices:
            score = scores[idx]
            if score <= 0.0:
                continue  # Exclude non-matching chunks

            chunk = self.chunks[idx]
            results.append({
                "chunk_id": self.chunk_ids[idx],
                "score": round(float(score), 4),
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            })

        return results


def main() -> None:
    """Runs Step 6 BM25 retrieval index verification."""
    print("==================================================")
    print(" Starting Step 6 BM25 Lexical Index Verification")
    print("==================================================")

    settings = Settings()
    loader = KnowledgeDocumentLoader(settings)
    chunker = KnowledgeChunker(chunk_size=300, chunk_overlap=30, settings=settings)
    bm25_retriever = BM25RetrieverIndex(settings)

    # Load and chunk local documents
    raw_docs = loader.load_directory(settings.documents_dir)
    chunks = chunker.split_documents(raw_docs)

    # Build BM25 Index
    bm25_retriever.build_index(chunks)

    # Test Search Queries
    test_queries = [
        "Neo4j nodes and relationships",
        "BM25 exact keyword matching",
        "LLM generation RAG"
    ]

    for query in test_queries:
        print(f"\n[+] Executing Query: '{query}'")
        search_results = bm25_retriever.search(query, top_k=2)

        if not search_results:
            print("  No relevant chunks found.")
            continue

        for res in search_results:
            print(f"  Chunk ID: {res['chunk_id']} | Score: {res['score']}")
            print(f"  Source: {res['metadata'].get('file_name')} (Row/Page: {res['metadata'].get('page')})")
            print(f"  Content Preview: {res['page_content'][:80]}...\n  ---")

    print("[SUCCESS] Step 6 BM25 Lexical Search Index complete!")


if __name__ == "__main__":
    main()