"""Step 8: Hybrid Search Combining Lexical BM25 and Vector Search via Reciprocal Rank Fusion (RRF)."""

import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import Settings, logger
from src.step02_document_loader import KnowledgeDocumentLoader
from src.step03_chunking import KnowledgeChunker
from src.step06_bm25_index import BM25RetrieverIndex
from src.step07_vector_index import Neo4jVectorIndexManager


class HybridRetriever:
    """Executes BM25 lexical and Neo4j vector queries, fusing candidates via RRF."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.bm25_index = BM25RetrieverIndex(self.settings)
        self.vector_mgr = Neo4jVectorIndexManager(self.settings)

    def initialize_index(self) -> None:
        """Loads data, generates embeddings, and initializes both search indices."""
        loader = KnowledgeDocumentLoader(self.settings)
        chunker = KnowledgeChunker(chunk_size=300, chunk_overlap=30, settings=self.settings)

        raw_docs = loader.load_directory(self.settings.documents_dir)
        chunks = chunker.split_documents(raw_docs)

        # Build BM25 lexical index
        self.bm25_index.build_index(chunks)

        # Populate vector embeddings in Neo4j
        self.vector_mgr.update_chunk_embeddings(chunks)

    @staticmethod
    def reciprocal_rank_fusion(
        bm25_results: List[Dict[str, Any]],
        vector_results: List[Dict[str, Any]],
        rrf_k: int = 60,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """Combines ranked search results using Reciprocal Rank Fusion (RRF)."""
        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Dict[str, Any]] = {}

        # 1. Score BM25 Rankings
        for rank, item in enumerate(bm25_results, start=1):
            chunk_id = item["chunk_id"]
            score = 1.0 / (rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
            chunk_map[chunk_id] = item

        # 2. Score Vector Rankings
        for rank, item in enumerate(vector_results, start=1):
            chunk_id = item["chunk_id"]
            score = 1.0 / (rrf_k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + score
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = item

        # 3. Sort by aggregated RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        fused_results = []
        for cid in sorted_ids:
            chunk_data = dict(chunk_map[cid])
            chunk_data["rrf_score"] = round(rrf_scores[cid], 6)
            fused_results.append(chunk_data)

        return fused_results

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Executes lexical + vector search and returns fused RRF results."""
        bm25_res = self.bm25_index.search(query, top_k=top_k * 2)
        vector_res = self.vector_mgr.vector_search(query, top_k=top_k * 2)

        return self.reciprocal_rank_fusion(bm25_res, vector_res, rrf_k=60, top_k=top_k)

    def close(self) -> None:
        """Closes Neo4j driver connection."""
        self.vector_mgr.close()


def main() -> None:
    """Runs Step 8 Hybrid Search verification."""
    print("==================================================")
    print(" Starting Step 8 Hybrid Search Verification")
    print("==================================================")

    hybrid_searcher = HybridRetriever()

    try:
        hybrid_searcher.initialize_index()

        test_query = "Neo4j graph database nodes BM25"
        print(f"\n[+] Executing Hybrid Search Query: '{test_query}'")

        fused_chunks = hybrid_searcher.search(test_query, top_k=3)

        print(f"\n[+] Hybrid RRF Top Results ({len(fused_chunks)} chunks):")
        for rank, res in enumerate(fused_chunks, start=1):
            chunk_id = res.get("chunk_id")
            score = res.get("rrf_score")
            text = res.get("page_content") or res.get("text", "")
            print(f"  Rank {rank} | Chunk ID: {chunk_id} | RRF Score: {score}")
            print(f"  Content Preview: {text[:80]}...\n  ---")

        print("[SUCCESS] Step 8 Hybrid Search complete!")
    except Exception as e:
        print(f"\n[-] Step 8 Hybrid Search failed: {e}")
    finally:
        hybrid_searcher.close()


if __name__ == "__main__":
    main()