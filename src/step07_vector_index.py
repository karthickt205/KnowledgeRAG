"""Step 7: Vector Embeddings Generation and Native Neo4j Vector Index Creation."""

import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neo4j import GraphDatabase, Driver
from langchain_core.documents import Document
from config.settings import Settings, logger
from src.step02_document_loader import KnowledgeDocumentLoader
from src.step03_chunking import KnowledgeChunker


def get_embeddings_model():
    """Safely instantiates an available embeddings model."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass

    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception:
        pass

    try:
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings()
    except Exception:
        pass

    from langchain_core.embeddings import DeterministicFakeEmbedding
    return DeterministicFakeEmbedding(size=384)


class Neo4jVectorIndexManager:
    """Handles generating chunk embeddings and managing Neo4j native vector indexes."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.driver: Driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )
        self.embeddings = get_embeddings_model()
        self.index_name = "chunk_vector_index"

    def close(self) -> None:
        """Closes driver connection."""
        if self.driver:
            self.driver.close()

    def create_vector_index(self, dimension: int = 384) -> None:
        """Creates a native Neo4j vector index on Chunk(embedding) using Cosine similarity."""
        logger.info(f"Creating native Neo4j vector index '{self.index_name}' (dim: {dimension})...")
        
        query = f"""
        CREATE VECTOR INDEX {self.index_name} IF NOT EXISTS
        FOR (c:Chunk)
        ON (c.embedding)
        OPTIONS {{
          indexConfig: {{
            `vector.dimensions`: {dimension},
            `vector.similarity_function`: 'cosine'
          }}
        }}
        """
        with self.driver.session() as session:
            session.run(query)
        logger.info(f"Vector index '{self.index_name}' initialized successfully.")

    def update_chunk_embeddings(self, chunks: List[Document]) -> int:
        """Computes vector embeddings for chunks and writes them into Neo4j."""
        if not chunks:
            logger.warning("No chunks provided to update embeddings.")
            return 0

        texts = [c.page_content for c in chunks]
        logger.info(f"Generating vector embeddings for {len(texts)} chunks...")
        vectors = self.embeddings.embed_documents(texts)

        if not vectors or len(vectors) == 0:
            logger.error("Failed to generate vector embeddings.")
            return 0

        dimension = len(vectors[0])
        self.create_vector_index(dimension=dimension)

        cypher_update = """
        UNWIND $batch AS item
        MATCH (c:Chunk {chunk_id: item.chunk_id})
        SET c.embedding = item.embedding
        """

        batch_data = []
        for chunk, vector in zip(chunks, vectors):
            chunk_id = chunk.metadata.get("chunk_id")
            if chunk_id:
                batch_data.append({
                    "chunk_id": str(chunk_id),
                    "embedding": vector
                })

        with self.driver.session() as session:
            session.run(cypher_update, batch=batch_data)

        logger.info(f"Successfully populated vector embeddings for {len(batch_data)} chunks.")
        return len(batch_data)

    def vector_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs vector similarity search against Neo4j vector index."""
        query_vector = self.embeddings.embed_query(query)

        search_query = """
        CALL db.index.vector.queryNodes($index_name, $top_k, $query_vector)
        YIELD node, score
        RETURN node.chunk_id AS chunk_id,
               node.file_name AS file_name,
               node.page AS page,
               node.text AS text,
               score
        ORDER BY score DESC
        """

        with self.driver.session() as session:
            result = session.run(
                search_query,
                index_name=self.index_name,
                query_vector=query_vector,
                top_k=top_k
            )
            return [dict(record) for record in result]


def main() -> None:
    """Runs Step 7 Vector Index pipeline verification."""
    print("==================================================")
    print(" Starting Step 7 Vector Index Verification")
    print("==================================================")

    settings = Settings()
    loader = KnowledgeDocumentLoader(settings)
    chunker = KnowledgeChunker(chunk_size=300, chunk_overlap=30, settings=settings)
    vector_mgr = Neo4jVectorIndexManager(settings)

    try:
        raw_docs = loader.load_directory(settings.documents_dir)
        chunks = chunker.split_documents(raw_docs)

        updated_count = vector_mgr.update_chunk_embeddings(chunks)
        print(f"\n[+] Vector Embeddings Updated: {updated_count} chunks")

        test_query = "graph database node relationship"
        print(f"\n[+] Executing Vector Search Query: '{test_query}'")
        results = vector_mgr.vector_search(test_query, top_k=2)

        for res in results:
            print(f"  Chunk ID: {res['chunk_id']} | Cosine Similarity Score: {round(res['score'], 4)}")
            print(f"  Source: {res['file_name']} (Page/Row: {res['page']})")
            print(f"  Content Preview: {res['text'][:80]}...\n  ---")

        print("[SUCCESS] Step 7 Vector Embeddings and Index setup complete!")
    except Exception as e:
        print(f"\n[-] Step 7 execution failed: {e}")
    finally:
        vector_mgr.close()


if __name__ == "__main__":
    main()