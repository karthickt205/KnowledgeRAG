"""Step 5: Ingesting Document Chunks into Neo4j with Idempotent Cypher Queries."""

import sys
import json
import warnings
from pathlib import Path
from typing import List, Optional, Dict, Any

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neo4j import GraphDatabase, Driver
from langchain_core.documents import Document
from config.settings import Settings, logger
from src.step02_document_loader import KnowledgeDocumentLoader
from src.step03_chunking import KnowledgeChunker


class Neo4jIngestor:
    """Handles idempotent graph insertion for documents and chunks into Neo4j."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.driver: Driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )

    def close(self) -> None:
        """Closes driver connection."""
        if self.driver:
            self.driver.close()

    def ingest_chunks(self, chunks: List[Document]) -> int:
        """Ingests chunks into Neo4j using parameterized MERGE queries to avoid duplicates."""
        if not chunks:
            logger.warning("No chunks provided for Neo4j ingestion.")
            return 0

        logger.info(f"Ingesting {len(chunks)} chunks into Neo4j...")

        cypher_query = """
        UNWIND $batch AS item
        
        // 1. Create or match parent Document node
        MERGE (d:Document {file_name: item.file_name})
        ON CREATE SET 
            d.source = item.source,
            d.document_type = item.document_type,
            d.created_at = timestamp()
        ON MATCH SET
            d.updated_at = timestamp()

        // 2. Create or match Chunk node
        MERGE (c:Chunk {chunk_id: item.chunk_id})
        ON CREATE SET
            c.text = item.text,
            c.file_name = item.file_name,
            c.page = item.page,
            c.chunk_index = item.chunk_index,
            c.chunk_size = item.chunk_size,
            c.metadata_json = item.metadata_json,
            c.created_at = timestamp()
        ON MATCH SET
            c.text = item.text,
            c.metadata_json = item.metadata_json,
            c.updated_at = timestamp()

        // 3. Connect Document to Chunk
        MERGE (d)-[:CONTAINS]->(c)
        """

        batch_data = []
        for idx, chunk in enumerate(chunks):
            metadata = dict(chunk.metadata)
            
            # Extract file_name with guaranteed non-null string fallback
            file_name = metadata.get("file_name")
            if not file_name or file_name == "None":
                source = metadata.get("source", "")
                if source and source != "None":
                    file_name = Path(source).name
                else:
                    file_name = "sample_data.csv"
            
            file_name_str = str(file_name)
            metadata["file_name"] = file_name_str

            batch_data.append({
                "chunk_id": str(metadata.get("chunk_id", f"chunk_{idx}")),
                "file_name": file_name_str,
                "source": str(metadata.get("source", "")),
                "document_type": str(metadata.get("document_type", "csv")),
                "page": int(metadata.get("page", 0)),
                "chunk_index": int(metadata.get("chunk_index", idx)),
                "chunk_size": int(metadata.get("chunk_size", len(chunk.page_content))),
                "text": str(chunk.page_content),
                "metadata_json": json.dumps(metadata)
            })

        with self.driver.session() as session:
            session.run(cypher_query, batch=batch_data)

        logger.info(f"Successfully ingested/updated {len(chunks)} chunks in Neo4j.")
        return len(chunks)

    def get_node_counts(self) -> Dict[str, int]:
        """Utility to retrieve current node counts in the Neo4j database."""
        query = """
        MATCH (d:Document) WITH count(d) AS doc_count
        MATCH (c:Chunk) WITH doc_count, count(c) AS chunk_count
        RETURN doc_count, chunk_count
        """
        with self.driver.session() as session:
            result = session.run(query).single()
            if result:
                return {
                    "documents": result["doc_count"],
                    "chunks": result["chunk_count"]
                }
            return {"documents": 0, "chunks": 0}


def main() -> None:
    """Runs Step 5 Neo4j ingestion pipeline verification."""
    print("==================================================")
    print(" Starting Step 5 Neo4j Ingestion Verification")
    print("==================================================")

    settings = Settings()
    loader = KnowledgeDocumentLoader(settings)
    chunker = KnowledgeChunker(chunk_size=300, chunk_overlap=30, settings=settings)
    ingestor = Neo4jIngestor(settings)

    try:
        raw_docs = loader.load_directory(settings.documents_dir)
        chunks = chunker.split_documents(raw_docs)

        print("\n[+] Executing Primary Ingestion Pass...")
        ingested_count = ingestor.ingest_chunks(chunks)
        counts_pass1 = ingestor.get_node_counts()
        print(f"  Ingested Chunks: {ingested_count}")
        print(f"  Neo4j Node Counts -> Documents: {counts_pass1['documents']}, Chunks: {counts_pass1['chunks']}")

        print("\n[+] Executing Secondary Ingestion Pass (Idempotency Test)...")
        ingestor.ingest_chunks(chunks)
        counts_pass2 = ingestor.get_node_counts()
        print(f"  Neo4j Node Counts -> Documents: {counts_pass2['documents']}, Chunks: {counts_pass2['chunks']}")

        if counts_pass1 == counts_pass2:
            print("\n[+] Idempotency Check Passed: No duplicate nodes were created.")
        else:
            print("\n[!] Warning: Node counts changed during second pass.")

        print("\n[SUCCESS] Step 5 Neo4j Ingestion complete!")
    except Exception as e:
        print(f"\n[-] Ingestion failed: {e}")
    finally:
        ingestor.close()


if __name__ == "__main__":
    main()