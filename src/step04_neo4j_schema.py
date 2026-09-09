"""Step 4: Neo4j Graph Database Schema, Constraints, and Indexes Setup."""

import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neo4j import GraphDatabase, Driver
from config.settings import Settings, logger


class Neo4jSchemaManager:
    """Manages creation and teardown of Neo4j graph constraints and indexes."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.driver: Driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )

    def close(self) -> None:
        """Closes the Neo4j driver connection."""
        if self.driver:
            self.driver.close()

    def create_schema(self) -> None:
        """Applies uniqueness constraints and search indexes to Neo4j."""
        logger.info("Applying Neo4j schema constraints and indexes...")
        
        # Cypher queries for constraints and indexes
        queries = [
            # 1. Uniqueness constraint for Document nodes
            """
            CREATE CONSTRAINT unique_document_filename IF NOT EXISTS
            FOR (d:Document) REQUIRE d.file_name IS UNIQUE
            """,
            # 2. Uniqueness constraint for Chunk nodes
            """
            CREATE CONSTRAINT unique_chunk_id IF NOT EXISTS
            FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
            """,
            # 3. Index on Chunk page/row number for filtering
            """
            CREATE INDEX chunk_page_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.page)
            """,
            # 4. Index on Chunk file_name for source lookup
            """
            CREATE INDEX chunk_filename_index IF NOT EXISTS
            FOR (c:Chunk) ON (c.file_name)
            """
        ]

        with self.driver.session() as session:
            for query in queries:
                session.run(query)
                
        logger.info("Neo4j schema successfully created/verified.")

    def drop_schema(self) -> None:
        """Removes defined constraints and indexes (useful for reset/testing)."""
        logger.warning("Dropping Neo4j schema constraints and indexes...")
        
        drop_queries = [
            "DROP CONSTRAINT unique_document_filename IF EXISTS",
            "DROP CONSTRAINT unique_chunk_id IF EXISTS",
            "DROP INDEX chunk_page_index IF EXISTS",
            "DROP INDEX chunk_filename_index IF EXISTS",
        ]

        with self.driver.session() as session:
            for query in drop_queries:
                session.run(query)

        logger.info("Neo4j schema constraints removed.")


def main() -> None:
    """Runs Step 4 schema initialization and verification."""
    print("==================================================")
    print(" Starting Step 4 Neo4j Schema Setup Verification")
    print("==================================================")

    schema_manager = Neo4jSchemaManager()
    try:
        schema_manager.create_schema()
        print("\n[+] Created constraints:")
        print("  - Document.file_name (UNIQUE)")
        print("  - Chunk.chunk_id (UNIQUE)")
        print("\n[+] Created indexes:")
        print("  - Chunk.page")
        print("  - Chunk.file_name")
        print("\n[SUCCESS] Step 4 Neo4j Schema creation complete!")
    except Exception as e:
        print(f"\n[-] Schema setup failed: {e}")
    finally:
        schema_manager.close()


if __name__ == "__main__":
    main()