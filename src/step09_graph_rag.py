"""Step 9: Graph-Augmented RAG Context Assembly combining Hybrid Retrieval and Neo4j Graph Traversal."""

import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neo4j import GraphDatabase, Driver
from config.settings import Settings, logger
from src.step08_hybrid_search import HybridRetriever


class GraphRAGContextBuilder:
    """Retrieves chunks via hybrid search and enriches them with Neo4j graph context."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.driver: Driver = GraphDatabase.driver(
            self.settings.neo4j_uri,
            auth=(self.settings.neo4j_username, self.settings.neo4j_password)
        )
        self.hybrid_retriever = HybridRetriever(self.settings)

    def initialize_index(self) -> None:
        """Initializes hybrid retrieval indices."""
        self.hybrid_retriever.initialize_index()

    def fetch_graph_subgraph(self, chunk_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetches surrounding graph structure for retrieved chunk nodes without triggering Cypher warnings."""
        if not chunk_ids:
            return []

        cypher_query = """
        MATCH (c:Chunk)
        WHERE c.chunk_id IN $chunk_ids
        OPTIONAL MATCH (c)-[r]-(target)
        RETURN c.chunk_id AS chunk_id,
               type(r) AS relationship,
               labels(target) AS target_labels,
               CASE 
                 WHEN target.chunk_id IS NOT NULL THEN target.chunk_id
                 WHEN target.file_name IS NOT NULL THEN target.file_name
                 ELSE elementId(target)
               END AS target_name
        """

        graph_facts = []
        with self.driver.session() as session:
            result = session.run(cypher_query, chunk_ids=chunk_ids)
            for record in result:
                if record["relationship"]:
                    labels = [l for l in record["target_labels"] if l != "Resource"] if record["target_labels"] else []
                    label_str = labels[0] if labels else "Entity"
                    graph_facts.append({
                        "source_chunk": record["chunk_id"],
                        "relationship": record["relationship"],
                        "target_type": label_str,
                        "target_name": record["target_name"]
                    })
        return graph_facts

    def assemble_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Performs hybrid retrieval and builds graph-augmented LLM context."""
        logger.info(f"Assembling Graph RAG context for query: '{query}'")

        # 1. Retrieve hybrid chunks
        top_chunks = self.hybrid_retriever.search(query, top_k=top_k)
        chunk_ids = [c["chunk_id"] for c in top_chunks if "chunk_id" in c]

        # 2. Fetch connected graph relationships
        graph_facts = self.fetch_graph_subgraph(chunk_ids)

        # 3. Format plain-text chunk context
        text_context_blocks = []
        for idx, chunk in enumerate(top_chunks, start=1):
            source_file = chunk.get("file_name", "Unknown")
            page_info = f" (Page/Row {chunk['page']})" if chunk.get("page") is not None else ""
            content = chunk.get("page_content") or chunk.get("text", "")
            rrf_score = chunk.get("rrf_score", 0.0)

            block = f"[Doc {idx}] Source: {source_file}{page_info} | Score: {rrf_score}\nText: {content}"
            text_context_blocks.append(block)

        formatted_text_context = "\n\n".join(text_context_blocks)

        # 4. Format knowledge graph context
        graph_context_blocks = []
        for fact in graph_facts:
            fact_str = f"-(:{fact['relationship']})-> ({fact['target_type']}: {fact['target_name']})"
            graph_context_blocks.append(fact_str)

        formatted_graph_context = (
            "\n".join(set(graph_context_blocks))
            if graph_context_blocks
            else "No direct graph connections found."
        )

        # 5. Combined prompt context block
        prompt_context = (
            f"=== RETRIEVED TEXT CHUNKS ===\n"
            f"{formatted_text_context}\n\n"
            f"=== KNOWLEDGE GRAPH CONNECTIONS ===\n"
            f"{formatted_graph_context}"
        )

        return {
            "query": query,
            "top_chunks": top_chunks,
            "graph_facts": graph_facts,
            "formatted_text_context": formatted_text_context,
            "formatted_graph_context": formatted_graph_context,
            "prompt_context": prompt_context
        }

    def close(self) -> None:
        """Closes connections."""
        self.hybrid_retriever.close()
        if self.driver:
            self.driver.close()


def main() -> None:
    """Runs Step 9 Graph-Augmented RAG verification."""
    print("==================================================")
    print(" Starting Step 9 Graph RAG Context Verification")
    print("==================================================")

    context_builder = GraphRAGContextBuilder()

    try:
        context_builder.initialize_index()

        test_query = "Neo4j graph database nodes BM25"
        print(f"\n[+] Building context for query: '{test_query}'")

        context = context_builder.assemble_context(test_query, top_k=2)

        print("\n[+] Assembled Context Output Preview:")
        print("--------------------------------------------------")
        print(context["prompt_context"])
        print("--------------------------------------------------")

        print("\n[SUCCESS] Step 9 Graph RAG Context builder ready!")
    except Exception as e:
        print(f"\n[-] Step 9 execution failed: {e}")
    finally:
        context_builder.close()


if __name__ == "__main__":
    main()