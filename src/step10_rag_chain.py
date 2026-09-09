"""Step 10: End-to-End Graph-Augmented RAG Chain Generation."""

import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config.settings import Settings, logger
from src.step09_graph_rag import GraphRAGContextBuilder


def get_llm_chain(settings: Settings):
    """Instantiates available LLM provider or returns fallback generator."""
    try:
        from langchain_openai import ChatOpenAI
        if settings.openai_api_key:
            logger.info("Using OpenAI ChatOpenAI model...")
            return ChatOpenAI(api_key=settings.openai_api_key, model="gpt-3.5-turbo", temperature=0.2)
    except Exception:
        pass

    try:
        from langchain_community.chat_models import ChatOllama
        logger.info("Attempting Ollama local ChatModel connection...")
        return ChatOllama(model="llama3", temperature=0.2)
    except Exception:
        pass

    logger.info("Using deterministic fallback generator...")
    return None


class GraphRAGChain:
    """Executes full Graph RAG context retrieval and LLM answer generation."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.context_builder = GraphRAGContextBuilder(self.settings)
        self.llm = get_llm_chain(self.settings)

    def initialize(self) -> None:
        """Initializes indices for hybrid search and Neo4j graph context."""
        self.context_builder.initialize_index()

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generates grounded answer using Graph RAG context."""
        context_data = self.context_builder.assemble_context(query, top_k=2)
        prompt_context = context_data["prompt_context"]

        system_prompt = (
            "You are an expert AI Assistant answering user questions using retrieved document text "
            "and Knowledge Graph relationships. Answer accurately and cite key facts from the context."
        )

        full_prompt = f"{system_prompt}\n\n{prompt_context}\n\nUser Question: {query}\nAnswer:"

        if self.llm:
            try:
                response = self.llm.invoke(full_prompt)
                answer_text = response.content if hasattr(response, "content") else str(response)
            except Exception as err:
                logger.warning(f"LLM invocation failed ({err}). Falling back to context summary.")
                answer_text = self._fallback_answer(context_data)
        else:
            answer_text = self._fallback_answer(context_data)

        return {
            "query": query,
            "answer": answer_text,
            "context": context_data
        }

    @staticmethod
    def _fallback_answer(context_data: Dict[str, Any]) -> str:
        """Generates structured answer directly from context when no live LLM API key is present."""
        top_chunks = context_data.get("top_chunks", [])
        graph_facts = context_data.get("graph_facts", [])

        if not top_chunks:
            return "No relevant information found in the knowledge base."

        summary_lines = ["Based on the retrieved Graph-RAG knowledge base:\n"]
        for idx, chunk in enumerate(top_chunks, start=1):
            text = chunk.get("page_content") or chunk.get("text", "")
            summary_lines.append(f"{idx}. {text.strip()}")

        if graph_facts:
            summary_lines.append("\nConnected Graph Entities:")
            for fact in graph_facts:
                summary_lines.append(f" - {fact['source_chunk']} -[{fact['relationship']}]-> {fact['target_type']}: {fact['target_name']}")

        return "\n".join(summary_lines)

    def close(self) -> None:
        """Closes connections."""
        self.context_builder.close()


def main() -> None:
    """Runs Step 10 Graph RAG Chain verification."""
    print("==================================================")
    print(" Starting Step 10 Graph RAG Chain Verification")
    print("==================================================")

    rag_chain = GraphRAGChain()

    try:
        rag_chain.initialize()

        test_query = "What is Neo4j and how does BM25 lexical search work?"
        print(f"\n[+] Executing Full RAG Chain Query: '{test_query}'")

        result = rag_chain.generate_response(test_query)

        print("\n[+] Generated RAG Answer:")
        print("--------------------------------------------------")
        print(result["answer"])
        print("--------------------------------------------------")

        print("\n[SUCCESS] Step 10 End-to-End Graph RAG pipeline complete!")
    except Exception as e:
        print(f"\n[-] Step 10 execution failed: {e}")
    finally:
        rag_chain.close()


if __name__ == "__main__":
    main()