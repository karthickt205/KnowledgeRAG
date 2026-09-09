"""Streamlit Web Application for Graph-Augmented RAG Pipeline."""

import sys
from pathlib import Path

# Ensure ROOT_DIR is in Python path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st
from config.settings import Settings
from src.step10_rag_chain import GraphRAGChain


# Page configuration
st.set_page_config(
    page_title="Graph RAG Explorer",
    page_icon="🕸️",
    layout="wide"
)

st.title("🕸️ Graph-Augmented RAG Explorer")
st.caption("Hybrid Search (BM25 + Neo4j Vector RRF) with Knowledge Graph Traversal")


@st.cache_resource(show_spinner=False)
def load_rag_chain():
    """Initializes and caches the Graph RAG chain and indexes."""
    settings = Settings()
    chain = GraphRAGChain(settings)
    chain.initialize()
    return chain


# Initialize RAG Pipeline
with st.spinner("Initializing Knowledge Graph & Hybrid Indices..."):
    try:
        rag_chain = load_rag_chain()
        st.sidebar.success("Neo4j & Hybrid Indices Connected!")
    except Exception as e:
        st.sidebar.error(f"Failed to initialize RAG Chain: {e}")
        st.stop()

# Sidebar Controls
st.sidebar.header(" Retrieval Configuration")
top_k = st.sidebar.slider("Top Chunks (Top-K)", min_value=1, max_value=10, value=3, step=1)
show_raw_context = st.sidebar.checkbox("Show Context & Graph Triples", value=True)

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Ask me anything grounded in your Neo4j knowledge graph.",
            "context": None
        }
    ]

# Display Existing Chat History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("context") and show_raw_context:
            with st.expander("🔍 View Retrieved Graph Context"):
                st.code(msg["context"]["prompt_context"], language="markdown")

# Handle User Query Input
if user_query := st.chat_input("Enter your question..."):
    # Render User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Generate Response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving hybrid context & traversing knowledge graph..."):
            try:
                # Custom top_k override for user flexibility
                context_data = rag_chain.context_builder.assemble_context(user_query, top_k=top_k)
                
                # Re-run LLM / fallback generator with configured top_k
                full_prompt = (
                    f"You are an expert AI Assistant answering user questions using retrieved document text "
                    f"and Knowledge Graph relationships.\n\n"
                    f"{context_data['prompt_context']}\n\nUser Question: {user_query}\nAnswer:"
                )

                if rag_chain.llm:
                    try:
                        response = rag_chain.llm.invoke(full_prompt)
                        answer_text = response.content if hasattr(response, "content") else str(response)
                    except Exception:
                        answer_text = rag_chain._fallback_answer(context_data)
                else:
                    answer_text = rag_chain._fallback_answer(context_data)

                # Render Answer
                st.markdown(answer_text)

                # Render Context Accordion
                if show_raw_context:
                    with st.expander("🔍 View Retrieved Graph Context"):
                        st.text("=== RETRIEVED TEXT CHUNKS ===")
                        for idx, chunk in enumerate(context_data["top_chunks"], start=1):
                            st.write(f"**Chunk #{idx}** (RRF Score: `{chunk.get('rrf_score', 0.0)}`)")
                            st.info(chunk.get("page_content") or chunk.get("text", ""))

                        st.text("=== KNOWLEDGE GRAPH CONNECTIONS ===")
                        if context_data["graph_facts"]:
                            for fact in context_data["graph_facts"]:
                                st.code(
                                    f"({fact['source_chunk']}) -[:{fact['relationship']}]-> ({fact['target_type']}: {fact['target_name']})",
                                    language="cypher"
                                )
                        else:
                            st.write("No direct graph connections found.")

                # Save Assistant Output
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "context": context_data
                })

            except Exception as err:
                st.error(f"Error processing query: {err}")