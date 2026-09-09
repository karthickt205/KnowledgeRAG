# Graph-Augmented RAG with Neo4j

A full-stack Python project for building a graph-augmented retrieval pipeline over local documents. It combines:

- document loading and chunking
- lexical BM25 retrieval
- Neo4j vector search
- reciprocal rank fusion (RRF)
- knowledge-graph context enrichment
- a Streamlit chat interface for Q&A

This project is organized as a step-by-step pipeline, from setup to final retrieval-augmented generation.

## Overview

The app ingests documents from the data folder, splits them into chunks, stores them in Neo4j, builds a BM25 index, creates vector embeddings, and retrieves relevant context using hybrid search. The final Graph RAG layer gathers both text chunks and graph relationships to provide richer grounded answers.

Core components:

- `app.py` — Streamlit user interface
- `config/settings.py` — environment and configuration
- `src/step01_setup.py` — environment and connectivity validation
- `src/step02_document_loader.py` — multi-format document loading
- `src/step03_chunking.py` — chunk generation and deterministic IDs
- `src/step04_neo4j_schema.py` — graph schema and indexes
- `src/step05_neo4j_ingestion.py` — chunk ingestion into Neo4j
- `src/step06_bm25_index.py` — lexical BM25 retriever
- `src/step07_vector_index.py` — Neo4j vector embeddings index
- `src/step08_hybrid_search.py` — BM25 + vector hybrid retrieval using RRF
- `src/step09_graph_rag.py` — graph-aware context assembly
- `src/step10_rag_chain.py` — final answer generation pipeline

## Features

- Supports CSV, PDF, TXT, and Markdown documents
- Builds deterministic chunk IDs for stable graph linking
- Creates Neo4j constraints and indexes
- Maintains idempotent ingestion for repeated runs
- Uses BM25 lexical search + vector similarity search
- Fuses both retrieval methods with Reciprocal Rank Fusion
- Pulls related graph facts for context enrichment
- Exposes a chat UI via Streamlit
- Falls back to deterministic answers if an LLM is unavailable

## Project Structure

```text
RAG/
├── app.py
├── requirment.txt
├── .env
├── config/
│   ├── __init__.py
│   └── settings.py
├── data/
│   ├── documents/
│   └── processed/
├── src/
│   ├── __init__.py
│   ├── step01_setup.py
│   ├── step02_document_loader.py
│   ├── step03_chunking.py
│   ├── step04_neo4j_schema.py
│   ├── step05_neo4j_ingestion.py
│   ├── step06_bm25_index.py
│   ├── step07_vector_index.py
│   ├── step08_hybrid_search.py
│   ├── step09_graph_rag.py
│   └── step10_rag_chain.py
└── README.md
```

## Requirements

- Python 3.10+
- Neo4j instance running and reachable
- OpenAI-compatible API key or a local LLM backend
- Internet access for package installation and model usage if applicable

## Setup

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

- Windows (PowerShell):

```powershell
.\venv\Scripts\Activate.ps1
```

- Windows (Command Prompt):

```cmd
venv\Scripts\activate.bat
```

3. Install dependencies:

```bash
pip install -r requirment.txt
```

4. Configure environment variables in a `.env` file:

```env
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password
LLM_API_KEY=your_api_key
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
DEFAULT_TOP_K=4
MIN_BM25_SCORE=0.1
```

> Make sure your Neo4j database is available before running the ingestion and search steps.

## Run the Application

Start the Streamlit app from the project root:

```bash
streamlit run app.py
```

The app loads the Graph RAG chain, connects to Neo4j, and opens a chat UI that allows you to ask questions grounded in your indexed documents and graph context.

## Pipeline Execution

You can run the pipeline stages in order to initialize and verify each component:

```bash
python src/step01_setup.py
python src/step02_document_loader.py
python src/step03_chunking.py
python src/step04_neo4j_schema.py
python src/step05_neo4j_ingestion.py
python src/step06_bm25_index.py
python src/step07_vector_index.py
python src/step08_hybrid_search.py
python src/step09_graph_rag.py
python src/step10_rag_chain.py
```

## Data Folder

Place your source documents in the `data/documents` directory. Supported types include:

- `.csv`
- `.pdf`
- `.txt`
- `.md`

The project then processes and chunks these documents for indexing and graph enrichment.

## How Retrieval Works

1. Documents are loaded and split into chunks.
2. Chunks are stored as `Chunk` nodes in Neo4j.
3. A lexical BM25 index is built over the chunk content.
4. Embeddings are generated for each chunk and stored in a Neo4j vector index.
5. A search query is executed in both engines.
6. Results are combined with reciprocal rank fusion.
7. Related graph nodes and relationships are retrieved from Neo4j.
8. The final prompt includes both text chunks and graph context for answer generation.

## Notes

- The app is designed for local knowledge-base exploration and graph-grounded answering.
- If no LLM API key is provided, the project gracefully falls back to a deterministic summary based on retrieved context.
- Some steps may require a running Neo4j instance with sufficient permissions to create indexes and constraints.

## License

This project is intended for local experimentation and internal use. Add your preferred license if you plan to distribute it.
