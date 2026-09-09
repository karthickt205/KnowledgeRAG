"""Application settings and environment configuration management."""

import logging
import sys
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory Setup
BASE_DIR = Path(__file__).resolve().parent.parent

class Settings(BaseSettings):
    """Application settings using Pydantic for environment variable parsing."""

    # Neo4j Settings
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_username: str = Field(default="neo4j", alias="NEO4J_USERNAME")
    neo4j_password: str = Field(..., alias="NEO4J_PASSWORD")

    # LLM Settings
    llm_api_key: str = Field(..., alias="LLM_API_KEY")
    llm_model: str = Field(default="gpt-4o-mini", alias="LLM_MODEL")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")

    # Retrieval Defaults
    default_top_k: int = Field(default=4, alias="DEFAULT_TOP_K")
    min_bm25_score: float = Field(default=0.1, alias="MIN_BM25_SCORE")

    # Paths
    data_dir: Path = BASE_DIR / "data"
    documents_dir: Path = BASE_DIR / "data" / "documents"
    processed_dir: Path = BASE_DIR / "data" / "processed"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configures structured logging across the application."""
    logger = logging.getLogger("knowledge_rag")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

logger = setup_logging()