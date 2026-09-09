"""Step 1: Configuration initialization, environment validation, and system connectivity checks."""

import sys
from pathlib import Path

# Add project root directory to python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from neo4j import GraphDatabase, exceptions
from langchain_openai import ChatOpenAI
from config.settings import Settings, logger

def validate_environment() -> Settings:
    """Loads and validates all required environment variables."""
    print("[+] Validating environment settings...")
    try:
        settings = Settings()
        print("[+] Environment configuration successfully validated.")
        return settings
    except Exception as e:
        print(f"[-] Environment validation failed: {e}")
        raise

def test_neo4j_connection(settings: Settings) -> bool:
    """Validates connectivity to the Neo4j instance."""
    print(f"[+] Connecting to Neo4j instance at {settings.neo4j_uri}...")
    try:
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password)
        )
        driver.verify_connectivity()
        driver.close()
        print("[+] Neo4j connection test successful.")
        return True
    except exceptions.DriverError as e:
        print(f"[-] Failed to connect to Neo4j database: {e}")
        return False
    except Exception as e:
        print(f"[-] Unexpected error during Neo4j check: {e}")
        return False

def test_llm_connection(settings: Settings) -> bool:
    """Validates connectivity to the configured LLM API provider."""
    print(f"[+] Initializing LLM provider test using model: {settings.llm_model}...")
    try:
        llm = ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            temperature=0
        )
        response = llm.invoke("Ping")
        if response and response.content:
            print("[+] LLM Provider API test successful.")
            return True
        print("[!] LLM API returned an empty response.")
        return False
    except Exception as e:
        print(f"[-] LLM API connectivity check failed: {e}")
        return False

def init_workspace(settings: Settings) -> None:
    """Creates initial workspace directory structures."""
    print("[+] Initializing workspace directories...")
    settings.data_dir.mkdir(exist_ok=True, parents=True)
    settings.documents_dir.mkdir(exist_ok=True, parents=True)
    settings.processed_dir.mkdir(exist_ok=True, parents=True)
    print("[+] Directory structure verified.")

def main() -> None:
    """Runs Step 1 setup verification pipeline."""
    print("==================================================")
    print(" Starting System Setup & Connections Validation")
    print("==================================================")
    try:
        settings = validate_environment()
        init_workspace(settings)
        
        neo4j_ok = test_neo4j_connection(settings)
        llm_ok = test_llm_connection(settings)

        if neo4j_ok and llm_ok:
            print("\n[SUCCESS] Step 1 Setup complete. All checks passed!")
        else:
            print("\n[WARNING] Step 1 Setup completed with warnings. Check logs above.")
    except Exception as e:
        print(f"\n[CRITICAL] Step 1 initialization aborted: {e}")

if __name__ == "__main__":
    main()