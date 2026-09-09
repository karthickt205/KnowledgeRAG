"""Step 2: Robust Multi-Format Document Loader supporting PDF, CSV, TXT, and Markdown files."""

import sys
import warnings
from pathlib import Path
from typing import List, Dict, Any, Optional

warnings.filterwarnings("ignore", category=DeprecationWarning)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from langchain_core.documents import Document
from config.settings import Settings, logger


class DocumentLoader:
    """Loads documents from supported formats (.pdf, .csv, .txt, .md)."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.supported_extensions = {".pdf", ".csv", ".txt", ".md"}

    @property
    def target_docs_dir(self) -> Path:
        """Safely gets the document directory attribute from Settings."""
        if hasattr(self.settings, "documents_dir"):
            return Path(self.settings.documents_dir)
        elif hasattr(self.settings, "docs_dir"):
            return Path(self.settings.docs_dir)
        elif hasattr(self.settings, "data_dir"):
            return Path(self.settings.data_dir) / "documents"
        return ROOT_DIR / "data" / "documents"

    def load_single_file(self, file_path: Path) -> List[Document]:
        """Loads a single file into a list of LangChain Document objects."""
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            return []

        ext = file_path.suffix.lower()
        if ext not in self.supported_extensions:
            logger.warning(f"Unsupported file extension '{ext}': {file_path}")
            return []

        docs: List[Document] = []
        logger.info(f"Loading document [{ext.replace('.', '')}]: {file_path.name}")

        try:
            if ext == ".csv":
                try:
                    from langchain_community.document_loaders.csv_loader import CSVLoader
                    loader = CSVLoader(file_path=str(file_path), encoding="utf-8")
                    docs = loader.load()
                except Exception:
                    # Fallback for CSV files with different encodings
                    import csv
                    with open(file_path, mode="r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        for idx, row in enumerate(reader, start=1):
                            text_content = "\n".join([f"{k}: {v}" for k, v in row.items() if v])
                            docs.append(Document(
                                page_content=text_content,
                                metadata={"source": str(file_path), "file_name": file_path.name, "row": idx}
                            ))

            elif ext == ".pdf":
                try:
                    from langchain_community.document_loaders import PyPDFLoader
                    loader = PyPDFLoader(str(file_path))
                    docs = loader.load()
                except Exception as pdf_err:
                    logger.warning(f"PyPDFLoader failed ({pdf_err}). Trying text fallback.")
                    docs = [Document(page_content=file_path.read_text(encoding="utf-8", errors="ignore"))]

            elif ext in {".txt", ".md"}:
                from langchain_community.document_loaders import TextLoader
                loader = TextLoader(str(file_path), encoding="utf-8")
                docs = loader.load()

            # Assign standard metadata
            for doc in docs:
                doc.metadata["file_name"] = file_path.name
                doc.metadata["file_type"] = ext.replace(".", "")
                if "source" not in doc.metadata:
                    doc.metadata["source"] = str(file_path)

            logger.info(f"Successfully loaded {len(docs)} units/pages from {file_path.name}")
            return docs

        except Exception as e:
            logger.error(f"Error loading file '{file_path.name}': {e}")
            return []

    def load_directory(self, dir_path: Optional[Path] = None) -> List[Document]:
        """Loads all supported documents from the configured directory."""
        target_dir = dir_path or self.target_docs_dir
        if not target_dir.exists():
            logger.warning(f"Directory does not exist: {target_dir}")
            return []

        all_docs: List[Document] = []
        for file_path in target_dir.glob("*"):
            if file_path.is_file() and file_path.suffix.lower() in self.supported_extensions:
                docs = self.load_single_file(file_path)
                all_docs.extend(docs)

        logger.info(f"Total documents loaded from {target_dir}: {len(all_docs)}")
        return all_docs


# Alias for backward compatibility with downstream modules expecting KnowledgeDocumentLoader
KnowledgeDocumentLoader = DocumentLoader


def main() -> None:
    """Runs Step 2 document loader verification on your actual CSV files."""
    print("==================================================")
    print(" Starting Step 2 Document Loader Verification")
    print("==================================================")

    loader = DocumentLoader()
    docs_dir = loader.target_docs_dir
    docs_dir.mkdir(parents=True, exist_ok=True)

    loaded_docs = loader.load_directory(docs_dir)

    print(f"\n[+] Target Directory: {docs_dir}")
    print(f"[+] Total Loaded Document Objects: {len(loaded_docs)}")

    if loaded_docs:
        print("\n--- Sample Loaded Document Metadata & Content ---")
        print(f"Source File: {loaded_docs[0].metadata.get('file_name')}")
        print(f"Content Preview:\n{loaded_docs[0].page_content[:300]}")
    else:
        print(f"\n[!] No supported files (.csv, .pdf, .txt, .md) found in {docs_dir}")
        print("Please place your custom CSV file in that folder and run again.")

    print("\n[SUCCESS] Step 2 Document Loader verification complete!")


if __name__ == "__main__":
    main()