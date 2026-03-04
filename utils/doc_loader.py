"""Utilities for loading documents in various formats."""
"""Utilities for loading documents in various formats."""
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from typing import List


def load_documents(file_paths: List[str]):
    """Given a list of file paths, return a list of LangChain Document objects.
    Supported formats: .txt, .pdf, .csv
    """
    docs = []
    for path in file_paths:
        if path.lower().endswith(".pdf"):
            # use PyPDFLoader instead of UnstructuredPDFLoader to avoid
            # heavy unstructured package dependency
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf8")
        elif path.lower().endswith(".csv"):
            loader = CSVLoader(path)
        else:
            raise ValueError(f"Unsupported file extension for {path}")
        docs.extend(loader.load())
    return docs
