"""Functions to create and persist FAISS index from documents."""
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from typing import List

import settings


def create_vector_store(docs: List[Document], persist_directory: str = None):
    """Create or load a FAISS vector store from provided docs.

    If the directory already contains a store, it will be loaded and the new
    documents added to it.
    """
    if persist_directory is None:
        persist_directory = settings.DATA_DIR

    # split documents into chunks to improve retrieval
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(docs)

    # choose embeddings based on LLM type and available credentials.
    # if using OpenAI we require an API key; otherwise fall back to a
    # HuggingFace model which works locally.
    if settings.LLM_TYPE == "openai" or os.environ.get("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    else:
        # fall back to local HuggingFace embeddings (no key needed)
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if os.path.exists(os.path.join(persist_directory, "index.faiss")):
        # loading a previously saved FAISS index uses pickle under the hood
        # which may execute arbitrary code if the file is tampered with. we
        # enable the flag because the index lives in our local data directory
        # and we assume it is trusted. do not enable this for untrusted sources.
        store = FAISS.load_local(
            persist_directory, embeddings,
            allow_dangerous_deserialization=True,
        )
        store.add_documents(docs)
    else:
        # from_documents does not accept persist_directory; create then save separately
        store = FAISS.from_documents(docs, embeddings)

    store.save_local(persist_directory)
    return store


def get_retriever(k: int = None):
    """Return a FAISS retriever with top k results."""
    if k is None:
        k = settings.TOP_K
    # same embedding logic as above; retrieve with whatever we created earlier
    if settings.LLM_TYPE == "openai" or os.environ.get("OPENAI_API_KEY"):
        embeddings = OpenAIEmbeddings()
    else:
        from langchain.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # if the index file doesn't exist yet, return a dummy retriever
    index_path = os.path.join(settings.DATA_DIR, "index.faiss")
    if not os.path.exists(index_path):
        class DummyRetriever:
            def get_relevant_documents(self, query):
                return []
        return DummyRetriever()

    store = FAISS.load_local(
        settings.DATA_DIR, embeddings,
        allow_dangerous_deserialization=True,
    )
    return store.as_retriever(search_kwargs={"k": k})
