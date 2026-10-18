"""
rag.py - document upload + retrieval for the chatbot (Chroma + HuggingFace embeddings).

Public functions:
    add_document_to_rag(file_path, thread_id, source_name=None) -> {"filename", "chunks"}
    retrive_from_rag(query, thread_id, k=4) -> str     (old spelling, still used by tools.py)
    retrieve_from_rag                                  (same function, correct spelling)
    delete_thread_documents(thread_id) -> int          (call when a conversation is deleted)
"""

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import certifi
import docx2txt
from dotenv import load_dotenv

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_core.documents import Document  # noqa: E402
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from pypdf import PdfReader  # noqa: E402

UPLOAD_DIR = Path("upload")
CHROMA_DIR = Path("chroma-db")
UPLOAD_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "agentic_chatbot_docs"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
ADD_BATCH_SIZE = 500  # Chroma rejects very large single inserts

_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


@lru_cache(maxsize=1)
def get_vector_store():
    """Create the embedding model and vector store on first use, not at import time.

    This way a slow or failed model download cannot stop the whole backend from starting;
    the error only appears when someone uploads or searches a document.
    """
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(
        embedding_function=embedding,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_DIR),
    )


def read_file_path(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(file_path)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        return docx2txt.process(file_path)
    if suffix in (".txt", ".csv", ".py", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV files.")


def _delete_existing(store, thread_id: str, source: str) -> None:
    """Remove old chunks of the same file in this thread so re-uploads don't duplicate."""
    found = store.get(where={"$and": [{"thread_id": thread_id}, {"source": source}]})
    ids = found.get("ids") or []
    if ids:
        store.delete(ids=ids)


def add_document_to_rag(file_path: str, thread_id: str, source_name: Optional[str] = None) -> dict:
    text = read_file_path(file_path)
    if not text.strip():
        raise ValueError("No readable text could be extracted from this file.")

    source = source_name or Path(file_path).name
    chunks = _splitter.split_text(text)
    docs = [
        Document(page_content=chunk, metadata={"thread_id": thread_id, "source": source})
        for chunk in chunks
    ]

    store = get_vector_store()
    _delete_existing(store, thread_id, source)
    for start in range(0, len(docs), ADD_BATCH_SIZE):
        store.add_documents(docs[start:start + ADD_BATCH_SIZE])

    return {"filename": source, "chunks": len(docs)}


def retrive_from_rag(query: str, thread_id: str, k: int = 4) -> str:
    if not (query or "").strip():
        return "Please provide a search query."

    docs = get_vector_store().similarity_search(query, k=k, filter={"thread_id": thread_id})
    if not docs:
        return "No relevant content found in the uploaded documents."

    parts = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "uploaded document")
        parts.append(f"[source {i}: {source}]\n{doc.page_content}")
    return "\n\n".join(parts)


retrieve_from_rag = retrive_from_rag  # correctly spelled alias


def delete_thread_documents(thread_id: str) -> int:
    """Delete every chunk stored for a conversation. Returns how many were removed."""
    store = get_vector_store()
    found = store.get(where={"thread_id": thread_id})
    ids = found.get("ids") or []
    if ids:
        store.delete(ids=ids)
    return len(ids)