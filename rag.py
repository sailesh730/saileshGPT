from pathlib import Path
import certifi
from dotenv import load_dotenv
import os
import docx2txt

load_dotenv()
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from langchain_huggingface import HuggingFaceEmbeddings


Path("upload").mkdir(exist_ok=True)
Path("chroma-db").mkdir(exist_ok=True)

embbeding =HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"

)

vector_store = Chroma(
    embedding_function=embbeding,
    collection_name="agentic_chatbot_docs",
    persist_directory="chroma-db"

)

def read_file_path(file_path:str):
    path =Path(file_path)
    suffix = path.suffix.lower()


    if suffix == ".pdf":
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n\n"
        return text
    if suffix == ".docx":
        return docx2txt.process(file_path)
    if suffix in [".txt",".csv",".py",".md"]:
        return path.read_text(encoding="utf-8",errors="ignore")
    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD, PY, or CSV files.")


def add_document_to_rag(file_path:str,thread_id:str,source_name:str|None=None):
    text = read_file_path(file_path)
    if not text.strip():
        raise ValueError("No readable text could be extracted from this file.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
    )
    chunks = splitter.split_text(text)

    docs :list[Document] = [
        Document(
            page_content=chunk,
            metadata = {
                "thread_id":thread_id,
                "source":source_name or Path(file_path).name
            }

        )

        for chunk in chunks

 ]

    vector_store.add_documents(docs)
    return {
        "filename":source_name or Path(file_path).name,
        "chunks":len(docs)
    }





def retrive_from_rag(query:str,thread_id:str,k:int=4) ->str:
    docs = vector_store.similarity_search(
        query,
        k=k,
        filter={"thread_id":thread_id}

    )

    if not docs:
        return "No relevant content found in the uploaded documents."
    result = []
    for i,doc in enumerate(docs,start=1):
        source = doc.metadata.get("source","uploaded document")
        result.append(
            f"[source {i} : {source}]\n\n {doc.page_content}"
        )
    return "\n\n".join(result)