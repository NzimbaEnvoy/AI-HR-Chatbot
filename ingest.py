import os
import hashlib
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = "data/hr_docs"
DB_DIR = "db"
COLLECTION = "hr_policies"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def infer_doc_type(source_path: str) -> str:
    """
    Tags docs so we can do hybrid retrieval (FAQ first, then policies).
    """
    p = source_path.replace("\\", "/").lower()
    fname = os.path.basename(p)

    if "faq" in p or fname.startswith("faq"):
        return "faq"

    # folder-based tagging if you use structure like data/hr_docs/benefits/...
    for key in ["benefits", "policies", "procedures", "onboarding", "forms", "payroll", "remote"]:
        if f"/{key}/" in p:
            return key

    return "policy"


def load_all_docs() -> List:
    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(DATA_DIR, glob="**/*.md", loader_cls=TextLoader, show_progress=True),
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True),
        DirectoryLoader(DATA_DIR, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, show_progress=True),
    ]

    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
        except Exception as e:
            print(f"⚠️ Skipped a loader due to error: {e}")

    # Attach metadata (source, hash, doc_type)
    for d in docs:
        src = d.metadata.get("source", "unknown")
        d.metadata["doc_type"] = infer_doc_type(src)
        if os.path.exists(src):
            d.metadata["sha256"] = sha256_file(src)

    return docs


def ingest():
    docs = load_all_docs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=180,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    vectordb = Chroma(
        collection_name=COLLECTION,
        persist_directory=DB_DIR,
        embedding_function=embeddings,
    )

    vectordb.add_documents(chunks)
    vectordb.persist()

    print(f"✅ Ingested {len(chunks)} chunks into Chroma at '{DB_DIR}' (collection='{COLLECTION}')")


if __name__ == "__main__":
    ingest()