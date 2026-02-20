from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings

DATA_DIR = "data/hr_docs"
DB_DIR = "db"

def ingest():
    loader = DirectoryLoader(
        DATA_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_DIR
    )
    vectordb.persist()
    print(f"✅ Ingested {len(chunks)} chunks into Chroma at '{DB_DIR}'")

if __name__ == "__main__":
    ingest()