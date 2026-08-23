import os
from datasets import load_dataset
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- Config (knobs at the top, easy to find and tune) ---
DATASET_NAME = "CShorten/ML-ArXiv-Papers"
NUM_PAPERS = 2000                 # slice size; raise later to scale up
CHUNK_SIZE = 800                  # characters per chunk
CHUNK_OVERLAP = 100               # overlap keeps context across chunk borders
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "data/chroma"       


def load_documents():
    """Load the dataset slice and turn each paper into a LangChain Document."""
    ds = load_dataset(DATASET_NAME, split="train")
    print("Columns:", ds.column_names)          # verify schema, never assume 
    ds = ds.select(range(min(NUM_PAPERS, len(ds))))

    docs = []
    for row in ds:
        title = (row.get("title") or "").strip()
        abstract = (row.get("abstract") or "").strip()
        if not abstract:
            continue                             # defensive: skip empty rows
        content = f"Title: {title}\n\nAbstract: {abstract}"
        docs.append(Document(page_content=content, metadata={"title": title}))

    print(f"Built {len(docs)} documents.")
    return docs


def chunk_documents(docs):
    """Split documents into overlapping chunks the embedder handles well."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],  # split on natural boundaries first
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks):
    """Embed chunks locally and persist them to Chroma on disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    print("Embedding + storing (slow part, ~1-3 min)...")
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Done. Vector store saved to '{PERSIST_DIR}'.")
    return vectordb


if __name__ == "__main__":
    os.makedirs(PERSIST_DIR, exist_ok=True)
    documents = load_documents()
    chunks = chunk_documents(documents)
    build_vector_store(chunks)