"""Query pipeline: question -> retrieve top-k chunks -> grounded answer with sources."""
import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()                                    # reads GROQ_API_KEY from .env

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERSIST_DIR = "data/chroma"
CHAT_MODEL = "openai/gpt-oss-20b"
TOP_K = 4

# --- 1. Load the persisted vector store (NO re-embedding of the corpus) ---
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)   # must match ingest!
vectordb = Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

# --- 2. Prompt: ground the model in retrieved context ---
SYSTEM = """You are a research assistant answering questions about machine learning papers.
Answer ONLY using the context below. Each context block starts with its paper title.
Cite the paper title(s) you used in your answer.
If the context does not contain the answer, say exactly:
"I couldn't find this in the indexed papers." Do not invent information."""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM + "\n\nContext:\n{context}"),
    ("human", "{question}"),
])


def format_docs(docs):
    """Turn retrieved Documents into one context string, titles included."""
    return "\n\n---\n\n".join(
        f"[{d.metadata.get('title', 'Unknown')}]\n{d.page_content}" for d in docs
    )


# --- 3. The chain (LCEL): retrieve -> format -> prompt -> LLM -> string ---
llm = ChatGroq(model=CHAT_MODEL, temperature=0)   # temperature 0: factual, reproducible

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


def ask(question: str) -> str:
    return chain.invoke(question)


if __name__ == "__main__":
    print(ask("What approaches exist for image classification in these papers?"))