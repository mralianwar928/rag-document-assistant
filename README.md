# RAG Document Assistant

Ask questions about 2,000 machine-learning papers and get answers grounded in the actual
papers, with sources cited. When the answer isn't in the corpus, it says so rather than
guessing.

<!-- Add a terminal GIF of it answering here — this is the single most convincing thing. -->
<!-- ![demo](docs/demo.gif) -->

## How it works

Ingestion runs once when papers are chunked, embedded locally, and stored in Chroma. Each
query embeds the question, retrieves the closest chunks, and asks the LLM to answer using
only those, citing the paper titles. Grounding plus refusal on out-of-scope questions is
what separates this from a plain chatbot.

**Stack:** Python · LangChain · Groq (`openai/gpt-oss-20b`) · sentence-transformers (local embeddings) · ChromaDB

## Run it

Needs Python 3.11.

```bash
python -m venv .venv && .venv\Scripts\activate     # source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
# add GROQ_API_KEY to a .env file (free key at console.groq.com)

python -m src.rag_documentassistant.ingest         # build the index (~2-3 min, first run)
python -m src.rag_documentassistant.app            # ask questions
```

## Results

Retrieval and generation are evaluated separately (`python evals/run_evals.py`):



## Notes

Embeddings run locally, so only generation calls the API free and offline for the heavy
step, which is how teams keep RAG costs down. The chat model is a one-line config value, so
it's trivial to swap when a provider deprecates a model.
