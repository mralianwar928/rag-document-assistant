"""Interactive CLI for the RAG assistant."""

from src.rag_documentassistant.rag import ask

print("RAG Document Assistant — ask about the indexed ML papers. Ctrl+C to exit.\n")
while True:
    try:
        q = input("You: ").strip()
        if not q:
            continue
        print("\nAssistant:", ask(q), "\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break