"""Minimal eval suite: retrieval hit-rate + grounded-refusal check.
Run AFTER ingest. Prints a small report you can paste into the README.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


from src.rag_documentassistant.rag import retriever, ask

# --- Test set: questions where we KNOW a relevant paper exists in the corpus.
# After ingesting, open a few papers you indexed and write questions about them.
RETRIEVAL_TESTS = [
    # {"question": "...", "expect_title_contains": "keyword from a real indexed title"},
]

# Questions the corpus can NOT answer — the system must refuse, not invent.
REFUSAL_TESTS = [
    "What is the capital of France?",
    "Who won the football world cup?",
]

REFUSAL_STRING = "couldn't find this in the indexed papers"


def eval_retrieval():
    if not RETRIEVAL_TESTS:
        print("! Add retrieval tests after inspecting your indexed papers.")
        return
    hits = 0
    for t in RETRIEVAL_TESTS:
        docs = retriever.invoke(t["question"])
        titles = " | ".join(d.metadata.get("title", "") for d in docs).lower()
        hit = t["expect_title_contains"].lower() in titles
        hits += hit
        print(("PASS" if hit else "FAIL"), "-", t["question"])
    print(f"\nRetrieval hit rate: {hits}/{len(RETRIEVAL_TESTS)}")


def eval_refusal():
    correct = 0
    for q in REFUSAL_TESTS:
        answer = ask(q).lower()
        ok = REFUSAL_STRING in answer
        correct += ok
        print(("PASS" if ok else "FAIL"), "- refuses:", q)
    print(f"\nGrounded refusal rate: {correct}/{len(REFUSAL_TESTS)}")


if __name__ == "__main__":
    print("== Retrieval evals ==")
    eval_retrieval()
    print("\n== Refusal evals ==")
    eval_refusal()