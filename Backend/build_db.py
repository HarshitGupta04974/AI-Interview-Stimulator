"""
build_db.py
===========
Run once before starting the server:
    python build_db.py

Populates ChromaDB with all questions from question.py.
Safe to re-run — clears and rebuilds collections from scratch.
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Import the computed RAW list dynamically from your question.py file
from Questions import RAW
load_dotenv()

CHROMA_DB_PATH  = os.getenv("CHROMA_DB_PATH", "./question_vectors")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 50

def build():
    print(f"Connecting to ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL)

    # ── Wipe and recreate both collections ───────────────────────────────────
    for name in ("dsa_questions", "core_questions"):
        try:
            client.delete_collection(name)
            print(f"  Deleted existing collection: {name}")
        except Exception:
            pass

    dsa_col  = client.create_collection("dsa_questions",  embedding_function=emb_fn)
    core_col = client.create_collection("core_questions", embedding_function=emb_fn)
    print("  Created fresh collections: dsa_questions, core_questions")

    # ── Split RAW into DSA and core subjects ──────────────────────────────────
    dsa_rows  = [r for r in RAW if r["subject"] == "DSA"]
    core_rows = [r for r in RAW if r["subject"] != "DSA"]
    print(f"  DSA entries  : {len(dsa_rows)}")
    print(f"  Core entries : {len(core_rows)}")

    # ── Helper: upsert in batches ─────────────────────────────────────────────
    def upsert_batch(collection, rows):
        for start in range(0, len(rows), BATCH_SIZE):
            batch = rows[start:start + BATCH_SIZE]

            # Unique key generation using combined properties to avoid collision overwrites
            ids = [f"{r['question_id']}" for r in batch]

            documents = [
                f"Question: {r['question_text']}\n"
                f"Approach: {r['approach_label']}\n"
                f"Summary: {r['approach_summary']}"
                for r in batch
            ]

            metadatas = [
                {
                    "question_id":        r["question_id"],
                    # Extracts base name parent ID (e.g., "DSA_1") for filter constraints
                    "parent_question_id": "_".join(r["question_id"].split("_")[:2]),
                    "subject":            r["subject"],
                    "topic":              r["topic"],
                    "difficulty":         r["difficulty"],
                    "approach_label":     r["approach_label"],
                    "is_optimal":         r["is_optimal"],
                }
                for r in batch
            ]

            collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
            print(f"    Upserted {start + len(batch)}/{len(rows)}", end="\r")
        print()

    print("\nBuilding dsa_questions...")
    upsert_batch(dsa_col, dsa_rows)

    print("Building core_questions...")
    upsert_batch(core_col, core_rows)

    # ── Verify counts are populated ───────────────────────────────────────────
    print(f"\nVerification:")
    print(f"  dsa_questions  count : {dsa_col.count()}")
    print(f"  core_questions count : {core_col.count()}")
    print("\nDone. Run: uvicorn main:app --reload --port 8000")

if __name__ == "__main__":
    build()