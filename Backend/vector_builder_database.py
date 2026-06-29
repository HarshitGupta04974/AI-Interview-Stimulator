import chromadb
from chromadb.utils import embedding_functions
from Questions import RAW


# def build_vector_database():
#     # 1. Initialize the ChromaDB client (Persistent storage)
#     # This creates a folder 'question_vectors' to save the data
#     client = chromadb.PersistentClient(path="./question_vectors")
#
#     # 2. Define the embedding function
#     # Using a standard open-source model that works well for technical text
#     model_name = "all-MiniLM-L6-v2"
#     emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
#
#     # 3. Create or get the collection
#     dsa_collection = client.get_or_create_collection(
#         name="dsa_questions",
#         embedding_function=emb_fn,
#         metadata={"hnsw:space": "cosine"}
#     )
#
#     core_collection = client.get_or_create_collection(
#         name="core_questions",
#         embedding_function=emb_fn,
#         metadata={"hnsw:space": "cosine"}
#     )
#
#     dsa_docs, dsa_meta, dsa_ids = [], [], []
#     core_docs, core_meta, core_ids = [], [], []
#
#     print(f"Processing {len(RAW)} question approaches...")
#
#     for entry in RAW:
#         combined_text = (
#             f"Question: {entry['question_text']}\n"
#             f"Approach Type: {entry['approach_label']}\n"
#             f"Explanation: {entry['approach_summary']}"
#         )
#
#         metadata = {
#             "question_id": entry["question_id"],
#             "subject": entry["subject"],
#             "topic": entry["topic"],
#             "difficulty": entry["difficulty"],
#             "is_optimal": entry["is_optimal"],
#             "approach_label": entry["approach_label"]
#         }
#
#         if entry["subject"] == "DSA":
#             dsa_docs.append(combined_text)
#             dsa_meta.append(metadata)
#             dsa_ids.append(f"{entry['question_id']}_{entry['approach_label']}")
#         else:
#             core_docs.append(combined_text)
#             core_meta.append(metadata)
#             core_ids.append(f"{entry['question_id']}_{entry['approach_label']}")
#
#     # 4. Add data to the collection in batches (Chroma handles the embedding generation automatically)
#     # We use a batch size to avoid memory issues if the dataset grows
#     batch_size = 100
#     for i in range(0, len(dsa_docs), batch_size):
#         dsa_collection.add(
#             documents=dsa_docs[i:i + batch_size],
#             metadatas=dsa_meta[i:i + batch_size],
#             ids=dsa_ids[i:i + batch_size]
#
#         )
#
#     # Insert CORE
#     for i in range(0, len(core_docs), batch_size):
#         core_collection.add(
#             documents=core_docs[i:i + batch_size],
#             metadatas=core_meta[i:i + batch_size],
#             ids=core_ids[i:i + batch_size]
#         )
#
#     print("DSA + Core vector DB built separately 🚀")


def query_example(query_text):
    client = chromadb.PersistentClient(path="./question_vectors")
    collection = client.get_collection(name="core_questions")

    # Search for the top 3 most relevant approaches
    results = collection.query(
        query_texts=[query_text],
        n_results=3,
        # Example filter: search only for 'Optimal' approaches in 'OS'
        # where={"$and": [{"subject": "OS"}, {"is_optimal": True}]}
    )

    return results


if __name__ == "__main__":
    # Build the DB
    # build_vector_database()

    # Test Search
    print("\n--- Test Search: 'How to handle thundering herd in socket programming?' ---")
    test_results = query_example("How to handle thundering herd in socket programming?")
    for i, doc in enumerate(test_results['documents'][0]):
        print(f"\nMatch {i + 1}:")
        print(doc)