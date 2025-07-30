import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Paths to vector store
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

print("📦 Loading FAISS index and metadata...")
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

# We stored chunks under the key "chunk"
actual_key = "chunk"

print("📥 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("🤖 Loading generation pipeline (Flan-T5)...")
generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    tokenizer="google/flan-t5-base"
)

def rag_answer(question: str, top_k: int = 5) -> str:
    query_vec = embedder.encode([question])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    retrieved_chunks = [metadata[i][actual_key] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    # Flan-T5 style prompt
    prompt = (
        f"Answer the question based on the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}"
    )

    response = generator(prompt, max_new_tokens=200, truncation=True)
    return response[0]["generated_text"].strip()

if __name__ == "__main__":
    print("Type 'exit' to quit.")
    while True:
        user_question = input("\n❓ Ask a question: ")
        if user_question.lower() == "exit":
            break
        print("🔎 Thinking...")
        answer = rag_answer(user_question)
        print("\n💬 Answer:\n", answer)
