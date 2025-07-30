import os
import pandas as pd
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
from tqdm import tqdm

# ✅ Load data
print("📥 Loading filtered complaints data...")
df = pd.read_csv("data/filtered_complaints.csv")

# ✅ Sample only 10,000 complaints to speed up
print("🔎 Sampling 10,000 complaints...")
df = df.sample(n=10000, random_state=42)

# ✅ Filter valid complaints
df = df[df["Consumer complaint narrative"].notnull()]
print(f"📊 Total complaints with text: {len(df)}")

# ✅ Load embedding model
print("🤖 Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ✅ Split complaints into chunks
print("🔪 Splitting complaints into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

all_chunks = []
all_metadata = []

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Chunking"):
    text = row["Consumer complaint narrative"]
    chunks = text_splitter.split_text(text)
    all_chunks.extend(chunks)
    all_metadata.extend([{"original_index": idx}] * len(chunks))

print(f"🧩 Total chunks created: {len(all_chunks)}")

# ✅ Embed in batches
print("⚡ Embedding chunks in batches...")
batch_size = 64
embedding_list = []

for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding"):
    batch = all_chunks[i:i+batch_size]
    embeddings = embedder.encode(batch, show_progress_bar=False)
    embedding_list.extend(embeddings)

embedding_matrix = np.array(embedding_list)
print(f"✅ Embedding complete. Shape: {embedding_matrix.shape}")

# ✅ Build FAISS index
print("📦 Building FAISS index...")
dimension = embedding_matrix.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embedding_matrix)

# ✅ Save index and metadata
os.makedirs("vector_store", exist_ok=True)
faiss.write_index(index, "vector_store/faiss_index.bin")

with open("vector_store/metadata.pkl", "wb") as f:
    pickle.dump(all_metadata, f)

print("✅ Task 2 complete: Chunking, embedding, and indexing done!")
