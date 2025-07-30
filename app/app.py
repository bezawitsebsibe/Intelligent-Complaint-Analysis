import streamlit as st
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load FAISS index and metadata
VECTOR_STORE_DIR = "vector_store"
INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "faiss_index.bin")
METADATA_PATH = os.path.join(VECTOR_STORE_DIR, "metadata.pkl")

@st.cache_resource
def load_resources():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    generator = pipeline("text-generation", model="gpt2", max_new_tokens=256)
    return index, metadata, embedder, generator

index, metadata, embedder, generator = load_resources()

def rag_answer(question: str, top_k: int = 5):
    query_vec = embedder.encode([question])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    
    retrieved_chunks = [metadata[i]["chunk"] for i in I[0]]
    context = "\n\n".join(retrieved_chunks)

    prompt = f"""
You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.

Use the following retrieved complaint excerpts to formulate your answer.
If the context doesn't contain the answer, say you don't have enough information.

Context:
{context}

Question: {question}
Answer:
"""
    response = generator(prompt, do_sample=True, temperature=0.7)
    answer = response[0]["generated_text"].split("Answer:")[-1].strip()
    return answer, retrieved_chunks

# Streamlit UI
st.set_page_config(page_title="Complaint RAG Assistant", layout="centered")
st.title("💬 CFPB Complaint Assistant (RAG-based)")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("❓ Ask a question about financial complaints:")
submit = st.button("Ask")
clear = st.button("Clear")

if submit and question:
    with st.spinner("Thinking..."):
        try:
            answer, sources = rag_answer(question)
            st.session_state.history.append((question, answer, sources))
        except Exception as e:
            st.error(f"An error occurred: {e}")

if clear:
    st.session_state.history = []

# Display history
for q, a, s in reversed(st.session_state.history):
    st.markdown(f"**Q:** {q}")
    st.markdown(f"**A:** {a}")
    with st.expander("🔍 Sources used"):
        for chunk in s:
            st.markdown(f"- {chunk}")
    st.markdown("---")
