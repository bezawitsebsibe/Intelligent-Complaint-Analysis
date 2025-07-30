# Intelligent-Complaint-Analysis

# Task 1: Exploratory Data Analysis (EDA) and Preprocessing

## Overview

In Task 1, we worked with the large CFPB consumer complaints dataset (~5GB) to prepare it for further analysis and building an AI complaint assistant.

## Steps Completed

1. **Data Loading in Chunks**  
   Since the dataset is very large, we loaded it in smaller chunks (100,000 rows at a time) to avoid memory issues.

2. **Filtering Relevant Products**  
   We focused only on 5 key financial product categories to keep our analysis manageable:
   - Credit card  
   - Payday loan  
   - Mortgage  
   - Student loan  
   - Bank account or service  

3. **Data Cleaning and Selection**  
   We kept only the most relevant columns needed for analysis and model building, such as:
   - Date received  
   - Product  
   - Issue  
   - Consumer complaint narrative  
   - Company  
   - Submitted via  
   - Company response  
   - Timely response  
   - Complaint ID  

4. **Exploratory Data Analysis (EDA)**  
   We checked for missing values, distribution of complaints by product type, complaint text lengths, and date ranges to understand the data quality and characteristics.

5. **Saving Cleaned Data**  
   The filtered and cleaned dataset containing approximately 850,403 complaints was saved as `data/filtered_complaints.csv` for the next steps.

## Summary

- The data spans from December 2011 to June 2025.  
- The average complaint text length is approximately 464 characters, but some complaints are very long.  
- Some columns contain missing values, especially in complaint narratives and company responses.  
- This cleaned dataset serves as the foundation for chunking, embedding, and building a retrieval system in later tasks.

---

## How to Run Task 1 Notebook

1. Make sure you have the full raw dataset `complaints.csv` in the `data/` folder.  
2. Open the Jupyter notebook: `notebooks/task1_eda_preprocessing.ipynb`  
3. Run all cells sequentially to reproduce the filtering and EDA steps.

---

# Task 2: Chunking, Embedding, and Vector Store Construction

## Overview

In Task 2, we prepared the cleaned complaint texts for retrieval by chunking long complaint narratives, generating embeddings, and building a vector store with FAISS for efficient similarity search.

## Steps Completed

1. **Text Chunking**  
   Long complaint narratives were split into manageable chunks of about 500 tokens to facilitate effective embedding.

2. **Embedding Generation**  
   We used the `all-MiniLM-L6-v2` SentenceTransformer model to generate dense vector embeddings for each text chunk.

3. **FAISS Vector Store Construction**  
   The embeddings were indexed using FAISS for fast approximate nearest neighbor search.

4. **Metadata Storage**  
   Along with the index, we saved metadata containing the original text chunks for retrieval during query time.

5. **Saving Artifacts**  
   The FAISS index was saved as `vector_store/faiss_index.bin` and the metadata was saved as `vector_store/metadata.pkl`.

## Summary

- Processed around 850,403 complaints from Task 1 were chunked into approximately [insert number if known] chunks.  
- Embeddings were generated and stored efficiently for retrieval.  
- These artifacts enable retrieval-augmented generation (RAG) in subsequent tasks.

---

## How to Run Task 2 Notebook

1. Ensure the cleaned complaints CSV `data/filtered_complaints.csv` is present.  
2. Open the notebook: `notebooks/task2_chunk_embed_faiss.ipynb`  
3. Run all cells to generate chunks, embeddings, build FAISS index, and save metadata.

---

# Task 3: Retrieval-Augmented Generation (RAG) Model Integration

## Overview

Task 3 involved building a RAG pipeline that integrates our vector store with a generative language model to answer user questions based on retrieved complaint excerpts.

## Steps Completed

1. **Loading FAISS Index and Metadata**  
   Loaded the saved FAISS index and corresponding metadata for similarity search.

2. **Embedding User Queries**  
   Encoded user questions with the same SentenceTransformer model to find relevant complaint chunks.

3. **Prompt Construction**  
   Created prompts by combining retrieved chunks with the user query for the generative model.

4. **Generation Model Setup**  
   Integrated GPT-2 or Flan-T5 models using Hugging Face transformers pipeline for text generation.

5. **Answer Generation**  
   Generated answers based on context, ensuring fallback to “not enough information” if context was insufficient.

6. **Testing and Validation**  
   Tested interactive CLI and Streamlit app interfaces with example questions.

## Summary

- Successfully combined retrieval and generation to create an intelligent complaint assistant.  
- Demonstrated accurate, context-aware answers for complex user queries.  
- Developed both CLI and web UI (Streamlit) for interaction.

---

## How to Run Task 3 Script

1. Ensure FAISS index (`vector_store/faiss_index.bin`) and metadata (`vector_store/metadata.pkl`) exist.  
2. Run the Python script: `python src/rag_pipeline.py` for CLI interaction.  
3. Alternatively, launch the Streamlit app: `streamlit run src/streamlit_app.py`.

---

# Task 4: Streamlit Application for User Interaction

## Overview

In Task 4, we created a user-friendly Streamlit web app to interact with the RAG pipeline, allowing users to ask questions and view generated answers with source excerpts.

## Steps Completed

1. **Resource Loading Optimization**  
   Cached loading of FAISS index, metadata, embedding, and generation models to improve performance.

2. **User Input Handling**  
   Added text input for questions and buttons for submitting queries and clearing history.

3. **Answer Display**  
   Presented generated answers clearly with expandable source excerpts for transparency.

4. **Session State Management**  
   Managed user query and answer history using Streamlit session state to support multiple questions per session.

5. **Deployment Ready**  
   Structured the app for easy deployment and local use.

## Summary

- The Streamlit UI enables smooth, interactive question answering.  
- Users can trace back answers to original complaint text chunks.  
- Efficient caching ensures responsive experience even with large models.

---

## How to Run Task 4 Streamlit App

1. Confirm the vector store files and model dependencies are in place.  
2. Run the app with: `streamlit run src/streamlit_app.py`  
3. Open the local URL in a browser to start interacting.

---

# Final Notes

- All tasks build upon each other starting from large-scale data handling to AI-powered user interaction.  
- The dataset used contains over 850,000 filtered complaints focused on key financial products.  
- The modular pipeline supports extension and integration with more advanced models or datasets.

---

