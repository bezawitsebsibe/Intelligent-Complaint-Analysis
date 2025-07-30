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
   The filtered and cleaned dataset containing 850,403 complaints was saved as `data/filtered_complaints.csv` for the next steps.

## Summary

- The data spans from December 2011 to June 2025.
- The average complaint text length is approximately 464 characters, but some complaints are very long.
- Some columns contain missing values, especially in complaint narratives and company responses.
- This cleaned dataset will serve as the foundation for chunking, embedding, and building a retrieval system in later tasks.

---

## How to Run Task 1 Notebook

1. Make sure you have the full raw dataset `complaints.csv` in the `data/` folder.  
2. Open the Jupyter notebook: `notebooks/task1_eda_preprocessing.ipynb`  
3. Run all cells sequentially to reproduce the filtering and EDA steps.
