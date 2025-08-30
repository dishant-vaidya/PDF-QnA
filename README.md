# PDF Q&A App (Personal Academic Project)

---

# What this project does?

This is a simple "Streamlit web app" that allows users to:

1. Upload a PDF file (e.g., reports, notes, books, research papers).
2. Ask natural language questions about the content of the PDF.
3. Get detailed answers generated using "Retrieval-Augmented Generation (RAG)" with FAISS and a local language model.

The goal of this project is to make information inside PDFs quickly searchable and more understandable without manually reading the entire file.

---

# Tools & Why They Were Used

- **Streamlit** → For building the interactive web app with minimal code.

- **PyPDF2** → To extract raw text from uploaded PDF files.

- **SentenceTransformers (all-MiniLM-L6-v2)** → To create embeddings (vector representations) of text chunks for semantic search.

- **FAISS (Facebook AI Similarity Search)** → To store embeddings and quickly retrieve the most relevant PDF chunks for a given question.

- **Hugging Face Transformers (FLAN-T5 model)** → A free, open-source language model that generates natural language answers using the retrieved PDF context.

- **PyTorch** → Backend framework for running the Hugging Face model efficiently.

- **NumPy** → Used for vector normalization and general array operations.

These tools together form a lightweight **RAG pipeline** (Retrieve → Read → Answer) without needing any paid APIs.

---

To run the app, install the required dependencies mentioned in requirements.txt, open the window's command prompt from the file's location, and use the command: 

**streamlit run PDF_Q&A.py**

---

# Note on Model Download

- The first time you run the app, the Hugging Face FLAN-T5 model weights (~1 GB) will be automatically downloaded. 
- Make sure you have a stable internet connection and enough disk space. 
- Subsequent runs will use the cached model, so they will start faster.

---

# Disclaimer

- This app is a college project created for educational purposes.

- While it uses advanced AI models to answer questions based on the content of a PDF, it may sometimes:

1. Give incomplete or incorrect answers

2. Hallucinate information not present in the PDF

3. Struggle with very large or image-based PDFs (since OCR is not included)

- This happens because:

* The underlying model (FLAN-T5) is trained to generate fluent text but does not always guarantee factual correctness.

* The retrieval process selects only the top few chunks from the PDF, so the model might miss relevant context if it is outside those chunks.

* PDF text extraction may fail if the PDF has complex formatting, images, or scanned text.

# Always verify the answers with the original PDF. This tool is not suitable for legal, medical, or financial decision-making.
