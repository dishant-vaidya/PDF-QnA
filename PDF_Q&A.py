#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
from io import BytesIO
import PyPDF2
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

st.set_page_config(page_title="Personal Project", layout="wide")

# Extracting text from PDF

def extract_text_from_pdf(data: bytes) -> str:
    reader = PyPDF2.PdfReader(BytesIO(data))
    texts = []
    for page in reader.pages:
        t = page.extract_text()
        if t:
            texts.append(t)
    return "\n".join(texts)

# Splitting text into overlapping chunks

def split_text(text, chunk_size=800, overlap=100):
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if len(chunk.strip()) > 50:
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

# Vector Embedding

def build_faiss_index(embeddings):
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings.astype(np.float32))
    return index

# Question Embedding for similarity

def retrieve(query, model, index, docs, k=4):
    q_emb = model.encode([query], convert_to_numpy=True)
    q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
    D, I = index.search(q_emb.astype(np.float32), k)
    return [docs[i] for i in I[0] if i >= 0]

# Loading model and tokenizer

@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    return embedder, tokenizer, model, device

embedder, tokenizer, gen_model, device = load_models()

# Streamlit UI

st.title("PDF Q&A")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the PDF")
if st.button("Answer"):
    if not pdf_file:
        st.error("Please upload a PDF first.")
    elif not question.strip():
        st.error("Please ask a question.")
    else:
        text = extract_text_from_pdf(pdf_file.read())
        chunks = split_text(text)

        emb = embedder.encode(chunks, convert_to_numpy=True)
        index = build_faiss_index(emb)

        retrieved = retrieve(question, embedder, index, chunks)

        context = "\n".join(retrieved)
        prompt = (
            f"Use the following context to answer the question in detail.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer clearly, with multiple sentences if needed:"
        )

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(device)
        
        # Answer generation
        
        outputs = gen_model.generate(
            **inputs,
            max_new_tokens=400,   
            num_beams=5,          
            temperature=0.7,      
            top_p=0.9,            
            no_repeat_ngram_size=3
        )
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.text_area("Answer", value=answer, height=300)

st.markdown("<center>---<center>", unsafe_allow_html=True)
st.markdown("<center>Dishant Vaidya</center>", unsafe_allow_html=True)
st.markdown("<center><small>vaidya.dishant@gmail.com</small></center>", unsafe_allow_html=True)


# In[ ]:




