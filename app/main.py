import streamlit as st
from pdf_parser import extract_text_from_pdf, chunk_text
from rag_pipeline import build_faiss_index, search_index
from qa_engine import generate_answer
import os
import tempfile

st.set_page_config(page_title="Legal Q&A System", layout="wide")
st.title("Legal Document Q&A")

uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.success("PDF uploaded successfully.")

    if st.button("Start"):
        text = extract_text_from_pdf(tmp_path)
        chunks = chunk_text(text)
        build_faiss_index(chunks)
        st.session_state["index"] = True
        st.session_state["doc_name"] = uploaded_file.name
        st.success(f"Indexed {len(chunks)}.")

        st.info("Go to question page.")