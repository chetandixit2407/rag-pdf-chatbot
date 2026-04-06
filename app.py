import streamlit as st
from transformers import logging
logging.set_verbosity_error()

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ✅ Cache embeddings
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# ✅ Cache vector DB
@st.cache_resource
def create_db(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # 🔥 Chunking (important)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(docs)

    embeddings = load_embeddings()
    return FAISS.from_documents(docs, embeddings)

st.title("💬 Chat with PDF")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    db = create_db("temp.pdf")

    llm = Ollama(model="llama3:8b")

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever()
    )

    query = st.text_input("Ask something")

    if st.button("Ask"):
        if query:
            with st.spinner("Thinking..."):
                response = qa.run(query)   # ✅ FIXED

            st.write(response)
        else:
            st.warning("Please enter a question")
