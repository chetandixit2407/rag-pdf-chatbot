import streamlit as st

from langchain_community.llms import Ollama
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFaceHub

from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate

st.title("💬 Chat with PDF")

file = st.file_uploader("Upload PDF", type="pdf")

if file:
    with open("temp.pdf", "wb") as f:
        f.write(file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(docs, embeddings)

    llm = Ollama(model="llama3")

    prompt = ChatPromptTemplate.from_template(
        "Answer using context:\n{context}\n\nQuestion: {input}"
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever()
)

    query = st.text_input("Ask question")

    if query:
       res = qa.run(query)
       st.write(res)
