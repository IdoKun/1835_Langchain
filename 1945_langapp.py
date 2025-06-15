import streamlit as st
import os
import tempfile
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3


def file_checker(file_path):
    try:
        loader = TextLoader(file_path)
        documents = loader.load()
    except FileNotFoundError:
        return "File not found."

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    openai_api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
    if not openai_api_key:
        return "OpenAI API key is missing. Please configure it in Streamlit secrets."

    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
    docsearch = Chroma.from_documents(texts, embedder)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4.1-nano-2025-04-14", openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 1}),
    )

    query = "Rate the following Python code out of 10. Give three improvements that can be made to it."
    answer = qa.invoke(query)

    return answer["result"]

# Streamlit UI
st.title("Python Code Reviewer with LangChain 🤖")

st.write(
    "Upload a Python file (.py), and the AI will rate it out of 10 and suggest improvements."
)

uploaded_file = st.file_uploader("Upload a Python file", type=["py"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".py") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    st.write("Analyzing your code... ⏳")
    result = file_checker(temp_path)

    st.subheader("AI Feedback 📝")
    st.write(result)

    os.remove(temp_path)  # Clean up temp file
