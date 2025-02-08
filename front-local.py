import os
import streamlit as st

from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

def file_checker(file, openai_api_key):
    try:
        loader = TextLoader(file)
        documents = loader.load()
    except FileNotFoundError:
        st.error(f"File not found: {file}")
        return None

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    if not openai_api_key:
        st.error("OpenAI API key not found. Set the OPENAI_API_KEY environment variable.")
        return None
    embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)

    docsearch = Chroma.from_documents(texts, embedder)

    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-turbo-preview", openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 1})
    )

    query = "Rate the following Python code out of 10. Give three improvements that can be made to it."
    answer = qa.invoke(query)

    return answer["result"]

st.title('Python Code Evaluator')

uploaded_file = st.file_uploader("Choose a Python file", type="py")

if uploaded_file is not None:
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"Running `{uploaded_file.name}`...")

    openai_api_key = st.secrets["OPENAI_API_KEY"]
    result = file_checker(file_path, openai_api_key)

    if result:
        st.subheader('Evaluation Result:')
        st.text(result)
    else:
        st.error("No answer generated.")
