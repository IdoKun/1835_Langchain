# import os
# import streamlit as st
# from langchain.chains import RetrievalQA
# from langchain_community.document_loaders import TextLoader
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import Chroma

import os
import sys

# Force LangChain to use Pydantic v1
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "fake-key"  # Just to prevent tracing issues

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma


# ✅ Debug: Ensure API Key is set up
st.write("🔍 Checking environment variables...")

openai_api_key = st.secrets.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("🚨 OpenAI API key not found! Set it in Streamlit Cloud's secrets.")
    st.stop()

st.write(f"✅ OpenAI API Key found: {openai_api_key[:5]}********")

def file_checker(file_path):
    """Loads a Python file, processes it with LangChain, and evaluates the code."""

    # ✅ Debug: Check if the file exists
    if not os.path.isfile(file_path):
        st.error(f"🚨 File not found: {file_path}")
        return None
    st.write(f"📂 File found: {file_path}")

    try:
        # Load the file using LangChain TextLoader
        loader = TextLoader(file_path)
        documents = loader.load()
    except Exception as e:
        st.error(f"🚨 Error loading file: {str(e)}")
        return None

    # ✅ Debug: Check the number of documents loaded
    st.write(f"📄 Documents loaded: {len(documents)}")

    # Split the document into smaller chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # ✅ Debug: Check the number of text chunks
    st.write(f"📜 Text chunks created: {len(texts)}")

    # Initialize OpenAI Embeddings
    #embedder = OpenAIEmbeddings(openai_api_key=openai_api_key)
    #embedder = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=openai_api_key)
    embedder = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)



    # Store vectors in Chroma DB
    docsearch = Chroma.from_documents(texts, embedder)

    # Create RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-4-turbo-preview", openai_api_key=openai_api_key),
        chain_type="stuff",
        retriever=docsearch.as_retriever(search_kwargs={"k": 1})
    )

    # Query for evaluation
    query = "Rate the following Python code out of 10. Give three improvements that can be made to it."

    # Invoke the model
    try:
        answer = qa.invoke(query)
        st.write(f"📝 Model output: {answer['result']}")
        return answer["result"]
    except Exception as e:
        st.error(f"🚨 Error during model invocation: {str(e)}")
        return None

# Streamlit UI
st.title('Python Code Evaluator')
st.write("📂 Upload a `.py` file to analyze its quality and get improvement suggestions.")

uploaded_file = st.file_uploader("Choose a Python file", type="py")

if uploaded_file is not None:
    file_path = os.path.join("/tmp", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write(f"🔍 Running `{uploaded_file.name}`...")

    # Run the file checker
    result = file_checker(file_path)

    if result:
        st.subheader('Evaluation Result:')
        st.text(result)
    else:
        st.error("❌ No answer generated.")
