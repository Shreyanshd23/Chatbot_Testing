import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Core and Community imports only
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Linear RAG Chatbot", page_icon="🤖", layout="wide")

# --- Model Initialization ---
@st.cache_resource
def get_embeddings():
    # Exactly same local embeddings as testing phase
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)

# --- App Logic ---
def process_docs(files):
    all_docs = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in files:
            file_path = Path(temp_dir) / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                all_docs.extend(loader.load())
            elif file_path.suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
                all_docs.extend(loader.load())
                
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI ---
st.title("🤖 Linear RAG Chatbot")

with st.sidebar:
    st.title("📁 Configuration")
    
    # Option B: UI-based API Key Input
    api_key_input = st.text_input("Enter Google Gemini API Key", type="password")
    
    # Priority: 1. UI Input, 2. Environment Secret
    google_api_key = api_key_input if api_key_input else os.getenv("GOOGLE_API_KEY")
    
    if not google_api_key:
        st.warning("⚠️ Please provide a Google API Key to enable the LLM.")
    else:
        st.success("✅ API Key loaded.")

    st.divider()
    uploaded_files = st.file_uploader("Upload Knowledge Base (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    
    st.divider()
    st.info("Settings: Size=1000, Overlap=200, TopK=3")

if uploaded_files:
    if not google_api_key:
        st.error("Please enter your Google API Key in the sidebar to chat.")
    else:
        emb_model = get_embeddings()
        llm_model = get_llm(google_api_key)
        
        with st.spinner("Indexing your documents..."):
            chunks = process_docs(uploaded_files)
            vectorstore = FAISS.from_documents(chunks, emb_model)
            retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
            
            template = """You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know. 

            Context:
            {context}

            Question: {question}
            Answer:"""
            
            prompt = ChatPromptTemplate.from_template(template)
            
            rag_chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm_model
                | StrOutputParser()
            )

        st.success(f"Ready! Indexed {len(chunks)} sections.")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask about your documents..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = rag_chain.invoke(user_input)
                    st.markdown(answer)
                    
                    docs = retriever.invoke(user_input)
                    with st.expander("Sources"):
                        for doc in docs:
                            st.caption(f"Source: {doc.page_content[:200]}...")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload documents in the sidebar to begin.")
