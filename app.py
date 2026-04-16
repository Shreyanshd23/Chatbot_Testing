import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Modern LangChain imports (v0.2+)
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Linear RAG Chatbot", page_icon="🤖", layout="wide")

# --- Model Initialization ---
@st.cache_resource
def get_models():
    # Winning configuration from benchmark
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    return embeddings, llm

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
                # Switched to Docx2txt (much lighter/stable than unstructured)
                loader = Docx2txtLoader(str(file_path))
                all_docs.extend(loader.load())
                
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(all_docs)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI ---
st.title("🤖 Linear RAG Chatbot (v2)")

with st.sidebar:
    st.title("📁 Setup")
    uploaded_files = st.file_uploader("Upload Knowledge Base (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    
    st.divider()
    st.info("Uses optimized settings: Size=1000, Overlap=200, TopK=3")

if uploaded_files:
    emb_model, llm_model = get_models()
    
    with st.spinner("Indexing your documents..."):
        chunks = process_docs(uploaded_files)
        vectorstore = FAISS.from_documents(chunks, emb_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Modern RAG Chain Construction
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm_model, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    st.success(f"Ready! Indexed {len(chunks)} sections.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("Ask something about your docs..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("AI is thinking..."):
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                with st.expander("Sources"):
                    for doc in response["context"]:
                        st.caption(f"Source: {doc.page_content[:200]}...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload documents to start.")
