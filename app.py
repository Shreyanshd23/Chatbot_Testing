import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(
    page_title="Linear RAG Pipeline",
    page_icon="🤖",
    layout="wide"
)

# --- Styling ---
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stSecondaryBlock {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    </style>
""", unsafe_allow_html=True)

# --- Model Initialization (Winner Model from Testing) ---
# Results showed Linear_Size1000_Overlap200_Top3 as the winner (100% Hit Rate, 1.0 MRR)

@st.cache_resource
def get_winner_configs():
    return {
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "top_k": 3,
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "llm_model": "gemini-1.5-flash"
    }

@st.cache_resource
def get_models():
    configs = get_winner_configs()
    embeddings = HuggingFaceEmbeddings(model_name=configs["embedding_model"])
    llm = ChatGoogleGenerativeAI(model=configs["llm_model"], temperature=0)
    return embeddings, llm

# --- Sidebar ---
with st.sidebar:
    st.title("📁 Knowledge Base")
    st.markdown("Upload your documents to chat with them.")
    uploaded_files = st.file_uploader("Choose PDF or DOCX files", type=["pdf", "docx"], accept_multiple_files=True)
    
    st.divider()
    st.markdown("### Winner Model Configuration")
    configs = get_winner_configs()
    st.info(f"""
    - **Chunk Size**: {configs['chunk_size']}
    - **Overlap**: {configs['chunk_overlap']}
    - **Top K**: {configs['top_k']}
    - **Embeddings**: {configs['embedding_model']}
    - **LLM**: {configs['llm_model']}
    """)

# --- Main App Logic ---

def process_documents(files, c_size, c_overlap):
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
                loader = UnstructuredWordDocumentLoader(str(file_path))
                all_docs.extend(loader.load())
                
    splitter = RecursiveCharacterTextSplitter(chunk_size=c_size, chunk_overlap=c_overlap)
    chunks = splitter.split_documents(all_docs)
    return chunks

@st.cache_resource
def build_vectorstore(_chunks, _embeddings):
    return FAISS.from_documents(_chunks, _embeddings)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Header ---
st.title("🤖 Linear RAG Pipeline (Optimized)")

if uploaded_files:
    emb_model, llm_model = get_models()
    configs = get_winner_configs()
    
    with st.spinner("Processing documents with optimized settings..."):
        chunks = process_documents(uploaded_files, configs['chunk_size'], configs['chunk_overlap'])
        vectorstore = build_vectorstore(chunks, emb_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": configs['top_k']})
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm_model,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
    
    st.success(f"Indexed {len(chunks)} chunks using winning configuration.")
    
    # Display Chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Generating answer..."):
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                sources = response["source_documents"]
                
                st.markdown(answer)
                
                with st.expander("Show Sources"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(doc.page_content)
                        st.caption(f"Metadata: {doc.metadata}")

        st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👋 Welcome! Please upload your documents in the sidebar to start chatting.")
    st.markdown(f"""
    ### Optimized Stack (Winner Model):
    - **Chunk Size**: {configs['chunk_size']}
    - **Chunk Overlap**: {configs['chunk_overlap']}
    - **Top K**: {configs['top_k']}
    - **Embeddings**: `{configs['embedding_model']}`
    - **LLM**: `{configs['llm_model']}`
    """)
