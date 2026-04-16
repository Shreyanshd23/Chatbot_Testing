import streamlit as st
import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Core and Community imports only (Avoids the problematic 'langchain.chains' module)
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
def get_models():
    # Exactly same as testing phase
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
    st.title("📁 Setup")
    uploaded_files = st.file_uploader("Upload Knowledge Base (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
    st.divider()
    st.info("Configuration: Size=1000, Overlap=200, TopK=3")

if uploaded_files:
    emb_model, llm_model = get_models()
    
    with st.spinner("Indexing your documents..."):
        chunks = process_docs(uploaded_files)
        vectorstore = FAISS.from_documents(chunks, emb_model)
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Build RAG Chain using modern LCEL (Pipe syntax)
        # This is more robust as it doesn't use the 'langchain.chains' module
        template = """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 

        Context:
        {context}

        Question: {question}
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        # This is the "Logic Pipe"
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

    if user_input := st.chat_input("Ask something..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Run the chain
                answer = rag_chain.invoke(user_input)
                st.markdown(answer)
                
                # Retrieve source docs manually for display
                docs = retriever.invoke(user_input)
                with st.expander("Sources"):
                    for doc in docs:
                        st.caption(f"Source: {doc.page_content[:200]}...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.info("Please upload documents in the sidebar to begin.")
