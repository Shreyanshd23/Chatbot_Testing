import streamlit as st
import os
from dotenv import load_dotenv

# Core and Community imports
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="CarbonTatva AI - Chatbot", page_icon="🌱", layout="wide")

# --- Model Initialization ---
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource
def get_llm(api_key):
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key, temperature=0)

@st.cache_resource
def get_vectorstore():
    index_name = os.getenv("PINECONE_INDEX_NAME")
    api_key = os.getenv("PINECONE_API_KEY")
    
    if not index_name or not api_key:
        return None
    
    embeddings = get_embeddings()
    return PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=api_key)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- UI Sidebar ---
with st.sidebar:
    st.title("🌱 CarbonTatva AI")
    st.markdown("### Backend Status")
    
    google_api_key = os.getenv("GOOGLE_API_KEY")
    pinecone_ready = False
    
    # Check Google API Key
    if google_api_key:
        st.success("✅ Gemini API Key Loaded")
    else:
        st.error("❌ Gemini API Key Missing")
        st.info("Please set GOOGLE_API_KEY in your .env or Secrets.")

    # Check Pinecone
    vectorstore = get_vectorstore()
    if vectorstore:
        st.success("✅ Knowledge Base Connected")
        pinecone_ready = True
    else:
        st.error("❌ Knowledge Base Disconnected")
        st.info("Ensure Pinecone keys are set in .env.")

    st.divider()
    st.info("Model: Gemini 1.5 Flash\nRetrieval: Pinecone (TopK=3)")
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# --- Main Logic ---
st.title("🤖 CarbonTatva Assistant")
st.markdown("I am your AI assistant specialized in the provided knowledge base. How can I help you today?")

if not google_api_key or not pinecone_ready:
    st.warning("⚠️ Chat is disabled until the API keys are correctly configured in the backend.")
else:
    llm_model = get_llm(google_api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    template = """You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 

    Context:
    {context}

    Question: {question}
    Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm_model
        | StrOutputParser()
    )

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat Input
    if user_input := st.chat_input("Ask about CarbonTatva..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                answer = rag_chain.invoke(user_input)
                st.markdown(answer)
                
                # Sources Expander
                docs = retriever.invoke(user_input)
                with st.expander("Show Sources"):
                    for doc in docs:
                        source = doc.metadata.get('source', 'Unknown')
                        st.caption(f"**From {os.path.basename(source)}:**")
                        st.write(f"{doc.page_content[:300]}...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
