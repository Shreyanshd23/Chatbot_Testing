import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Configuration (Matching the wining model)
DOC_PATH = r"c:\Users\dewan\Desktop\CarbonTatva\Database\Test_data\Fictional Company Data Zenith Textiles Ltd.docx"
SAVE_PATH = r"c:\Users\dewan\Desktop\CarbonTatva\streamlit_deployment\faiss_index"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

def create_backend_index():
    print(f"Loading document: {DOC_PATH}")
    loader = Docx2txtLoader(DOC_PATH)
    docs = loader.load()
    
    print("Splitting documents...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    
    print("Generating embeddings and building index...")
    # Matches the exact embedding model from testing
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    
    print(f"Saving index to: {SAVE_PATH}")
    vectorstore.save_local(SAVE_PATH)
    print("Success!")

if __name__ == "__main__":
    create_backend_index()
