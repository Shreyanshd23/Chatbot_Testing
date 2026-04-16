import os
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path

# Load environment variables
load_dotenv()

# --- Configuration ---
# Scanning the Database folder specifically for .docx files
KNOWLEDGE_BASE_DIR = r"../Database"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "carbon-tatva-kb")
API_KEY = os.getenv("PINECONE_API_KEY")

if not API_KEY:
    print("❌ Error: PINECONE_API_KEY not found in environment.")
    exit(1)

# --- Initialize Pinecone ---
pc = Pinecone(api_key=API_KEY)

# Pinecone Index Setup
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating it...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print(f"✅ Index '{INDEX_NAME}' created.")
else:
    print(f"✅ Using existing index: {INDEX_NAME}")

def get_docx_docs(folder_path):
    all_docs = []
    path = Path(folder_path)
    
    # Strictly searching for .docx to keep it fast and free
    print(f"📂 Scanning for .docx files in: {folder_path}...")
    files = list(path.rglob("*.docx"))
    
    if not files:
        print(f"⚠️ No .docx documents found.")
        return []

    print(f"📂 Found {len(files)} .docx files.")
    
    for file_path in files:
        print(f"  📄 Processing: {file_path.name}")
        try:
            loader = Docx2txtLoader(str(file_path))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"    ❌ Error loading {file_path}: {e}")
            
    return all_docs

def main():
    # 1. Load DOCX
    docs = get_docx_docs(KNOWLEDGE_BASE_DIR)
    if not docs:
        print("❌ No documents were loaded.")
        return

    print(f"📄 Total sections loaded: {len(docs)}")
    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"🔤 Total characters extracted: {total_chars}")

    # 2. Split
    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    # 3. Embeddings (Local & Free)
    print("🧠 Initializing embeddings...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Upload to Cloud Backend
    print(f"🚀 Uploading {len(chunks)} chunks to Pinecone...")
    vectorstore = PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=INDEX_NAME
    )
    print("✨ Knowledge Base Ready!")
    print("🤖 Run 'streamlit run app.py' to start chatting.")

if __name__ == "__main__":
    main()
