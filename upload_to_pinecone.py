import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from pathlib import Path

# Load environment variables from streamlit_deployment/.env or root .env
load_dotenv()

# --- Configuration ---
# You can change these paths as needed
KNOWLEDGE_BASE_DIR = r"../Database/Knowledgebase_data"
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "carbon-tatva-kb")
API_KEY = os.getenv("PINECONE_API_KEY")

if not API_KEY:
    print("❌ Error: PINECONE_API_KEY not found in environment.")
    exit(1)

# --- Initialize Pinecone ---
pc = Pinecone(api_key=API_KEY)

# Check if index exists, create if not
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Index '{INDEX_NAME}' not found. Creating it...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=384, # Matches sentence-transformers/all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1") # Common free tier region
    )
    print(f"✅ Index '{INDEX_NAME}' created.")
else:
    print(f"✅ Using existing index: {INDEX_NAME}")

# --- Processing Logic ---
def get_all_docs(folder_path):
    all_docs = []
    path = Path(folder_path)
    
    # Recursive search for pdf and docx
    files = list(path.rglob("*.pdf")) + list(path.rglob("*.docx"))
    
    if not files:
        print(f"⚠️ No documents found in {folder_path}")
        return []

    print(f"📂 Found {len(files)} files. Starting processing...")
    
    for file_path in files:
        print(f"  📄 Processing: {file_path.name}")
        try:
            if file_path.suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                all_docs.extend(loader.load())
            elif file_path.suffix == ".docx":
                loader = Docx2txtLoader(str(file_path))
                all_docs.extend(loader.load())
        except Exception as e:
            print(f"    ❌ Error loading {file_path}: {e}")
            
    return all_docs

def main():
    # 1. Load Docs
    docs = get_all_docs(KNOWLEDGE_BASE_DIR)
    if not docs:
        return

    # 2. Split
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"✅ Created {len(chunks)} chunks.")

    # 3. Embeddings
    print("🧠 Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # 4. Upload
    print("🚀 Uploading to Pinecone (this may take a few minutes)...")
    vectorstore = PineconeVectorStore.from_documents(
        chunks, 
        embeddings, 
        index_name=INDEX_NAME
    )
    print(f"✨ Successfully uploaded {len(chunks)} chunks to Pinecone!")
    print("🎉 Your backend knowledge base is ready.")

if __name__ == "__main__":
    main()
