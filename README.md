# 🌱 CarbonTatva AI - Chatbot

This is a production-ready Streamlit RAG application powered by **Gemini 1.5 Flash** and **Pinecone**.

## 🚀 Key Features
- **Managed Backend**: Knowledge base is stored in Pinecone (no large files in GitHub).
- **Auto-Config**: Loads API keys automatically from `.env` or Streamlit Secrets.
- **Zero-Touch UI**: No need to manually upload documents or enter keys in the app.

## 🛠️ Setup Instructions

### 1. Preparation
1. Create a free account at [Pinecone.io](https://app.pinecone.io/).
2. Create an index named `carbon-tatva-kb` (or your choice) with **384 dimensions** and **cosine** metric.
3. Obtain your **Pinecone API Key**.

### 2. Local Configuration
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure your `.env` file (based on `.env.example`):
   ```env
   GOOGLE_API_KEY=...
   PINECONE_API_KEY=...
   PINECONE_INDEX_NAME=carbon-tatva-kb
   ```

### 3. One-Time Indexing
Upload your documents to the cloud backend:
```bash
python upload_to_pinecone.py
```
*Note: This script scans the `../Database/Knowledgebase_data` directory by default.*

### 4. Run the App
```bash
streamlit run app.py
```

## 🌐 Deployment to Streamlit Cloud

1. Push your code to GitHub (exclude your `.env`).
2. In Streamlit Cloud, add your `.env` variables to **Secrets**:
   ```toml
   GOOGLE_API_KEY = "..."
   PINECONE_API_KEY = "..."
   PINECONE_INDEX_NAME = "carbon-tatva-kb"
   ```
3. The app will automatically connect to your managed Pinecone backend!
