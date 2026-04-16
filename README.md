# Linear RAG Pipeline Deployment

This folder contains the code for a Streamlit-based Linear RAG (Retrieval-Augmented Generation) pipeline, using the exact models from the testing phase.

## 🚀 Exact Stack
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (Local HuggingFace model).
- **LLM**: `gemini-1.5-flash` (Google Generative AI).
- **Vector Store**: FAISS.

## 🛠️ Local Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repo-url>
   cd streamlit_deployment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**:
   Create a `.env` file in this directory and add your Google API key:
   ```env
   GOOGLE_API_KEY=your_actual_key_here
   ```

4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

## 🌐 Deployment to Streamlit Cloud

1. Upload this folder to a GitHub repository.
2. Go to [Streamlit Cloud](https://share.streamlit.io/).
3. Connect your GitHub repo.
4. Set the `GOOGLE_API_KEY` in the **Secrets** section of the Streamlit dashboard settings.

## 📄 Requirements
- Python 3.9+
- A Google Gemini API Key
