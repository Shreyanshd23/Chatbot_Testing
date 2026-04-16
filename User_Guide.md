# 📖 User Guide: Linear RAG Chatbot

Welcome to the **Linear RAG Chatbot**! This application allows you to chat with your own documents (PDF and DOCX) using an optimized Retrieval-Augmented Generation (RAG) pipeline.

---

## 🛠️ Prerequisites

To use this app, you will need:
- **Google Gemini API Key**: This is used for generating answers. You can get one for free at the [Google AI Studio](https://aistudio.google.com/).
- **Documents**: Your knowledge base in `.pdf` or `.docx` format.

---

## 🚀 Getting Started

### 1. Upload Your Documents
In the **sidebar** (left-hand side):
1. Click on **"Choose PDF or DOCX files"**.
2. Select one or more files from your computer.
3. The app will automatically start processing and indexing your documents.

### 2. Monitoring Progress
You will see a spinner indicating that the documents are being "Indexed". 
- **Embeddings**: The app uses a local model (`all-MiniLM-L6-v2`) to turn your text into numbers.
- **Indexing**: It splits your documents into chunks of 1000 characters and stores them in a FAISS vector database.

Once finished, a green success message will appear showing how many chunks were created.

---

## 💬 How to Chat

### 1. Ask a Question
At the bottom of the screen, you'll see a chat input box. Type any question related to the documents you uploaded, for example:
> "What are the main financial highlights mentioned in the report?"

### 2. View the Answer
The AI will generate an answer based **only** on the context retrieved from your documents. This ensures high accuracy and reduces hallucinations.

### 3. Check the Sources
Below each AI response, you can click on **"Show Sources"**. 
- This will reveal the exact snippets (chunks) of text the AI used to formulate its answer.
- This is helpful for verifying facts and seeing where the information came from.

---

## ⚙️ Optimized Configuration (Under the Hood)
This app is pre-configured with the **winning parameters** from our benchmarking phase:
- **Chunk Size**: 1000 characters (Good balance of context and precision).
- **Chunk Overlap**: 200 characters (Ensures transition text isn't lost).
- **Top-K**: 3 (Retrieves the 3 most relevant sections for every query).
- **LLM**: Gemini 1.5 Flash (Fast and intelligent).

---

## 🆘 Troubleshooting
- **No Answer Generated**: Ensure your `GOOGLE_API_KEY` is correctly set in the environment or Streamlit Secrets.
- **Slow Processing**: If you upload very large documents (100+ pages), the indexing might take a minute as it runs on the server's CPU.
- **File Not Supported**: Ensure your file ends in `.pdf` or `.docx`.

### One small but CRITICAL thing you must do manually:
Streamlit Cloud cannot read your .env file (for security reasons). You must provide your API key through their dashboard:

On your app page, click "Manage app" (bottom right).
Click the three dots (⋮) and select "Settings".
Go to "Secrets".
Paste your key like this:
toml
GOOGLE_API_KEY = "your-actual-api-key-here"

---
*Happy Chatting with your Data!*
