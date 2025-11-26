📄 Chat with PDF (RAG Pipeline)

A Retrieval-Augmented Generation (RAG) chatbot that allows users to chat with their own PDF documents. Built using LangChain, OpenAI, and Streamlit.

🚀 Live Demo

[Link to your deployed streamlit app here]

🛠️ Tech Stack

Framework: Streamlit

LLM Orchestration: LangChain

Vector Database: FAISS (Facebook AI Similarity Search)

Model: OpenAI GPT-3.5 & Text-Embedding-Ada-002

⚙️ How it Works

Ingestion: The app reads the uploaded PDF and extracts raw text.

Chunking: Text is split into smaller chunks (1000 chars) to fit within the LLM context window.

Embedding: Chunks are converted into vector embeddings using OpenAI.

Retrieval: When a user asks a question, the system finds the most semantically similar chunks in the FAISS index.

Generation: The relevant chunks + user question are sent to GPT-3.5 to generate an accurate answer based only on the document.

📦 Local Installation

Clone the repo

Install requirements: pip install -r requirements.txt

Run the app: streamlit run app.py
