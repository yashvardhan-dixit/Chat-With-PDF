import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
import os

# Sidebar for API Key (Secure way to handle keys in a demo)
with st.sidebar:
    st.title("🔐 Configuration")
    api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not api_key:
        st.warning("Please enter your API Key to proceed.")
        st.stop()
    os.environ["OPENAI_API_KEY"] = api_key

def main():
    st.header("📄 Chat with your PDF (RAG Pipeline)")

    # 1. Upload PDF
    pdf = st.file_uploader("Upload your PDF here", type="pdf")
    
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
            
        # 2. Split Text (Chunking)
        # We split text into chunks of 1000 characters with 200 overlap to maintain context
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 3. Create Embeddings & Vector Store
        # This converts text chunks into vectors (numbers)
        embeddings = OpenAIEmbeddings()
        
        # FAISS (Facebook AI Similarity Search) is our local vector database
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # 4. User Question
        query = st.text_input("Ask a question about your PDF:")
        
        if query:
            # 5. Search for relevant chunks (Similarity Search)
            docs = knowledge_base.similarity_search(query)
            
            # 6. Initialize LLM
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            # 7. Generate Answer using the "stuff" chain (stuffs context into prompt)
            chain = load_qa_chain(llm, chain_type="stuff")
            
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=query)
                print(cb) # Prints token usage to terminal (good for debugging costs)
                
            st.write(response)

if __name__ == '__main__':
    main()
