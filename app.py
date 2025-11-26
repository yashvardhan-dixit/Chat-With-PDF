import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
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
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # 3. Create Embeddings & Vector Store
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        # 4. User Question
        query = st.text_input("Ask a question about your PDF:")
        
        if query:
            # 5. Initialize LLM
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            
            # 6. Create the Chain (Modern LangChain LCEL)
            # Define prompt
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context:

            <context>
            {context}
            </context>

            Question: {input}
            """)

            # Create a "Stuff" chain (stuffs valid context into prompt)
            document_chain = create_stuff_documents_chain(llm, prompt)

            # Create the Retrieval Chain (connects FAISS to the Stuff chain)
            retriever = knowledge_base.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)
            
            # 7. Generate Response
            with get_openai_callback() as cb:
                response = retrieval_chain.invoke({"input": query})
                print(cb)
                
            st.write(response["answer"])

if __name__ == '__main__':
    main()
