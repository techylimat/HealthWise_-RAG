import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# This is where the core RAG logic is handled.

# Function to get the vector store from documents
@st.cache_resource(show_spinner=False)
def get_vector_store():
    """
    Loads documents, splits them into chunks, creates embeddings,
    and stores them in a FAISS vector store.
    """
    with st.spinner("Loading and processing documents..."):
        # 1. Document Ingestion
        # Use a trusted public health source. You can replace these with
        # other URLs from sites like the WHO, CDC, or others.
        urls = [
            "https://www.cdc.gov/chronicdisease/about/index.htm",
            "https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases-and-their-risk-factors",
            "https://www.cdc.gov/chronicdisease/resources/infographic/chronic-diseases.htm",
            "https://www.who.int/news-room/questions-and-answers/item/chronic-diseases--faqs"
        ]
        loader = WebBaseLoader(urls)
        docs = loader.load()

        # 2. Chunking Strategy
        # Splits the text into chunks of 500 characters with a 100 character overlap.
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)

        # 3. Embedding Generation
        # Using a self-contained sentence-transformers model.
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # 4. Vector Storage in FAISS
        # Creates a FAISS vector store from the document chunks and embeddings.
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        return vector_store

# Function to create the retrieval chain
def get_retrieval_chain(vector_store, llm):
    """
    Creates a retrieval chain with a prompt template.
    """
    # 5. Retrieval Method
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Optional Stretch Goal: Reranker integration
    # You would typically integrate a reranker here.
    # For example:
    # from langchain.retrievers import ContextualCompressionRetriever
    # from langchain_community.document_compressors import FlashrankRerank
    # compressor = FlashrankRerank()
    # compression_retriever = ContextualCompressionRetriever(
    #     base_compressor=compressor, base_retriever=retriever
    # )
    # retriever = compression_retriever

    # 6. Prompt Template for Generation
    prompt_template = """
    You are an expert public health assistant. Use the following context to answer the user's question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Your answer should be accurate, concise, and based ONLY on the provided context.

    Context: {context}

    Question: {input}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])

    # 7. Runnable Chain
    # Combines the retriever and the LLM into a single chain.
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# Function to get the LLM (Large Language Model)
def get_llm():
    """
    Initializes and returns the Large Language Model.
    Requires HUGGINGFACEHUB_API_TOKEN environment variable.
    """
    # Using an open-source model from Hugging Face.
    # Note: Requires a Hugging Face API token set as an environment variable.
    # Set it like this: os.environ["HUGGINGFACEHUB_API_TOKEN"] = "hf_..."
    repo_id = "google/flan-t5-xxl"
    llm = HuggingFaceHub(
        repo_id=repo_id,
        model_kwargs={"temperature": 0.5, "max_length": 64},
    )
    return llm
