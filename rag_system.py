from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.chains import create_retrieval_chain
import os

def get_vector_store():
    # 1. Ingest documents
    urls = [
        "https://www.who.int/news-room/fact-sheets/detail/the-top-10-causes-of-death",
        "https://www.who.int/health-topics/chronic-diseases",
        "https://www.cdc.gov/chronicdisease/about/index.htm"
    ]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    
    # 2. Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    chunks = text_splitter.split_documents(docs)
    
    # 3. Create embeddings
    # We'll use a local, open-source model for embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 4. Create vector store using ChromaDB
    vector_store = Chroma.from_documents(chunks, embeddings)
    return vector_store

def get_llm():
    # Use HuggingFaceHub for the LLM
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature": 0.5, "max_length": 64}
    )
    return llm

def get_retrieval_chain():
    # 1. Get the vector store and LLM
    vector_store = get_vector_store()
    llm = get_llm()
    retriever = vector_store.as_retriever()

    # 2. Define the prompt template
    prompt_template = """
    You are a helpful assistant.
    Answer the user's question based on the following context.
    If you don't know the answer, just say that you don't know.
    
    Context:
    {context}
    
    Question:
    {input}
    """
    
    prompt = PromptTemplate(
        input_variables=["context", "input"],
        template=prompt_template
    )

    # 3. Create the document chain and the retrieval chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain
