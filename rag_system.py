import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Set the USER_AGENT environment variable to identify requests to Hugging Face Hub
os.environ["USER_AGENT"] = "healthwise-rag-app"

def get_text_chunks_from_web(urls):
    """
    Loads text from a list of URLs and splits it into chunks.
    """
    try:
        loader = WebBaseLoader(urls)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return text_splitter.split_documents(documents)
    except Exception as e:
        print(f"Error loading and splitting documents: {e}")
        return []

def get_vector_store(text_chunks):
    """
    Creates a Chroma vector store from text chunks.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

def get_retrieval_chain(huggingfacehub_api_token):
    """
    Creates and returns a RetrievalQA chain.
    """
    try:
        urls = [
            "https://www.who.int/health-topics/diabetes",
            "https://www.who.int/news-room/fact-sheets/detail/diabetes",
            "https://en.wikipedia.org/wiki/Diabetes"
        ]
        text_chunks = get_text_chunks_from_web(urls)
        if not text_chunks:
            return None

        vector_store = get_vector_store(text_chunks)
        if not vector_store:
            return None

        # Use the updated HuggingFaceEndpoint class
        llm = HuggingFaceEndpoint(
            repo_id="google/flan-t5-xxl",
            huggingfacehub_api_token=huggingfacehub_api_token,
        )

        template = """You are a helpful and knowledgeable assistant specializing in public health. Your task is to provide concise and accurate answers to questions based on the provided context.
        If the answer is not contained in the context, say "Sorry, I couldn't find a relevant answer in the provided documents."
        Context: {context}
        Question: {question}
        Answer:"""
        prompt = PromptTemplate.from_template(template)

        retrieval_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(),
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt},
            input_key="query"
        )
        return retrieval_chain
    except Exception as e:
        print(f"An error occurred during retrieval chain setup: {e}")
        return None
