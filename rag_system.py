import os
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set the USER_AGENT environment variable to identify requests to Hugging Face Hub
os.environ["USER_AGENT"] = "healthwise-rag-app"

def get_text_chunks_from_web(urls):
    """
    Loads text from a list of URLs and splits it into chunks.
    """
    try:
        print("DEBUG: Starting to load documents from the web...")
        loader = WebBaseLoader(urls)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)
        print("DEBUG: Documents loaded and split into chunks successfully.")
        return chunks
    except requests.exceptions.ConnectionError as e:
        print(f"DEBUG: Connection Error: Please check your internet connection or URL validity. Details: {e}")
        return []
    except Exception as e:
        print(f"DEBUG: Error loading and splitting documents: {e}")
        return []

def get_vector_store(text_chunks):
    """
    Creates a Chroma vector store from text chunks.
    """
    try:
        print("DEBUG: Creating vector store...")
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = Chroma.from_documents(text_chunks, embedding=embeddings)
        print("DEBUG: Vector store created successfully.")
        return vector_store
    except Exception as e:
        print(f"DEBUG: Error creating vector store: {e}")
        return None

def get_retrieval_chain():
    """
    Creates and returns a RetrievalQA chain.
    """
    try:
        print("DEBUG: Setting up the retrieval chain...")
        urls = [
            "https://www.who.int/health-topics/diabetes",
            "https://www.who.int/news-room/fact-sheets/detail/diabetes",
            "https://en.wikipedia.org/wiki/Diabetes"
        ]
        text_chunks = get_text_chunks_from_web(urls)
        if not text_chunks:
            print("DEBUG: Failed to get text chunks. Aborting setup.")
            return None

        vector_store = get_vector_store(text_chunks)
        if not vector_store:
            print("DEBUG: Failed to create vector store. Aborting setup.")
            return None
            
        print("DEBUG: Initializing LLM...")
        try:
            # Using a locally run model to avoid API issues
            model_name = "google/flan-t5-base"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
            )
            
            llm = HuggingFacePipeline(pipeline=pipe)
            print("DEBUG: LLM initialized successfully.")
        except Exception as e:
            print(f"DEBUG: Failed to initialize LLM: {e}")
            return None

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
        print("DEBUG: Retrieval chain setup complete. The app is ready to answer questions.")
        return retrieval_chain
    except Exception as e:
        print(f"DEBUG: An error occurred during retrieval chain setup: {e}")
        return None
