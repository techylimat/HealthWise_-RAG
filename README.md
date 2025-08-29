Healthwise RAG
This project is a simple Retrieval-Augmented Generation (RAG) system built to answer questions about chronic diseases and public health, using authoritative sources like the CDC and WHO. It demonstrates the core principles of document ingestion, vector indexing, retrieval, and grounded generation.

The final application is a user-friendly web interface built with Streamlit.

Chosen Tech Stack
Framework: LangChain in Python

User Interface: Streamlit

Document Loader: WebBaseLoader

Text Splitter: RecursiveCharacterTextSplitter

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS

LLM: google/flan-t5-xxl from Hugging Face

Project Files
rag_system.py: Contains all the core logic, including document processing, vector store creation, and the RAG chain.

app.py: The Streamlit application that handles the user interface and interacts with rag_system.py.

requirements.txt: A list of all necessary Python libraries.

Setup Instructions
Clone the Repository:

git clone <your-repo-url>
cd <your-repo-name>




Set Up a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`




Install Dependencies:

pip install -r requirements.txt




Set Hugging Face Environment Variables:

Sign up for a free account on Hugging Face.

Go to "Settings" -> "Access Tokens" and create a new token.

Copy the token.

Set the USER_AGENT environment variable (optional but recommended):

export HF_HUB_USER_AGENT="my-rag-system/1.0"



Set the API token as an environment variable:

export HUGGINGFACEHUB_API_TOKEN="hf_..."



On Windows, use set HF_HUB_USER_AGENT="my-rag-system/1.0" and set HUGGINGFACEHUB_API_TOKEN="hf_..."

Run the Streamlit Application:

streamlit run app.py




Your web browser should automatically open the application.

Example Queries
"What are the major chronic diseases?"

"How can chronic diseases be prevented?"

"What is a non-communicable disease?"

"Give me some key facts about chronic diseases."

Known Issues & Limitations
API Token: The application requires a Hugging Face API token to function.

Document Scope: The system's knowledge is limited to the documents from the CDC and WHO URLs defined in rag_system.py.

Generative Model: The google/flan-t5-xxl model is a general-purpose LLM. For more accurate and specific medical advice, a specialized LLM would be required. This is for demonstration purposes only and should not be used for medical advice.
