import streamlit as st
import time
from rag_system import get_retrieval_chain

# Custom CSS for a beautiful and coherent theme
st.markdown(
    """
    <style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        color: #007bff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .st-emotion-cache-18ni7ap.ezrtsby2 {
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 20px;
    }
    .st-emotion-cache-1c19b6.ezrtsby2 {
        background-color: #e9ecef;
        border-radius: 10px;
    }
    .st-emotion-cache-1p6f2r4.ezrtsby2 {
        border-radius: 10px;
    }
    .st-emotion-cache-1627l4p.e1tzp5d2 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .st-emotion-cache-1w0l7e4.e1f1d6gn4 {
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Main Application ---
st.set_page_config(
    page_title="Healthwise RAG",
    page_icon="⚕️",
    layout="wide",
)

st.title("⚕️ Healthwise RAG")

# Add a description or instructions
st.markdown(
    """
    **Ask a question about chronic diseases or public health.**
    
    _This system uses a Retrieval-Augmented Generation (RAG) model to find relevant information from trusted sources and provide a grounded, accurate answer._
    """
)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# --- Setup and Error Handling with Progress Bar ---
try:
    with st.spinner("Setting up the RAG system..."):
        # Faking the progress to provide user feedback
        progress_bar = st.progress(0)
        st.text("Step 1 of 3: Initializing model and services...")
        progress_bar.progress(33)
        time.sleep(0.5)

        st.text("Step 2 of 3: Loading documents and creating vector store...")
        progress_bar.progress(66)
        time.sleep(0.5)

        if st.session_state.retrieval_chain is None:
            st.session_state.retrieval_chain = get_retrieval_chain()

        st.text("Step 3 of 3: System is ready to go!")
        progress_bar.progress(100)
        time.sleep(0.5)

    st.success("System ready!")

except Exception as e:
    st.error(
        f"An error occurred during setup: {e}"
    )
    st.warning(
        "Please check your environment variables, especially your Hugging Face API token, and ensure you have an active internet connection."
    )
    # This will allow the error message to be displayed and then stop the application flow
    st.stop()

# --- Chat History Display ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Handler ---
if prompt := st.chat_input("Ask a question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Use a Streamlit-friendly streaming method
        response_generator = st.session_state.retrieval_chain.stream({"input": prompt})

        for chunk in response_generator:
            if isinstance(chunk, str):
                full_response += chunk
            elif isinstance(chunk, dict) and "answer" in chunk:
                full_response += chunk["answer"]

            time.sleep(0.01) # Simulate typing for better user experience
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
    st.session_state.messages.append({"role": "assistant", "content": full_response})
