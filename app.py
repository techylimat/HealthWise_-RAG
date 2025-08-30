import streamlit as st
import os
import pysqlite3 as sqlite3
import sys
import traceback

# Set the pysqlite3 path for ChromaDB
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from rag_system import get_retrieval_chain

# --- UI elements ---
st.set_page_config(page_title=" ⚕️Healthwise RAG ", layout="wide")

# Custom CSS for a beautiful UI
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    h1 {
        color: #004d40;
        font-family: 'Inter', sans-serif;
    }
    .st-emotion-cache-1c5v41f {
        background-color: #e0f7fa;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .st-emotion-cache-1629p8f {
        font-family: 'Inter', sans-serif;
    }
    .user-message {
        background-color: #00796b;
        color: white;
        padding: 10px 15px;
        border-radius: 15px 15px 0 15px;
        margin-bottom: 10px;
    }
    .assistant-message {
        background-color: #cfd8dc;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 15px 15px 0;
        margin-bottom: 10px;
    }
    .st-emotion-cache-1629p8f {
        padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "retrieval_chain" not in st.session_state:
    st.session_state.retrieval_chain = None

# --- Main Logic ---
try:
    # Acknowledge that the pattern is for a different app type but we'll adapt the structure
    st.sidebar.title("Healthwise RAG ⚕️")
    st.sidebar.write("Your reliable source for public health information.")

    if st.session_state.retrieval_chain is None:
        with st.sidebar:
            with st.spinner("⏳ Setting up RAG system..."):
                st.session_state.retrieval_chain = get_retrieval_chain()
        if st.session_state.retrieval_chain is None:
            st.error("RAG system failed to initialize. Please check the logs.")
        else:
            st.sidebar.success("✅ RAG system is ready!")

    st.header("💬 Ask a question")

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What is diabetics"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Ensure retrieval chain is available before proceeding
        if st.session_state.retrieval_chain is None:
            with st.chat_message("assistant"):
                st.error("LLM connection failed. Please try again later.")
            st.session_state.messages.append({"role": "assistant", "content": "LLM connection failed. Please try again later."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching for answers..."):
                    try:
                        print("DEBUG: Attempting to invoke retrieval chain...")
                        response = st.session_state.retrieval_chain.invoke({"query": prompt})
                        result = response.get("result", "Sorry, I could not find a relevant answer.")
                        st.markdown(result)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        print("DEBUG: Response generation successful.")
                    except Exception as e:
                        error_message = f"An error occurred during response generation: {e}"
                        st.error(error_message)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        print("DEBUG: An error occurred during response generation.")
                        print(traceback.format_exc()) # This will print the full traceback to the console
            
except Exception as e:
    st.error(f"An error occurred during app execution: {e}")
