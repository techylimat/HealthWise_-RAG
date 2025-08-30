import streamlit as st
import os
import pysqlite3 as sqlite3
import sys
import traceback

# Set the pysqlite3 path for ChromaDB
sys.modules["sqlite3"] = sys.modules["pysqlite3"]

from rag_system import get_retrieval_chain

# --- UI elements ---
st.set_page_config(page_title=" ⚕️Healthwise RAG", layout="wide")

# Custom CSS for a beautiful UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, .stApp {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Inter', sans-serif;
    }
    .stApp > header {
        background-color: #1a1a1a;
        padding: 0.5rem 1rem;
    }
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    .st-emotion-cache-1c5v41f {
        background-color: #1a1a1a;
        border-radius: 15px;
        padding: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    
    .st-emotion-cache-1629p8f {
        padding-top: 1rem;
    }

    /* Streamlit's chat message containers */
    .stChatMessage {
        background-color: transparent !important;
        border: none !important;
        padding: 0.5rem;
    }

    /* Specific user and assistant message styling */
    .user-message-bubble {
        background-color: #004d40;
        color: #ffffff;
        padding: 12px 18px;
        border-radius: 20px 20px 5px 20px;
        display: inline-block;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .assistant-message-bubble {
        background-color: #263238;
        color: #e0e0e0;
        padding: 12px 18px;
        border-radius: 20px 20px 20px 5px;
        display: inline-block;
        max-width: 80%;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2);
    }
    .st-emotion-cache-6q9r41 {
        background-color: #1a1a1a;
        color: #e0e0e0;
        border-radius: 25px;
        border: 1px solid #444;
        padding: 1rem 1.5rem;
    }
    .st-emotion-cache-13vn43k {
        background-color: #00796b;
        color: white;
        border-radius: 25px;
        padding: 0.5rem 1rem;
    }
    .st-emotion-cache-1c5v41f {
        border-radius: 20px;
    }
    .st-emotion-cache-1ghh0z9 {
        padding: 0.5rem;
        background-color: #263238;
        border-radius: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
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

    st.markdown("<h1 style='text-align: center; color: #4CAF50; padding: 1rem;'>💬 Chat with Healthwise</h1>", unsafe_allow_html=True)

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(f"<div class='user-message-bubble'>{message['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='assistant-message-bubble'>{message['content']}</div>", unsafe_allow_html=True)

    # Accept user input
    if prompt := st.chat_input("What is diabetics"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(f"<div class='user-message-bubble'>{prompt}</div>", unsafe_allow_html=True)

        # Ensure retrieval chain is available before proceeding
        if st.session_state.retrieval_chain is None:
            with st.chat_message("assistant"):
                st.markdown("<div class='assistant-message-bubble'>LLM connection failed. Please try again later.</div>", unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": "LLM connection failed. Please try again later."})
        else:
            with st.chat_message("assistant"):
                with st.spinner("Searching for answers..."):
                    try:
                        print("DEBUG: Attempting to invoke retrieval chain...")
                        response = st.session_state.retrieval_chain.invoke({"query": prompt})
                        result = response.get("result", "Sorry, I could not find a relevant answer.")
                        st.markdown(f"<div class='assistant-message-bubble'>{result}</div>", unsafe_allow_html=True)
                        # Add assistant response to chat history
                        st.session_state.messages.append({"role": "assistant", "content": result})
                        print("DEBUG: Response generation successful.")
                    except Exception as e:
                        error_message = f"An error occurred during response generation: {e}"
                        st.markdown(f"<div class='assistant-message-bubble'>{error_message}</div>", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": error_message})
                        print("DEBUG: An error occurred during response generation.")
                        print(traceback.format_exc()) # This will print the full traceback to the console
            
except Exception as e:
    st.error(f"An error occurred during app execution: {e}")
