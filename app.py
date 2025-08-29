import streamlit as st
import os

# This is a patch to ensure the correct sqlite3 version is used
# for chromadb compatibility. It must be at the very top of the script.
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from rag_system import get_retrieval_chain

# --- Set page configuration for a beautiful UI ---
st.set_page_config(
    page_title="Healthwise RAG",
    page_icon="⚕️",
    layout="centered",
)

# Custom CSS for a beautiful, clean UI
st.markdown("""
<style>
    /* Main container and title styling */
    .stApp {
        background-color: #f0f2f6;
        color: #1f2937;
    }
    .st-emotion-cache-12ms92i {
        background-color: #ffffff;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        padding: 30px;
        margin-bottom: 20px;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Streamlit chat message styling */
    .st-emotion-cache-1c7y2c1 { /* User message container */
        background-color: #d1fae5;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }
    .st-emotion-cache-1ch051d { /* Assistant message container */
        background-color: #ffffff;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
    }

    /* Title and header styling */
    h1 {
        color: #0b9389;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    h3 {
        color: #4b5563;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        text-align: center;
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# Set page title and header
st.title("Healthwise RAG ⚕️")
st.markdown("### Your reliable source for public health information")

# Add a text input for the Hugging Face API token
api_token = st.text_input(
    "Enter your Hugging Face API Token:", type="password", help="You can find your token in your Hugging Face settings."
)
if api_token:
    # Use st.session_state to store the token and trigger a rerun
    st.session_state["HUGGINGFACEHUB_API_TOKEN"] = api_token

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main application logic ---
try:
    with st.spinner('Preparing the knowledge base...'):
        # Check if retrieval chain is already in session state
        if "retrieval_chain" not in st.session_state:
            # Check for Hugging Face token
            current_token = st.session_state.get("HUGGINGFACEHUB_API_TOKEN")
            if not current_token:
                st.error(
                    "Error: The HUGGINGFACEHUB_API_TOKEN environment variable is not set. "
                    "Please enter your token above and click Enter."
                )
            else:
                st.session_state.retrieval_chain = get_retrieval_chain(current_token)

    # Get the retrieval chain from session state
    retrieval_chain = st.session_state.get("retrieval_chain")

    # Only proceed if the retrieval chain has been successfully initialized
    if retrieval_chain:
        # Accept user input
        if user_query := st.chat_input("Ask a question about public health..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": user_query})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Searching for answers..."):
                    response = retrieval_chain.invoke({"input": user_query})
                    # Check for a valid response
                    if response and "answer" in response:
                        st.markdown(response["answer"])
                        st.session_state.messages.append(
                            {"role": "assistant", "content": response["answer"]}
                        )
                    else:
                        st.warning("Sorry, I could not find a relevant answer.")

except Exception as e:
    # Display a user-friendly error message
    st.error(f"An error occurred during setup: {e}")
    st.warning(
        "Please check your environment variables, especially your Hugging Face API token, "
        "and ensure you have an active internet connection."
    )
