
import streamlit as st
import utils as utils
import os
from dotenv import load_dotenv

# Load environment variables if local
local = False
if local:
    load_dotenv()

# Set page configuration
st.set_page_config(page_title="Chat with Your Codebase", layout="wide")

# Inject custom CSS
st.markdown(
    """
    <style>
    /* Gradient background */
    body {
        background: linear-gradient(to right, #fbc2eb, #a6c0fe);
        font-family: 'Arial', sans-serif;
    }
    
    /* Custom font for title */
    .title {
        font-family: 'Roboto', sans-serif;
        font-size: 2.5em;
        color: #333;
    }
    
    /* Style chat messages */
    .chat-message {
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
        background-color: #fff;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    
    .user {
        background-color: #d1e7dd;
    }
    
    .assistant {
        background-color: #f8d7da;
    }
    
    /* Style buttons and input fields */
    .css-ffhzg2, .css-1r6slb0 {
        border-radius: 8px;
        padding: 10px;
        font-size: 1em;
        border: 1px solid #ddd;
    }
    
    .css-ffhzg2:hover, .css-1r6slb0:hover {
        background-color: #e2e2e2;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Page title and description
st.markdown('<h1 class="title">CodeQuestAI</h1>', unsafe_allow_html=True)
st.write("Welcome! This application allows you to interact with your codebase through a chat interface.")

# User input for API key and GitHub link
if local:
    api_key_input = os.environ.get('GOOGLE_API_KEY')
else:
    api_key_input = st.text_input("Enter your GEMINI Key", type="password")
repo_url_input = st.text_input("Enter GitHub Link to your public codebase", "")

# Check if API key and repo URL are provided
if api_key_input and repo_url_input:
    os.environ['GOOGLE_API_KEY'] = api_key_input

    # Initialize Embedder
    embedder = utils.Embedder(repo_url_input)

    
    with st.spinner("Cloning and processing repository..."):
        embedder.clone_repo()
    with st.spinner("Processing Repo"):
        embedder.load_db()
    st.success("Repository loaded successfully. You can now ask questions about your codebase.")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="chat-message {message["role"]}">{message["content"]}</div>', unsafe_allow_html=True)

    # Accept user input for questions
    prompt = st.chat_input("Type your question here:")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message user">{prompt}</div>', unsafe_allow_html=True)
        # Fetch response from Embedder
        response = embedder.retrieve_results(prompt)
        with st.chat_message("assistant"):
            st.markdown(f'<div class="chat-message assistant">{response}</div>', unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response})

else:
    if not api_key_input:
        st.warning("Please enter your Google API Key.")
    if not repo_url_input:
        st.warning("Please enter a GitHub repository link.")