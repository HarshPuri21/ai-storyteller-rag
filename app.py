# app.py
# This script creates a simple web interface for the AI Storyteller
# using Streamlit, allowing users to interact with the RAG pipeline.

import streamlit as st
from storyteller_rag import create_rag_pipeline # Import the pipeline builder

# --- 1. App Configuration and State Management ---

# Set the page title and layout
st.set_page_config(page_title="AI Storyteller", layout="wide")

# Use Streamlit's session state to cache the RAG pipeline.
# This prevents the time-consuming process of building the pipeline
# from scratch every time the user interacts with the app.
if 'rag_chain' not in st.session_state:
    with st.spinner("Warming up the AI Storyteller... (this may take a moment)"):
        # Build and store the pipeline in the session state
        st.session_state.rag_chain = create_rag_pipeline()

# --- 2. User Interface ---

# Header
st.title("📚 AI Storyteller for Cultural Narratives")
st.markdown("""
This tool uses a Retrieval-Augmented Generation (RAG) pipeline to write creative stories based on a custom knowledge base of cultural folklore and mythology. 
Enter a prompt below to get started!
""")

# Example Prompts
st.subheader("Example Prompts")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("A story about a trickster spider from Africa"):
        st.session_state.prompt = "A story about a trickster spider from Africa"
with col2:
    if st.button("A tale of a hidden city of gold"):
        st.session_state.prompt = "A tale of a hidden city of gold"
with col3:
    if st.button("A myth about the Norse world tree"):
        st.session_state.prompt = "A myth about the Norse world tree"


# User Input
st.subheader("Your Story Request")
prompt = st.text_input("Enter your story idea here:", key="prompt", placeholder="e.g., Tell me a story about a mythical fox from Japan.")

# Generate Story Button
if st.button("Generate Story"):
    if prompt:
        # If there's a prompt, invoke the RAG chain
        with st.spinner("The AI is writing your story..."):
            try:
                # Get the LangChain RAG pipeline from the session state
                storyteller = st.session_state.rag_chain
                
                # Invoke the chain to get the result
                result = storyteller.invoke(prompt)
                
                # Store the result in the session state to display it
                st.session_state.generated_story = result

            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a story idea.")

# --- 3. Display Output ---

# Display the generated story if it exists in the session state
if 'generated_story' in st.session_state and st.session_state.generated_story:
    st.subheader("Your Generated Story")
    st.markdown(f"> {st.session_state.generated_story}")
    
    # Add a feedback section
    st.markdown("---")
    st.write("Was this story helpful and relevant?")
    feedback_col1, feedback_col2, _ = st.columns([1,1,5])
    with feedback_col1:
        if st.button("👍 Yes"):
            st.success("Thank you for your feedback!")
    with feedback_col2:
        if st.button("👎 No"):
            st.success("Thank you for your feedback!")

