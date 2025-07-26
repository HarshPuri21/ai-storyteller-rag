AI Storyteller for Cultural Narratives using RAG and LangChain

This project is an advanced AI application that generates culturally relevant stories by implementing a Retrieval-Augmented Generation (RAG) pipeline. It leverages a custom knowledge base, the LangChain framework for prompt engineering, and a Streamlit web interface for user interaction and evaluation.

This project demonstrates a deep understanding of the modern AI development lifecycle, from data curation and pipeline construction to user-centered deployment.
Architecture: Retrieval-Augmented Generation (RAG)

The core of this project is the RAG pipeline, which enhances the creativity of a Large Language Model (LLM) by providing it with relevant, factual information from a custom knowledge base at inference time.

    Knowledge Base: A curated collection of documents containing information on cultural folklore, myths, and legends.

    Embedding & Vector Store: The documents in the knowledge base are converted into numerical representations (embeddings) using a sentence-transformer model. These embeddings are stored in a FAISS vector store, which allows for highly efficient similarity searches.

    Retriever: When a user provides a prompt (e.g., "a story about a mythical fox"), the retriever searches the vector store for the most semantically similar documents from the knowledge base.

    Prompt Engineering (LangChain): A sophisticated prompt template is used to combine the user's original query with the retrieved context from the knowledge base.

    LLM Generation: This combined prompt is then sent to the LLM, which uses both its inherent creativity and the provided factual context to generate a high-quality, relevant, and culturally-aware story.

Project Structure

This repository contains two main scripts:

    storyteller_rag.py: This script builds the core RAG pipeline using LangChain.

        Defines the custom knowledge base.

        Initializes the embedding model and creates the FAISS vector store.

        Constructs the LangChain Expression Language (LCEL) chain, which links the retriever, prompt template, and LLM together.

    app.py: This script creates an interactive web application using Streamlit.

        It imports and initializes the RAG pipeline from storyteller_rag.py.

        It uses st.session_state to cache the pipeline, preventing it from reloading on every interaction.

        Provides a simple UI for users to enter a story prompt and view the AI-generated narrative.

        Includes a feedback mechanism to demonstrate user-centered evaluation.

How to Run
Prerequisites

You will need a Python environment with the following libraries installed:

pip install streamlit langchain langchain-community faiss-cpu sentence-transformers

1. Run the Streamlit Application

    Ensure both storyteller_rag.py and app.py are in the same directory.

    From your terminal, run the Streamlit app:

    streamlit run app.py

    This will open a new tab in your web browser with the AI Storyteller interface.

2. How to Use

    Wait for the initial "Warming up" process to complete as the RAG pipeline is built.

    Enter a story idea into the text box (e.g., "a tale about a hidden city of gold").

    Click "Generate Story" and wait for the AI to write and display the narrative.
