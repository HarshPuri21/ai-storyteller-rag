# storyteller_rag.py
# This script builds the core Retrieval-Augmented Generation (RAG) pipeline
# for the AI Storyteller using Python, LangChain, and a vector database.

import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional

# --- 1. Custom Knowledge Base ---
# In a real project, this would be a large collection of documents (txt, pdf, etc.)
# loaded from a directory. For this demonstration, we use a list of strings
# representing our curated cultural narratives.
knowledge_base_texts = [
    "In Japanese folklore, Kitsune are intelligent foxes that possess paranormal abilities that increase with their age and wisdom. They are known for having multiple tails—up to nine.",
    "The story of Momotarō (Peach Boy) is a famous Japanese folktale about a boy born from a giant peach who goes on to defeat a band of oni (demons) with the help of his animal companions: a dog, a monkey, and a pheasant.",
    "In ancient Greek mythology, the Trojan War was a legendary conflict between the Achaeans (Greeks) and the city of Troy. It is most famously known for the tale of the Trojan Horse, a wooden horse used by the Greeks to enter the city.",
    "Anansi the Spider is a popular Akan folktale character from West Africa. He is a trickster god who often triumphs over more powerful opponents through his cunning and wit, and is credited with giving humanity wisdom.",
    "The legend of El Dorado in South America speaks of a lost city of immense wealth, hidden deep in the Amazon rainforest. Many explorers have searched for it, but it remains a myth.",
    "Norse mythology features a world tree called Yggdrasil, which connects the Nine Worlds. At its roots lives the dragon Níðhöggr, and an eagle sits at its top."
]

# --- 2. Custom LLM Wrapper for API Simulation ---
# This class simulates a call to an LLM API (like OpenAI) for use within LangChain.
class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        # This is where a real API call would be made.
        # We simulate a response based on the prompt.
        print("\n--- LLM PROMPT ---")
        print(prompt)
        print("--------------------\n")
        if "Kitsune" in prompt:
            return "Once upon a time, in a moonlit forest, a young fox named Kiko discovered she had grown a second tail, a sign of her growing magical abilities. She decided to use her newfound powers of illusion to protect the nearby village from mischievous spirits."
        elif "Trojan Horse" in prompt:
            return "In the final days of a long and bitter war, the clever strategist Odysseus proposed a daring plan. The invading army would build a great wooden horse as a supposed offering, but inside, the city's greatest heroes would hide, ready to open the gates from within."
        else:
            return "A lone traveler, guided by an ancient map, sought a legendary hidden city deep within a dense, uncharted jungle. The journey was perilous, filled with ancient traps and mythical creatures, but the promise of discovery drove the traveler forward."

# --- 3. Building the RAG Pipeline ---

def create_rag_pipeline():
    """
    Creates and returns a LangChain RAG pipeline.
    """
    print("Building RAG pipeline...")

    # a. Create Embeddings Model
    # We use a local, open-source model from Hugging Face to turn text into vectors.
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # b. Create Vector Store
    # FAISS is a library for efficient similarity search. We create an in-memory
    # vector store from our knowledge base.
    print("Creating vector store from knowledge base...")
    vectorstore = FAISS.from_texts(texts=knowledge_base_texts, embedding=embeddings)

    # c. Create Retriever
    # The retriever's job is to fetch the most relevant documents from the vector store
    # based on the user's query.
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) # Retrieve top 2 most relevant docs

    # d. Engineer the Prompt Chain using a PromptTemplate
    # This template structures the input for the LLM, combining the retrieved context
    # with the user's original question.
    template = """
    You are an AI storyteller specializing in cultural narratives.
    Use the following retrieved context to help you write a short, creative story based on the user's request.
    The story should be engaging, culturally relevant, and have a clear beginning, middle, and end.

    CONTEXT:
    {context}

    USER'S REQUEST:
    {question}

    YOUR STORY:
    """
    prompt_template = PromptTemplate.from_template(template)

    # e. Initialize the LLM
    llm = MockLLM()

    # f. Chain Everything Together
    # This is the final LangChain Expression Language (LCEL) chain.
    # It defines the flow of data:
    # 1. The user's question is passed to the retriever.
    # 2. The retriever's output (context) and the original question are passed to the prompt template.
    # 3. The formatted prompt is passed to the LLM.
    # 4. The LLM's output is parsed into a string.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )
    
    print("RAG pipeline built successfully.")
    return rag_chain

# --- 4. Main Execution ---
if __name__ == "__main__":
    storyteller_chain = create_rag_pipeline()

    print("\n--- Invoking Storyteller ---")
    user_request = "Tell me a story about a mythical fox from Japan."
    print(f"User Request: {user_request}")
    
    # Invoke the chain with the user's request
    generated_story = storyteller_chain.invoke(user_request)

    print("\n--- Generated Story ---")
    print(generated_story)
    print("------------------------")
    
    print("\n--- Invoking Storyteller Again ---")
    user_request_2 = "Write a short tale about a clever trick involving a wooden horse in an ancient war."
    print(f"User Request: {user_request_2}")
    generated_story_2 = storyteller_chain.invoke(user_request_2)
    print("\n--- Generated Story ---")
    print(generated_story_2)
    print("------------------------")

