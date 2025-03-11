import streamlit as st
from dotenv import load_dotenv

from rag.embeddings.google_embeddings import GoogleEmbeddings
from rag.retrievers.primitive_retriever import PrimitiveRetriever
from rag.vector_stores.chroma_vector_store import ChromaVectorStore
from rag.models.google_model import GoogleModel
import os

# Configuration paths
ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"

# Load environment variables
load_dotenv(dotenv_path=ENV_PATH)


def setup_rag_inference():
    """Initialize the RAG components"""
    embeddings = GoogleEmbeddings()

    vector_store = ChromaVectorStore(
        collection_name="agh_edu",
        persist_directory=VECTOR_STORE_PATH,
        embedding_function=embeddings
    )

    retriever = PrimitiveRetriever(vector_store=vector_store)
    model = GoogleModel()

    return retriever, model


def main():
    st.set_page_config(page_title="RAG Q&A System", layout="wide")

    st.title("RAG Q&A System")
    st.write("Ask a question and get an answer based on retrieved documents")

    # Initialize session state for history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Initialize RAG components
    retriever, model = setup_rag_inference()

    # User input
    query = st.text_input("Wprowadź pytanie:", key="user_question")

    if st.button("Wyślij pytanie"):
        if query:
            with st.spinner("Przetwarzanie pytania..."):
                # Retrieve documents
                source_docs = retriever.invoke(query)

                # Generate response
                response = model.generate_from_documents(query, source_docs)

                # Store in session state
                st.session_state.history.append({
                    "query": query,
                    "source_docs": source_docs,
                    "response": response
                })

    # Display history
    if st.session_state.history:
        latest_interaction = st.session_state.history[-1]

        st.header("Wyniki")

        # Display the query
        st.subheader("Pytanie:")
        st.write(latest_interaction["query"])

        # Two-column layout for results
        col1, col2 = st.columns(2)

        # Display response in first column
        with col1:
            st.subheader("Odpowiedź modelu:")
            st.write(latest_interaction["response"])

        # Display source documents in second column
        with col2:
            st.subheader("Źródłowe dokumenty:")
            for i, doc in enumerate(latest_interaction["source_docs"]):
                with st.expander(f"Dokument {i + 1}"):
                    st.write(doc.page_content)
                    st.write("---")
                    st.write(f"Źródło: {doc.metadata.get('source', 'Unknown')}")

        # Show history
        if len(st.session_state.history) > 1:
            with st.expander("Historia pytań"):
                for i, interaction in enumerate(st.session_state.history[:-1]):
                    st.write(f"**Pytanie {i + 1}:** {interaction['query']}")
                    st.write(f"**Odpowiedź:** {interaction['response'][:100]}...")
                    st.write("---")

if __name__ == "__main__":
    main()