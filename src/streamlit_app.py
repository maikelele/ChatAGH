import os
import streamlit as st
import tempfile

from rag.utils.utils import load_env
from rag.indexing import indexing
from rag.inference import inference


def main():
    st.title("Chat AGH development")

    load_env()

    default_collection = os.environ.get("QDRANT_COLLECTION_NAME", "default_collection")

    tab1, tab2 = st.tabs(["Indexing", "Inference"])

    with tab1:
        st.header("Document Indexing")

        collection_name = st.text_input("Collection Name", value=default_collection)

        index_method = st.radio(
            "Choose data source method:",
            ("Upload File", "Specify Directory Path")
        )

        file_path = None

        if index_method == "Upload File":
            uploaded_file = st.file_uploader("Choose a JSON file", type="json")
            if uploaded_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    file_path = tmp_file.name
        else:
            file_path = st.text_input("Enter directory path containing JSON files")

        col1, col2 = st.columns(2)
        chunk_size = col1.number_input("Chunk Size", value=1000, min_value=100, max_value=10000)
        chunk_overlap = col2.number_input("Chunk Overlap", value=100, min_value=0, max_value=chunk_size - 1)

        if st.button("Process and Index"):
            if file_path:
                delete_after = index_method == "Upload File"

                with st.spinner("Indexing documents..."):
                    try:
                        num_chunks = indexing(file_path, collection_name, chunk_size, chunk_overlap)
                        st.success(f"Successfully indexed {num_chunks} chunks into collection '{collection_name}'")
                    except Exception as e:
                        st.error(f"Error during indexing: {str(e)}")
                    finally:
                        if delete_after:
                            os.unlink(file_path)
            else:
                st.error("Please select a file to index")

    with tab2:
        st.header("Query Inference")

        # collection_name = st.text_input("Collection Name for Inference", value=default_collection)

        query = st.text_area("Enter your query", height=100)

        if st.button("Run Inference"):
            if query:
                with st.spinner("Running inference..."):
                    try:
                        response, source_docs = inference(query)

                        st.subheader("Model Response")
                        st.write(response)

                        st.subheader("Retrieved Documents")
                        for i, doc in enumerate(source_docs):
                            with st.expander(f"Document {i + 1}"):
                                st.write(doc)
                    except Exception as e:
                        st.error(f"Error during inference: {str(e)}")
            else:
                st.error("Please enter a query")


if __name__ == "__main__":
    main()