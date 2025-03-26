import os
import streamlit as st
import tempfile
import time

from rag.utils.utils import load_env
from rag.indexing import indexing_single
from rag.inference import inference
from rag.utils.logger import LOG_FILE


def read_logs():
    """Read logs from the log file."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as file:
            return file.readlines()
    except FileNotFoundError:
        return ["Log file not found. Please check if the log file exists."]
    except Exception as e:
        return [f"Error reading logs: {str(e)}"]


def auto_refresh_logs():
    """Auto refresh logs if auto-refresh is enabled."""
    if st.session_state.get("auto_refresh", False):
        st.experimental_rerun()


def main():
    st.title("Chat AGH development")

    load_env()

    default_collection = os.environ.get("QDRANT_COLLECTION_NAME", "default_collection")
    tab1, tab2, tab3 = st.tabs(["Indexing", "Inference", "Logs"])

    with tab1:
        st.header("Document Indexing")

        collection_name = st.text_input("Collection Name", value=default_collection)
        index_method = st.radio("Choose data source method:", ("Upload File", "Specify Directory Path"))
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
                        num_chunks = indexing_single(file_path, collection_name, chunk_size, chunk_overlap)
                        st.success(f"Successfully indexed {num_chunks} chunks into collection '{collection_name}'")
                    except Exception as e:
                        st.error(f"Error during indexing: {str(e)}")
                    finally:
                        if delete_after:
                            os.unlink(file_path)
            else:
                st.error("Please select a file to index")

    with tab2:
        st.header("Inference")
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

    with tab3:
        st.header("Application Logs")

        log_lines = read_logs()

        filter_options = ["All Logs", "INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        selected_filter = st.selectbox("Filter logs by level:", filter_options)

        search_term = st.text_input("Search logs:", "")

        filtered_logs = []
        for line in log_lines:
            if selected_filter != "All Logs" and selected_filter not in line:
                continue
            if search_term and search_term.lower() not in line.lower():
                continue
            filtered_logs.append(line)

        st.text_area("Log Output", value="".join(filtered_logs), height=400, disabled=True)

        st.download_button(
            label="Download Logs",
            data="".join(filtered_logs),
            file_name=f"ChatAGH_logs_{time.strftime('%Y-%m-%d_%H-%M-%S')}.log",
            mime="text/plain"
        )


if __name__ == "__main__":
    main()