import os

from dotenv import load_dotenv
from rag.utils.utils import load_data
from rag.chunkers.langchain_chunker import LangChainChunker
from rag.embeddings.google_embeddings import GoogleEmbeddings
from rag.vector_store.qdrant_hybrid_search import QdrantHybridSearchVectorStore

ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
DATA_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/data_2/agh_edu"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"


def indexing(data_path, collection_name, chunk_size=1000, chunk_overlap=100):
    """
    Index documents from the given data path into the vector store
    """
    load_dotenv(dotenv_path=ENV_PATH)
    data = load_data(data_path)

    chunker = LangChainChunker(chunk_size, chunk_overlap, remove_duplicates=True)
    chunks = chunker.chunk(data)
    print(len(chunks))

    vector_store = QdrantHybridSearchVectorStore(collection_name=collection_name)
    vector_store.index(chunks)

    return len(chunks)

if __name__ == "__main__":
    indexing(DATA_PATH, os.environ.get("QDRANT_COLLECTION_NAME"))
