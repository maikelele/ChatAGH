import os
from dotenv import load_dotenv

from rag.models.google_model import GoogleModel
from rag.vector_stores.qdrant_hybrid_search import QdrantHybridSearchVectorStore

ENV_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH/config/.env"
VECTOR_STORE_PATH = "/Users/wnowogorski/PycharmProjects/CHAT_AGH"


def inference(query, collection_name):
    """
    Run inference on the given query using the specified collection
    """
    vector_store = QdrantHybridSearchVectorStore(collection_name=collection_name)
    source_docs = vector_store.search(query)

    model = GoogleModel()
    response = model.generate_from_documents(query, source_docs)

    return response, source_docs


if __name__ == "__main__":
    load_dotenv(dotenv_path=ENV_PATH)

    query = "Jak mogę dokonac płatności?"

    inference(query, collection_name=os.getenv("QDRANT_COLLECTION_NAME"))
